"""CLI to run Section 4.2.2 local unsupervised content metrics on MPDD pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.local_content_unsupervised.dialogue_act import (  # noqa: E402
    DialogueActTagger,
    RuleBasedDialogueActTagger,
    Seq2SeqDialogueActTagger,
    TransformersDialogueActTagger,
    train_transition_model,
)
from src.local_content_unsupervised.lexical import (  # noqa: E402
    LexicalFlowConfig,
    WordEmbeddings,
    flowing_weighted_snr,
)
from src.local_content_unsupervised.message import (  # noqa: E402
    CachedSentenceEmbedder,
    SentenceEmbedder,
    message_semantic_novelty,
)
from src.local_content_unsupervised.loglikelihood import (  # noqa: E402
    CausalLMScorer,
    build_ll_prompt_ids,
    load_context_token_cache_bundle,
    save_context_token_cache,
)
from src.local_content_unsupervised.types import to_turn_records  # noqa: E402
from src.local_content_unsupervised.topic_expansion import (  # noqa: E402
    BERTopicTheta,
    load_tes_context_cache,
    save_tes_context_cache,
    topic_expansion_score,
)

DEFAULT_OUTPUT_DIR = Path("src/local_content_unsupervised/output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=Path("data/mpdd_local_content_unsupervised.jsonl"),
        help="Path to processed (context, next_message) JSONL file.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Evaluate one example or all examples.",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "all",
            "lexical_weighted_snr",
            "message_semantic_novelty_min",
            "message_semantic_novelty_avg",
            "dialogue_act_transition_fit",
            "flowing_ll",
            "topic_expansion_score",
        ],
        default="all",
        help="Metric to run; use 'all' to compute the default metric set.",
    )
    parser.add_argument(
        "--dialogue-id",
        type=str,
        default=None,
        help="Select a specific dialogue_id in single mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when sampling in single mode.",
    )

    # Lexical weighted SNR
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help="Similarity threshold (tau) for lexical novelty filtering.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["pkuseg", "jieba", "english", "spacy", "english_regex"],
        default="pkuseg",
        help="Tokenizer backend used to extract lexical terms (spaCy-backed for English).",
    )
    parser.add_argument(
        "--embeddings-kv",
        type=Path,
        default=Path("data/embeddings/cc.zh.300.kv"),
        help="Path to word vectors: prefer gensim `.kv`, but `.vec`/`.vec.gz` also works (slower).",
    )
    parser.add_argument(
        "--embeddings-limit",
        type=int,
        default=None,
        help="When using `.vec`/`.vec.gz`, optionally load only the first N vectors (faster, less coverage).",
    )
    parser.add_argument(
        "--idf-scope",
        choices=["pair", "corpus"],
        default="pair",
        help="IDF weighting scope: per-pair or corpus-wide (loaded from --idf-path).",
    )
    parser.add_argument(
        "--idf-path",
        type=Path,
        default=Path("data/mpdd_local_content_idf.json"),
        help="Path to a JSON IDF map (only used when --idf-scope corpus).",
    )

    # Message semantic novelty
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Context window size k for message-level novelty (default: all context turns).",
    )
    parser.add_argument(
        "--sentence-embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name/path for message-level semantic novelty.",
    )
    parser.add_argument(
        "--sentence-embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for sentence embedding encoding.",
    )
    parser.add_argument(
        "--sentence-embedding-cache",
        action="store_true",
        help="Enable an in-memory LRU cache for message embeddings (batch speedup).",
    )
    parser.add_argument(
        "--sentence-embedding-cache-size",
        type=int,
        default=20000,
        help="Max number of messages to cache when --sentence-embedding-cache is set.",
    )

    # Dialogue act transition fit (DAF)
    parser.add_argument(
        "--dialogue-act-tagger",
        choices=["rule_based", "hf", "crosswoz_mt5"],
        default="rule_based",
        help="Dialogue act tagger backend.",
    )
    parser.add_argument(
        "--dialogue-act-model",
        type=str,
        default=None,
        help="HF model name/path for --dialogue-act-tagger hf or crosswoz_mt5.",
    )
    parser.add_argument(
        "--dialogue-act-local-files-only",
        action="store_true",
        help="Require the dialogue act model to be available locally (no network).",
    )
    parser.add_argument(
        "--dialogue-act-device",
        type=int,
        default=-1,
        help="CUDA device index for DA tagger; use -1 for CPU.",
    )
    parser.add_argument(
        "--dialogue-act-batch-size",
        type=int,
        default=32,
        help="Batch size for DA tagger inference.",
    )
    parser.add_argument(
        "--dialogue-act-max-length",
        type=int,
        default=256,
        help="Max token length for DA tagger input truncation.",
    )
    parser.add_argument(
        "--dialogue-act-max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for seq2seq DA generation (crosswoz_mt5).",
    )
    parser.add_argument(
        "--dialogue-act-num-beams",
        type=int,
        default=1,
        help="Beam size for seq2seq DA generation (crosswoz_mt5).",
    )
    parser.add_argument(
        "--dialogue-act-m",
        type=int,
        default=2,
        help="Number of previous utterances to include when tagging each act.",
    )
    parser.add_argument(
        "--daf-order",
        type=int,
        default=4,
        help="N-gram order for DAF transition model (history length).",
    )
    parser.add_argument(
        "--daf-alpha",
        type=float,
        default=0.1,
        help="Add-alpha smoothing parameter for DAF transition probabilities.",
    )
    parser.add_argument(
        "--daf-acts-cache",
        type=Path,
        default=Path("data/mpdd_dialogue_acts.json"),
        help="Cache file for dialogue act sequences (JSON mapping dialogue_id -> acts).",
    )
    parser.add_argument(
        "--daf-refresh-acts-cache",
        action="store_true",
        help="Recompute dialogue act sequences even if the cache file exists.",
    )
    parser.add_argument(
        "--daf-train-limit",
        type=int,
        default=None,
        help="Cap number of dialogues used to train the DAF transition model (for quick smoke tests).",
    )

    # Log-likelihood flow (Flowing_LL)
    parser.add_argument(
        "--ll-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF model name/path for Flowing_LL (causal LM).",
    )
    parser.add_argument(
        "--ll-local-files-only",
        action="store_true",
        help="Require the Flowing_LL model/tokenizer to be available locally (no network).",
    )
    parser.add_argument(
        "--ll-trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for Flowing_LL model/tokenizer loading.",
    )
    parser.add_argument(
        "--ll-device",
        type=int,
        default=-1,
        help="CUDA device index for Flowing_LL; use -1 for CPU.",
    )
    parser.add_argument(
        "--ll-max-length",
        type=int,
        default=None,
        help="Optional max sequence length override for Flowing_LL.",
    )
    parser.add_argument(
        "--ll-dtype",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Compute dtype for Flowing_LL model weights (on GPU, auto usually picks bf16/fp16).",
    )
    parser.add_argument(
        "--ll-context-cache",
        type=Path,
        default=Path("data/mpdd_ll_context_tokens_qwen3_8b.json"),
        help="On-disk cache for tokenized contexts (dialogue_id -> token ids).",
    )
    parser.add_argument(
        "--ll-refresh-context-cache",
        action="store_true",
        help="Recompute context token cache even if it exists.",
    )
    parser.add_argument(
        "--ll-next-tag",
        type=str,
        default="Next:",
        help="Fixed next-turn tag appended to the prompt (not scored).",
    )
    parser.add_argument(
        "--ll-scale-c",
        type=float,
        default=25.0,
        help="Monotonic rescaling constant c for reporting: ll_scaled = 1 - exp(-c * flowing_ll).",
    )
    parser.add_argument(
        "--ll-prompt-format",
        choices=["raw", "chat"],
        default="raw",
        help="Prompt format for Flowing_LL: raw speaker-tag transcript or model chat template.",
    )
    parser.add_argument(
        "--ll-system-prompt",
        type=str,
        default=None,
        help="Optional system prompt used when --ll-prompt-format chat.",
    )

    # Topic expansion score (TES)
    parser.add_argument(
        "--tes-model-path",
        type=Path,
        default=Path("data/mpdd_tes_bertopic_model_min5"),
        help="Path to load/save the BERTopic model used for TES.",
    )
    parser.add_argument(
        "--tes-refresh-model",
        action="store_true",
        help="Refit the BERTopic model even if --tes-model-path exists.",
    )
    parser.add_argument(
        "--tes-embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name/path used as BERTopic embedding model.",
    )
    parser.add_argument(
        "--tes-device",
        type=str,
        default="cpu",
        help="Device for TES embeddings: cpu or cuda.",
    )
    parser.add_argument(
        "--tes-min-topic-size",
        type=int,
        default=5,
        help="BERTopic min_topic_size.",
    )
    parser.add_argument(
        "--tes-nr-topics",
        type=int,
        default=None,
        help="Optional BERTopic nr_topics.",
    )
    parser.add_argument(
        "--tes-ell",
        type=int,
        default=2,
        help="Small context window ℓ for Q_p = θ(C^{-ℓ} ⊕ u_p).",
    )
    parser.add_argument(
        "--tes-up-repeat",
        type=int,
        default=0,
        help="Repeat u_p this many times when forming Q_p to reduce dilution; 0 means auto by length ratio.",
    )
    parser.add_argument(
        "--tes-up-repeat-cap",
        type=int,
        default=20,
        help="Cap for auto u_p repetition when --tes-up-repeat=0.",
    )
    parser.add_argument(
        "--tes-delta",
        type=int,
        default=2,
        help="Expansion step size Δ used when selecting k*.",
    )
    parser.add_argument(
        "--tes-rho",
        type=float,
        default=0.8,
        help="Dominant-topic mass threshold ρ for T_dom(P*,ρ).",
    )
    parser.add_argument(
        "--tes-jsd-tau",
        type=float,
        default=0.1,
        help="JSD threshold τ for stopping k* expansion (base-2 JSD in [0,1]).",
    )
    parser.add_argument(
        "--tes-context-cache",
        type=Path,
        default=Path("data/mpdd_tes_context_cache_min5.json"),
        help="Per-dialogue TES cache (k_star and dominant topic indices).",
    )
    parser.add_argument(
        "--tes-refresh-context-cache",
        action="store_true",
        help="Recompute TES per-dialogue cache even if it exists.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store batch evaluation outputs.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional base name for batch outputs. When set (and --scores-path/--summary-path are not set), "
            "writes <output_name>_scores.jsonl and <output_name>_summary.json under --output-dir."
        ),
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=None,
        help=(
            "Optional override for the per-pair scores JSONL output path. "
            "If a relative path is given, it is resolved relative to --output-dir."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help=(
            "Optional override for the summary JSON output path. "
            "If a relative path is given, it is resolved relative to --output-dir."
        ),
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def sample_single_record(
    path: Path, *, dialogue_id: Optional[str], seed: Optional[int]
) -> Dict[str, Any]:
    if dialogue_id is not None:
        for record in iter_jsonl(path):
            if str(record.get("dialogue_id", "")) == str(dialogue_id):
                return record
        raise ValueError(f"dialogue_id={dialogue_id} not found in {path}")

    rng = random.Random(seed)
    chosen: Optional[Dict[str, Any]] = None
    count = 0
    for record in iter_jsonl(path):
        count += 1
        if rng.randrange(count) == 0:
            chosen = record
    if chosen is None:
        raise RuntimeError(f"No records found in {path}")
    return chosen


def load_corpus_idf(idf_path: Path) -> Dict[str, float]:
    with idf_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return {str(k): float(v) for k, v in raw.items()}


def load_da_cache(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    cache: Dict[str, List[str]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, list):
                cache[str(k)] = [str(a) for a in v]
    return cache


def save_da_cache(path: Path, cache: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(cache, fp, ensure_ascii=False)


def ensure_dialogue_acts(
    pairs_path: Path,
    tagger: DialogueActTagger,
    *,
    m: int,
    cache: Dict[str, List[str]],
    refresh: bool,
    limit: int | None,
    ensure_ids: Sequence[str] = (),
    progress_desc: str = "Tagging dialogue acts",
) -> Dict[str, List[str]]:
    if refresh:
        cache.clear()

    ensure_set = {str(x) for x in ensure_ids if x is not None}

    selected: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for rec in iter_jsonl(pairs_path):
        did = str(rec.get("dialogue_id"))
        if not did or did in seen:
            continue
        seen.add(did)

        under_limit = limit is None or len(selected) < int(limit)
        if not (under_limit or did in ensure_set):
            continue

        ctx = [str(t.get("utterance", "")) for t in rec.get("context", [])]
        selected.append((did, ctx))

    to_tag = [(did, ctx) for (did, ctx) in selected if did not in cache]
    for did, ctx in tqdm(to_tag, desc=progress_desc, total=len(to_tag)):
        cache[did] = [str(a) for a in tagger.tag_conversation(ctx, m=m)]

    return {did: cache[did] for (did, _ctx) in selected if did in cache}


def _selected_metrics(args: argparse.Namespace) -> list[str]:
    if args.metric == "all":
        return [
            "lexical_weighted_snr",
            "message_semantic_novelty_min",
            "message_semantic_novelty_avg",
            "dialogue_act_transition_fit",
            "flowing_ll",
            "topic_expansion_score",
        ]
    return [args.metric]


def init_dialogue_act_tagger(args: argparse.Namespace) -> DialogueActTagger:
    if args.dialogue_act_tagger == "hf":
        if not args.dialogue_act_model:
            raise ValueError("--dialogue-act-model is required for --dialogue-act-tagger hf")
        return TransformersDialogueActTagger(
            args.dialogue_act_model,
            local_files_only=args.dialogue_act_local_files_only,
            batch_size=args.dialogue_act_batch_size,
            max_length=args.dialogue_act_max_length,
            device=args.dialogue_act_device,
        )

    if args.dialogue_act_tagger == "crosswoz_mt5":
        model = args.dialogue_act_model or "ConvLab/mt5-small-nlu-all-crosswoz"
        return Seq2SeqDialogueActTagger(
            model,
            local_files_only=args.dialogue_act_local_files_only,
            batch_size=max(1, args.dialogue_act_batch_size // 2),
            max_length=args.dialogue_act_max_length,
            max_new_tokens=args.dialogue_act_max_new_tokens,
            num_beams=args.dialogue_act_num_beams,
            device=args.dialogue_act_device,
        )

    return RuleBasedDialogueActTagger()


def score_record(
    record: Dict[str, Any],
    metrics: Sequence[str],
    args: argparse.Namespace,
    *,
    embeddings: WordEmbeddings | None,
    lexical_config: LexicalFlowConfig | None,
    sentence_embedder: SentenceEmbedder | CachedSentenceEmbedder | None,
    dialogue_act_tagger: DialogueActTagger | None,
    dialogue_act_model,
    ll_scorer: CausalLMScorer | None,
    ll_context_token_cache: Dict[str, List[int]] | None,
    tes_theta_model: BERTopicTheta | None,
    tes_context_cache: Dict[str, Dict[str, Any]] | None,
    corpus_idf: Dict[str, float] | None,
) -> Dict[str, float]:
    context_raw = record.get("context", [])
    next_message = str(record.get("next_message", ""))
    context_records = to_turn_records(context_raw)
    context_texts = [turn.utterance for turn in context_records]

    cached_acts = record.get("_cached_context_acts")

    results: Dict[str, float] = {}
    for metric in metrics:
        if metric == "lexical_weighted_snr":
            if embeddings is None or lexical_config is None:
                raise RuntimeError("Word embeddings/config not initialized for lexical metric")
            results[metric] = flowing_weighted_snr(
                context_records,
                next_message,
                embeddings=embeddings,
                config=lexical_config,
                idf_scope=args.idf_scope,
                idf=corpus_idf,
            )
        elif metric == "message_semantic_novelty_min":
            if sentence_embedder is None:
                raise RuntimeError("Sentence embedder not initialized for message semantic novelty")
            results[metric] = message_semantic_novelty(
                context_texts,
                next_message,
                embedder=sentence_embedder,
                k=args.k,
                mode="min",
            )
        elif metric == "message_semantic_novelty_avg":
            if sentence_embedder is None:
                raise RuntimeError("Sentence embedder not initialized for message semantic novelty")
            results[metric] = message_semantic_novelty(
                context_texts,
                next_message,
                embedder=sentence_embedder,
                k=args.k,
                mode="avg",
            )
        elif metric == "dialogue_act_transition_fit":
            if dialogue_act_tagger is None or dialogue_act_model is None:
                raise RuntimeError("DA tagger / transition model not initialized for DAF")
            acts = (
                cached_acts
                if isinstance(cached_acts, list)
                else dialogue_act_tagger.tag_conversation(context_texts, m=args.dialogue_act_m)
            )
            history = acts[-args.daf_order :] if args.daf_order > 0 else []
            next_act = dialogue_act_tagger.tag_next(context_texts, next_message, m=args.dialogue_act_m)
            results[metric] = float(
                dialogue_act_model.probability(str(record.get("dialogue_id")), history, next_act)
            )
        elif metric == "flowing_ll":
            if ll_scorer is None:
                raise RuntimeError("LL scorer not initialized for Flowing_LL")
            did = str(record.get("dialogue_id", ""))
            if ll_context_token_cache is None or did not in ll_context_token_cache:
                prompt_ids = build_ll_prompt_ids(
                    ll_scorer,
                    context_turns=record.get("context", []),
                    next_tag=args.ll_next_tag,
                    prompt_format=args.ll_prompt_format,
                    system_prompt=args.ll_system_prompt,
                )
            else:
                prompt_ids = ll_context_token_cache[did]
            up_ids = ll_scorer.tokenize(next_message)
            raw_ll = ll_scorer.flowing_ll_from_token_ids(prompt_ids=prompt_ids, up_ids=up_ids)
            results[metric] = float(raw_ll)
            c = float(args.ll_scale_c)
            if c > 0:
                results["flowing_ll_scaled"] = float(1.0 - math.exp(-c * float(raw_ll)))
        elif metric == "topic_expansion_score":
            if tes_theta_model is None:
                raise RuntimeError("TES topic model not initialized")
            did = str(record.get("dialogue_id", ""))
            if tes_context_cache is not None and did in tes_context_cache:
                # Use cached k* and dominant topics; only compute Q_p (depends on u_p).
                cached = tes_context_cache[did]
                k_star = int(cached.get("k_star", 0))
                tdom = set(int(x) for x in cached.get("tdom", []))
                if k_star > 0:
                    ctx_turns = record.get("context", [])
                    ell = int(args.tes_ell)
                    ell_turns = ctx_turns[-ell:] if ell > 0 else []
                    up = next_message.strip()
                    up_repeat = int(args.tes_up_repeat)
                    up_repeat_cap = int(args.tes_up_repeat_cap)
                    if up_repeat < 0:
                        raise ValueError("--tes-up-repeat must be >= 0")
                    if up_repeat_cap <= 0:
                        raise ValueError("--tes-up-repeat-cap must be > 0")

                    ell_text = "\n".join(
                        str(t.get("utterance", "")).strip()
                        for t in ell_turns
                        if str(t.get("utterance", "")).strip()
                    )
                    if up_repeat == 0:
                        denom = max(len(up), 1)
                        repeat_eff = max(1, min(int(math.ceil(len(ell_text) / float(denom))), up_repeat_cap))
                    else:
                        repeat_eff = max(1, up_repeat)
                    up_aug = "\n".join([up] * repeat_eff) if up else ""
                    q_doc = "\n".join(
                        [
                            ell_text,
                            up_aug,
                        ]
                    ).strip()
                    q_p = tes_theta_model.theta(q_doc)
                    inactive = [i for i in range(int(q_p.size)) if i not in tdom]
                    results[metric] = float(q_p[inactive].sum()) if inactive else 0.0
                else:
                    results[metric] = 0.0
            else:
                results[metric] = topic_expansion_score(
                    record.get("context", []),
                    next_message,
                    theta_model=tes_theta_model,
                    ell=int(args.tes_ell),
                    delta=int(args.tes_delta),
                    rho=float(args.tes_rho),
                    tau=float(args.tes_jsd_tau),
                    up_repeat=int(args.tes_up_repeat),
                    up_repeat_cap=int(args.tes_up_repeat_cap),
                )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        std = math.sqrt(self.m2 / self.count)
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "min": float(self.min),
            "max": float(self.max),
            "std": float(std),
        }


def run_single(args: argparse.Namespace) -> None:
    corpus_idf = load_corpus_idf(args.idf_path) if args.idf_scope == "corpus" else None
    embeddings: WordEmbeddings | None = None
    lexical_config: LexicalFlowConfig | None = None
    sentence_embedder: SentenceEmbedder | CachedSentenceEmbedder | None = None
    dialogue_act_tagger: DialogueActTagger | None = None
    dialogue_act_model = None
    ll_scorer: CausalLMScorer | None = None
    ll_ctx_cache: Dict[str, List[int]] | None = None
    tes_model: BERTopicTheta | None = None
    tes_cache: Dict[str, Dict[str, Any]] | None = None

    metrics = _selected_metrics(args)

    if "lexical_weighted_snr" in metrics:
        embeddings = WordEmbeddings(args.embeddings_kv, limit=args.embeddings_limit)
        lexical_config = LexicalFlowConfig(
            similarity_threshold=float(args.tau),
            tokenizer=args.tokenizer,
        )

    if any(m.startswith("message_semantic_novelty") for m in metrics):
        base = SentenceEmbedder(args.sentence_embedding_model, batch_size=args.sentence_embedding_batch_size)
        sentence_embedder = (
            CachedSentenceEmbedder(base, max_cache_items=args.sentence_embedding_cache_size)
            if args.sentence_embedding_cache
            else base
        )

    record = sample_single_record(args.pairs_path, dialogue_id=args.dialogue_id, seed=args.seed)

    if "flowing_ll" in metrics:
        ll_scorer = CausalLMScorer(
            args.ll_model,
            device=args.ll_device,
            local_files_only=bool(args.ll_local_files_only),
            max_length=args.ll_max_length,
            dtype=str(args.ll_dtype),
            trust_remote_code=bool(args.ll_trust_remote_code),
        )
        if args.ll_context_cache and args.ll_context_cache.exists() and not args.ll_refresh_context_cache:
            ll_ctx_cache, meta = load_context_token_cache_bundle(args.ll_context_cache)
            expected = {
                "ll_model": args.ll_model,
                "next_tag": args.ll_next_tag,
                "prompt_format": args.ll_prompt_format,
                "system_prompt": args.ll_system_prompt,
            }
            if any(str(meta.get(k)) != str(v) for k, v in expected.items()):
                ll_ctx_cache = None

    if "topic_expansion_score" in metrics:
        tes_model = BERTopicTheta(
            model_path=args.tes_model_path,
            embedding_model=args.tes_embedding_model,
            device=args.tes_device,
            min_topic_size=args.tes_min_topic_size,
            nr_topics=args.tes_nr_topics,
        )
        # Fit/load topic model on contexts (one doc per dialogue).
        docs: list[str] = []
        seen: set[str] = set()
        for rec in iter_jsonl(args.pairs_path):
            did = str(rec.get("dialogue_id", ""))
            if not did or did in seen:
                continue
            seen.add(did)
            ctx = rec.get("context", [])
            if isinstance(ctx, list):
                docs.append("\n".join(str(t.get("utterance", "")).strip() for t in ctx if str(t.get("utterance", "")).strip()))
        tes_model.load_or_fit(docs, refresh=bool(args.tes_refresh_model))

        tes_cache, meta = load_tes_context_cache(args.tes_context_cache)
        expected = {
            "tes_model_path": str(args.tes_model_path),
            "tes_embedding_model": args.tes_embedding_model,
            "tes_device": args.tes_device,
            "tes_min_topic_size": int(args.tes_min_topic_size),
            "tes_nr_topics": args.tes_nr_topics,
            "tes_delta": int(args.tes_delta),
            "tes_jsd_tau": float(args.tes_jsd_tau),
            "tes_rho": float(args.tes_rho),
        }
        if bool(args.tes_refresh_context_cache) or any(str(meta.get(k)) != str(v) for k, v in expected.items()):
            tes_cache = {}

        did = str(record.get("dialogue_id", ""))
        if did and did not in tes_cache:
            ctx_turns = record.get("context", [])
            if isinstance(ctx_turns, list):
                # Precompute k* and dominant topics for this dialogue.
                from src.local_content_unsupervised.topic_expansion import choose_k_star  # noqa: E402
                k_star = choose_k_star(ctx_turns, theta_model=tes_model, delta=int(args.tes_delta), tau=float(args.tes_jsd_tau))
                if k_star > 0:
                    p_doc = "\n".join(str(t.get("utterance", "")).strip() for t in ctx_turns[-k_star:] if str(t.get("utterance", "")).strip())
                    p_star = tes_model.theta(p_doc)
                    from src.local_content_unsupervised.topic_expansion import _tdom_indices  # noqa: E402
                    tdom = sorted(_tdom_indices(p_star, float(args.tes_rho)))
                else:
                    tdom = []
                tes_cache[did] = {"k_star": int(k_star), "tdom": tdom}
                save_tes_context_cache(args.tes_context_cache, cache=tes_cache, meta=expected)

    if "dialogue_act_transition_fit" in metrics:
        dialogue_act_tagger = init_dialogue_act_tagger(args)

        da_cache = load_da_cache(args.daf_acts_cache)
        did = str(record.get("dialogue_id"))
        dialogue_acts = ensure_dialogue_acts(
            args.pairs_path,
            dialogue_act_tagger,
            m=args.dialogue_act_m,
            cache=da_cache,
            refresh=bool(args.daf_refresh_acts_cache),
            limit=args.daf_train_limit,
            ensure_ids=[did],
            progress_desc="Tagging dialogue acts (training)",
        )
        save_da_cache(args.daf_acts_cache, da_cache)

        vocab = list(dialogue_act_tagger.labels())
        dialogue_act_model = train_transition_model(
            dialogue_acts,
            order=args.daf_order,
            alpha=args.daf_alpha,
            act_vocab=vocab if vocab else None,
        )
        record["_cached_context_acts"] = da_cache.get(did)

    scores = score_record(
        record,
        metrics,
        args,
        embeddings=embeddings,
        lexical_config=lexical_config,
        sentence_embedder=sentence_embedder,
        dialogue_act_tagger=dialogue_act_tagger,
        dialogue_act_model=dialogue_act_model,
        ll_scorer=ll_scorer,
        ll_context_token_cache=ll_ctx_cache,
        tes_theta_model=tes_model,
        tes_context_cache=tes_cache,
        corpus_idf=corpus_idf,
    )
    payload: Dict[str, Any] = {"dialogue_id": record.get("dialogue_id"), **scores}
    print(json.dumps(payload, ensure_ascii=False))


def run_batch(args: argparse.Namespace) -> None:
    corpus_idf = load_corpus_idf(args.idf_path) if args.idf_scope == "corpus" else None
    embeddings: WordEmbeddings | None = None
    lexical_config: LexicalFlowConfig | None = None
    sentence_embedder: SentenceEmbedder | CachedSentenceEmbedder | None = None
    dialogue_act_tagger: DialogueActTagger | None = None
    dialogue_act_model = None
    da_cache: Dict[str, List[str]] | None = None
    ll_scorer: CausalLMScorer | None = None
    ll_ctx_cache: Dict[str, List[int]] | None = None
    tes_model: BERTopicTheta | None = None
    tes_cache: Dict[str, Dict[str, Any]] | None = None

    metrics = _selected_metrics(args)

    if "lexical_weighted_snr" in metrics:
        embeddings = WordEmbeddings(args.embeddings_kv, limit=args.embeddings_limit)
        lexical_config = LexicalFlowConfig(
            similarity_threshold=float(args.tau),
            tokenizer=args.tokenizer,
        )

    if any(m.startswith("message_semantic_novelty") for m in metrics):
        base = SentenceEmbedder(args.sentence_embedding_model, batch_size=args.sentence_embedding_batch_size)
        sentence_embedder = (
            CachedSentenceEmbedder(base, max_cache_items=args.sentence_embedding_cache_size)
            if args.sentence_embedding_cache
            else base
        )

    if "dialogue_act_transition_fit" in metrics:
        dialogue_act_tagger = init_dialogue_act_tagger(args)
        da_cache = load_da_cache(args.daf_acts_cache)
        dialogue_acts = ensure_dialogue_acts(
            args.pairs_path,
            dialogue_act_tagger,
            m=args.dialogue_act_m,
            cache=da_cache,
            refresh=bool(args.daf_refresh_acts_cache),
            limit=args.daf_train_limit,
            progress_desc="Tagging dialogue acts (training)",
        )
        save_da_cache(args.daf_acts_cache, da_cache)

        vocab = list(dialogue_act_tagger.labels())
        dialogue_act_model = train_transition_model(
            dialogue_acts,
            order=args.daf_order,
            alpha=args.daf_alpha,
            act_vocab=vocab if vocab else None,
        )

    if "flowing_ll" in metrics:
        ll_scorer = CausalLMScorer(
            args.ll_model,
            device=args.ll_device,
            local_files_only=bool(args.ll_local_files_only),
            max_length=args.ll_max_length,
            dtype=str(args.ll_dtype),
            trust_remote_code=bool(args.ll_trust_remote_code),
        )
        ll_ctx_cache = {}
        meta_loaded: dict[str, Any] = {}
        if args.ll_context_cache.exists() and not args.ll_refresh_context_cache:
            ll_ctx_cache, meta_loaded = load_context_token_cache_bundle(args.ll_context_cache)
        expected = {
            "ll_model": args.ll_model,
            "next_tag": args.ll_next_tag,
            "prompt_format": args.ll_prompt_format,
            "system_prompt": args.ll_system_prompt,
        }
        if any(str(meta_loaded.get(k)) != str(v) for k, v in expected.items()):
            ll_ctx_cache = {}

        # Populate cache for all dialogue_ids in this pairs file (contexts are fixed across methods).
        seen: set[str] = set(ll_ctx_cache.keys())
        to_add: list[tuple[str, list[dict[str, Any]]]] = []
        for rec in iter_jsonl(args.pairs_path):
            did = str(rec.get("dialogue_id", ""))
            if not did or did in seen:
                continue
            seen.add(did)
            ctx_raw = rec.get("context", [])
            if isinstance(ctx_raw, list):
                to_add.append((did, ctx_raw))

        for did, ctx_raw in tqdm(to_add, desc="Tokenizing contexts (LL cache)"):
            ll_ctx_cache[did] = build_ll_prompt_ids(
                ll_scorer,
                context_turns=ctx_raw,
                next_tag=args.ll_next_tag,
                prompt_format=args.ll_prompt_format,
                system_prompt=args.ll_system_prompt,
            )

        save_context_token_cache(
            args.ll_context_cache,
            contexts=ll_ctx_cache,
            meta={
                "ll_model": args.ll_model,
                "tokenizer": ll_scorer.tokenizer_name_or_path if ll_scorer else None,
                "next_tag": args.ll_next_tag,
                "prompt_format": args.ll_prompt_format,
                "system_prompt": args.ll_system_prompt,
                "pairs_path": str(args.pairs_path),
            },
        )

    if "topic_expansion_score" in metrics:
        tes_model = BERTopicTheta(
            model_path=args.tes_model_path,
            embedding_model=args.tes_embedding_model,
            device=args.tes_device,
            min_topic_size=args.tes_min_topic_size,
            nr_topics=args.tes_nr_topics,
        )
        # Fit/load topic model on contexts (one doc per dialogue).
        docs: list[str] = []
        seen: set[str] = set()
        for rec in iter_jsonl(args.pairs_path):
            did = str(rec.get("dialogue_id", ""))
            if not did or did in seen:
                continue
            seen.add(did)
            ctx = rec.get("context", [])
            if isinstance(ctx, list):
                docs.append("\n".join(str(t.get("utterance", "")).strip() for t in ctx if str(t.get("utterance", "")).strip()))
        tes_model.load_or_fit(docs, refresh=bool(args.tes_refresh_model))

        tes_cache, meta = load_tes_context_cache(args.tes_context_cache)
        expected = {
            "tes_model_path": str(args.tes_model_path),
            "tes_embedding_model": args.tes_embedding_model,
            "tes_device": args.tes_device,
            "tes_min_topic_size": int(args.tes_min_topic_size),
            "tes_nr_topics": args.tes_nr_topics,
            "tes_delta": int(args.tes_delta),
            "tes_jsd_tau": float(args.tes_jsd_tau),
            "tes_rho": float(args.tes_rho),
        }
        if bool(args.tes_refresh_context_cache) or any(str(meta.get(k)) != str(v) for k, v in expected.items()):
            tes_cache = {}

        # Populate per-dialogue cache: k* and T_dom(P*,rho).
        from src.local_content_unsupervised.topic_expansion import choose_k_star, _tdom_indices  # noqa: E402
        to_add: list[tuple[str, list[dict[str, Any]]]] = []
        for rec in iter_jsonl(args.pairs_path):
            did = str(rec.get("dialogue_id", ""))
            if not did or did in tes_cache:
                continue
            ctx_turns = rec.get("context", [])
            if isinstance(ctx_turns, list):
                to_add.append((did, ctx_turns))

        for did, ctx_turns in tqdm(to_add, desc="Precomputing TES context cache"):
            k_star = choose_k_star(ctx_turns, theta_model=tes_model, delta=int(args.tes_delta), tau=float(args.tes_jsd_tau))
            if k_star > 0:
                p_doc = "\n".join(str(t.get("utterance", "")).strip() for t in ctx_turns[-k_star:] if str(t.get("utterance", "")).strip())
                p_star = tes_model.theta(p_doc)
                tdom = sorted(_tdom_indices(p_star, float(args.tes_rho)))
            else:
                tdom = []
            tes_cache[did] = {"k_star": int(k_star), "tdom": tdom}

        save_tes_context_cache(args.tes_context_cache, cache=tes_cache, meta=expected)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = args.scores_path
    if scores_path is None:
        if args.output_name:
            scores_path = args.output_dir / f"{args.output_name}_scores.jsonl"
        else:
            scores_path = args.output_dir / "local_content_scores.jsonl"
    elif not scores_path.is_absolute():
        scores_path = args.output_dir / scores_path
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, RunningStats] = {metric: RunningStats() for metric in metrics}
    if "flowing_ll" in metrics:
        stats["flowing_ll_scaled"] = RunningStats()
    with scores_path.open("w", encoding="utf-8") as fp:
        for record in tqdm(iter_jsonl(args.pairs_path), desc="Scoring local content metrics"):
            if "dialogue_act_transition_fit" in metrics and da_cache is not None:
                did = str(record.get("dialogue_id"))
                record["_cached_context_acts"] = da_cache.get(did)
            scores = score_record(
                record,
                metrics,
                args,
                embeddings=embeddings,
                lexical_config=lexical_config,
                sentence_embedder=sentence_embedder,
                dialogue_act_tagger=dialogue_act_tagger,
                dialogue_act_model=dialogue_act_model,
                ll_scorer=ll_scorer,
                ll_context_token_cache=ll_ctx_cache,
                tes_theta_model=tes_model,
                tes_context_cache=tes_cache,
                corpus_idf=corpus_idf,
            )
            for metric, value in scores.items():
                if metric in stats:
                    stats[metric].update(float(value))
            row: Dict[str, Any] = {"dialogue_id": record.get("dialogue_id"), **scores}
            json.dump(row, fp, ensure_ascii=False)
            fp.write("\n")

    summary = {
        "metrics": {metric: stats[metric].as_dict() for metric in metrics},
        "pairs_path": str(args.pairs_path),
        "scores_path": str(scores_path),
        "summary_path": None,
        "idf_scope": args.idf_scope,
        "idf_path": str(args.idf_path) if args.idf_scope == "corpus" else None,
        "k": args.k,
        "daf_order": args.daf_order,
        "daf_alpha": args.daf_alpha,
        "dialogue_act_m": args.dialogue_act_m,
        "dialogue_act_tagger": args.dialogue_act_tagger,
        "dialogue_act_model": args.dialogue_act_model,
        "ll_model": args.ll_model if "flowing_ll" in metrics else None,
    }
    if "flowing_ll" in metrics:
        summary["metrics"]["flowing_ll_scaled"] = stats["flowing_ll_scaled"].as_dict()
        summary["ll_prompt_format"] = args.ll_prompt_format
        summary["ll_system_prompt"] = args.ll_system_prompt
        summary["ll_next_tag"] = args.ll_next_tag
        summary["ll_context_cache"] = str(args.ll_context_cache)
        summary["ll_scale_c"] = float(args.ll_scale_c)

    if "topic_expansion_score" in metrics:
        summary.update(
            {
                "tes_model_path": str(args.tes_model_path),
                "tes_embedding_model": args.tes_embedding_model,
                "tes_device": args.tes_device,
                "tes_min_topic_size": int(args.tes_min_topic_size),
                "tes_nr_topics": args.tes_nr_topics,
                "tes_ell": int(args.tes_ell),
                "tes_up_repeat": int(args.tes_up_repeat),
                "tes_up_repeat_cap": int(args.tes_up_repeat_cap),
                "tes_delta": int(args.tes_delta),
                "tes_rho": float(args.tes_rho),
                "tes_jsd_tau": float(args.tes_jsd_tau),
                "tes_context_cache": str(args.tes_context_cache),
            }
        )
    if "lexical_weighted_snr" in metrics:
        summary.update(
            {
                "tokenizer": args.tokenizer,
                "embeddings_kv": str(args.embeddings_kv),
                "embeddings_limit": args.embeddings_limit,
                "tau": args.tau,
            }
        )
    if any(m.startswith("message_semantic_novelty") for m in metrics):
        summary.update(
            {
                "sentence_embedding_model": args.sentence_embedding_model,
                "sentence_embedding_batch_size": args.sentence_embedding_batch_size,
                "sentence_embedding_cache": bool(args.sentence_embedding_cache),
                "sentence_embedding_cache_size": args.sentence_embedding_cache_size,
            }
        )
        if isinstance(sentence_embedder, CachedSentenceEmbedder):
            summary["sentence_embedding_cache_info"] = sentence_embedder.cache_info

    summary_path = args.summary_path
    if summary_path is None:
        if args.output_name:
            summary_path = args.output_dir / f"{args.output_name}_summary.json"
        else:
            summary_path = args.output_dir / "local_content_summary.json"
    elif not summary_path.is_absolute():
        summary_path = args.output_dir / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["summary_path"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
