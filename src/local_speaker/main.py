"""CLI to run all Section 4.1 local speaker metrics on MPDD samples."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.local_speaker.utils import (
    TopicModeler,
    average_embedding_similarity_score,
    direct_name_reference_score,
    implicit_reference_score,
    maximum_embedding_similarity_score,
    participation_frequency_score,
    topic_alignment_score,
)
from src.local_speaker.utils.embedding_similarity import (
    SentenceTransformerEmbedder,
    TfidfEmbedder,
)
from src.local_speaker.utils.types import TurnRecord
from src.processing.mpdd.sliding_window_pairs import (
    build_examples,
    load_dialogues,
    write_jsonl,
)

DEFAULT_OUTPUT_DIR = Path("src/local_speaker/output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dialogue-path",
        type=Path,
        default=Path("mpdd/dialogue.json"),
        help="Path to MPDD dialogue JSON file.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Number of previous turns (k) to include in the context window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window extraction.",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path("data/mpdd_local_pairs.jsonl"),
        help="Where to cache generated (context, next speaker) pairs.",
    )
    parser.add_argument(
        "--refresh-pairs",
        action="store_true",
        help="Regenerate the cached pairs file even if it exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when sampling a pair.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embedding-based metrics.",
    )
    parser.add_argument(
        "--topic-backend",
        type=str,
        default="bertopic",
        choices=["auto", "lda", "bertopic"],
        help="Backend to use for topic alignment scoring.",
    )
    parser.add_argument(
        "--topic-embedding-model",
        type=str,
        default=None,
        help="Embedding model name passed to BERTopic when applicable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages while computing metrics.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Evaluation mode: single sample or batch over all pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store batch evaluation outputs.",
    )
    return parser.parse_args()


def ensure_pairs_file(
    dialogue_path: Path,
    window_size: int,
    stride: int,
    output_path: Path,
    refresh: bool,
) -> Path:
    if output_path.exists() and not refresh:
        return output_path

    dialogues = load_dialogues(dialogue_path)
    examples = build_examples(dialogues, window_size, stride)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(examples, output_path)
    return output_path


def sample_random_pair(path: Path, seed: Optional[int]) -> Dict[str, Any]:
    rng = random.Random(seed)
    chosen: Optional[Dict[str, Any]] = None
    max_context = -1
    matches = 0
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            context_len = len(record.get("context", []))
            if context_len > max_context:
                max_context = context_len
                chosen = record
                matches = 1
            elif context_len == max_context and context_len >= 0:
                matches += 1
                if rng.randrange(matches) == 0:
                    chosen = record
    if chosen is None:
        raise RuntimeError(f"No pairs found in {path}")
    return chosen


def to_turn_record(turn: Dict[str, Any]) -> TurnRecord:
    return TurnRecord(
        speaker=str(turn.get("speaker", "")),
        utterance=str(turn.get("utterance", "")),
        listener=turn.get("listener"),
        emotion=turn.get("emotion"),
    )


def build_embedder(model_name: str):
    try:
        return SentenceTransformerEmbedder(model_name=model_name)
    except Exception:
        return TfidfEmbedder()


def build_topic_modeler(backend: str, embedding_model: str | None) -> TopicModeler:
    return TopicModeler(backend=backend, embedding_model=embedding_model)


def evaluate_metrics(
    context_records: Iterable[TurnRecord],
    predicted_speaker: str,
    embedder,
    topic_model: TopicModeler,
    logger: Callable[[str], None] | None = None,
) -> Dict[str, float]:
    context_list = list(context_records)

    def log(message: str) -> None:
        if logger:
            logger(message)

    average_sim = average_embedding_similarity_score(
        predicted_speaker, context_list, embedder=embedder
    )
    log(f"Average embedding similarity: {average_sim:.4f}")
    max_sim = maximum_embedding_similarity_score(
        predicted_speaker, context_list, embedder=embedder
    )
    log(f"Maximum embedding similarity: {max_sim:.4f}")

    topic_score = topic_alignment_score(
        predicted_speaker,
        context_list,
        model=topic_model,
    )
    log(f"Topic alignment: {topic_score:.4f}")

    dnr = direct_name_reference_score(predicted_speaker, context_list)
    log(f"Direct name reference: {dnr:.4f}")
    ir = implicit_reference_score(predicted_speaker, context_list)
    log(f"Implicit reference: {ir:.4f}")
    pf = participation_frequency_score(predicted_speaker, context_list)
    log(f"Participation frequency: {pf:.4f}")

    unique_speakers: Dict[str, float] = {}
    for turn in context_list:
        speaker = turn.speaker
        if speaker not in unique_speakers:
            unique_speakers[speaker] = implicit_reference_score(speaker, context_list)
            log(f"Implicit reference ({speaker}): {unique_speakers[speaker]:.4f}")

    return {
        "direct_name_reference": dnr,
        "implicit_reference": ir,
        "implicit_reference_all": unique_speakers,
        "participation_frequency": pf,
        "average_embedding_similarity": average_sim,
        "maximum_embedding_similarity": max_sim,
        "topic_alignment": topic_score,
    }


def main() -> None:
    args = parse_args()

    def log(message: str) -> None:
        if args.verbose:
            print(f"[local-speaker] {message}")

    if args.mode == "single":
        run_single_mode(args, log)
    else:
        run_batch_mode(args, log)


def run_single_mode(args: argparse.Namespace, log: Callable[[str], None]) -> None:
    pairs_path = ensure_pairs_file(
        dialogue_path=args.dialogue_path,
        window_size=args.window_size,
        stride=args.stride,
        output_path=args.pairs_output,
        refresh=args.refresh_pairs,
    )
    log(f"Using pairs from {pairs_path}")

    sampled = sample_random_pair(pairs_path, args.seed)
    context_raw = sampled.get("context", [])
    context_records = [to_turn_record(turn) for turn in context_raw]
    next_speaker = str(sampled.get("next_speaker", ""))

    log(
        "Sampled dialogue {did} with {ctx} context turns; next speaker: {speaker}".format(
            did=sampled.get("dialogue_id"),
            ctx=len(context_records),
            speaker=next_speaker or "<unknown>",
        )
    )

    embedder = build_embedder(args.embedding_model)
    log(f"Embedding backend: {getattr(embedder, 'model_name', embedder.__class__.__name__)}")
    topic_model = build_topic_modeler(args.topic_backend, args.topic_embedding_model)
    log(f"Topic backend: {args.topic_backend}")

    metrics = evaluate_metrics(
        context_records=context_records,
        predicted_speaker=next_speaker,
        embedder=embedder,
        topic_model=topic_model,
        logger=log if args.verbose else None,
    )

    report = round_data({
        "dialogue_id": sampled.get("dialogue_id"),
        "window_size": args.window_size,
        "context": context_raw,
        "next_speaker": next_speaker,
        "metrics": metrics,
        "embedding_model": getattr(embedder, "model_name", embedder.__class__.__name__),
        "topic_backend": args.topic_backend,
    })

    if "target_index" in sampled:
        report["target_index"] = sampled["target_index"]

    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_batch_mode(args: argparse.Namespace, log: Callable[[str], None]) -> None:
    pairs_path = ensure_pairs_file(
        dialogue_path=args.dialogue_path,
        window_size=args.window_size,
        stride=args.stride,
        output_path=args.pairs_output,
        refresh=args.refresh_pairs,
    )
    log(f"Batch evaluation over pairs in {pairs_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = output_dir / "pair_scores.jsonl"
    summary_path = output_dir / "summary.json"

    embedder = build_embedder(args.embedding_model)
    metric_values: Dict[str, List[float]] = defaultdict(list)
    pair_count = 0

    with pairs_path.open("r", encoding="utf-8") as in_fp, per_pair_path.open(
        "w", encoding="utf-8"
    ) as out_fp:
        progress = tqdm(in_fp, disable=not args.verbose, desc="Pairs", unit="pair")
        for line in progress:
            record = json.loads(line)
            context_raw = record.get("context", [])
            if not context_raw:
                continue
            next_speaker = str(record.get("next_speaker", ""))
            context_records = [to_turn_record(turn) for turn in context_raw]

            topic_model = build_topic_modeler(args.topic_backend, args.topic_embedding_model)
            metrics = evaluate_metrics(
                context_records=context_records,
                predicted_speaker=next_speaker,
                embedder=embedder,
                topic_model=topic_model,
                logger=None,
            )

            output_record = round_data({
                "dialogue_id": record.get("dialogue_id"),
                "context_size": len(context_raw),
                "next_speaker": next_speaker,
                "metrics": metrics,
            })
            json.dump(output_record, out_fp, ensure_ascii=False)
            out_fp.write("\n")

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    metric_values[metric_name].append(float(metric_value))

            pair_count += 1

    summary = round_data({
        "total_pairs": pair_count,
        "metrics": {
            name: summarize_metric(values) for name, values in metric_values.items()
        },
        "embedding_model": getattr(embedder, "model_name", embedder.__class__.__name__),
        "topic_backend": args.topic_backend,
        "pairs_report": str(per_pair_path),
    })

    with summary_path.open("w", encoding="utf-8") as summary_fp:
        json.dump(summary, summary_fp, ensure_ascii=False, indent=2)

    log(
        f"Batch reports written to {per_pair_path} (per-pair) and {summary_path} (summary)."
    )


def summarize_metric(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    avg = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": avg,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def round_data(obj: Any, decimals: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {key: round_data(value, decimals) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_data(item, decimals) for item in obj]
    return obj


if __name__ == "__main__":
    main()
