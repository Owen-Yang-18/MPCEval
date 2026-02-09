"""Dialogue-act tagging and transition fit (DAF) for Sec. 4.2.2.

Implements Eq. (19):
    FlowingDAF = p(â | a_{m-k+1:m})

Where â is the dialogue act of the predicted message u_p and the history is the
sequence of dialogue acts for the last k messages in the context window. We
estimate p(·) using an n-gram model over dialogue acts with leave-one-dialogue-
out (LOO) counts and backoff smoothing.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Act = str
History = Tuple[Act, ...]


def _prepare_da_input(
    utterances: Sequence[str],
    index: int,
    *,
    m: int = 2,
) -> str:
    """Build DA tagger input for u[index] using up to m previous utterances."""

    current = (utterances[index] or "").strip()
    if not current:
        return ""
    start = max(0, index - max(m, 0))
    prev = [u.strip() for u in utterances[start:index] if u and u.strip()]
    if prev:
        return "\n".join([*prev, current])
    return current


class DialogueActTagger:
    def labels(self) -> Sequence[str]:
        raise NotImplementedError

    def tag_inputs(self, inputs: Sequence[str]) -> List[Act]:
        raise NotImplementedError

    def tag_conversation(self, utterances: Sequence[str], *, m: int = 2) -> List[Act]:
        inputs = [_prepare_da_input(utterances, i, m=m) for i in range(len(utterances))]
        acts = self.tag_inputs(inputs)
        return acts

    def tag_next(self, context_utterances: Sequence[str], next_message: str, *, m: int = 2) -> Act:
        combined = [*context_utterances, next_message]
        inp = _prepare_da_input(combined, len(combined) - 1, m=m)
        return self.tag_inputs([inp])[0]


class RuleBasedDialogueActTagger(DialogueActTagger):
    """Very small heuristic tagger used as a fallback (low quality)."""

    _labels = ("QUESTION", "GREETING", "THANKS", "OTHER")

    def labels(self) -> Sequence[str]:
        return self._labels

    def tag_inputs(self, inputs: Sequence[str]) -> List[Act]:
        acts: list[str] = []
        for text in inputs:
            t = (text or "").strip()
            if not t:
                acts.append("OTHER")
                continue
            if any(ch in t for ch in ("?", "？")):
                acts.append("QUESTION")
            elif any(x in t for x in ("你好", "您好", "嗨", "哈囉", "哈喽")):
                acts.append("GREETING")
            elif any(x in t for x in ("謝謝", "谢谢", "感謝", "感谢")):
                acts.append("THANKS")
            else:
                acts.append("OTHER")
        return acts


class TransformersDialogueActTagger(DialogueActTagger):
    """HF Transformers sequence-classification-based dialogue act tagger."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        local_files_only: bool = False,
        batch_size: int = 32,
        max_length: int = 256,
        device: int = -1,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        import torch

        self.model_name_or_path = model_name_or_path
        self.local_files_only = bool(local_files_only)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = int(device)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=self.local_files_only,
            use_fast=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            local_files_only=self.local_files_only,
        )

        if self.device >= 0 and torch.cuda.is_available():
            self._model.to(f"cuda:{self.device}")
        self._model.eval()

        id2label = getattr(self._model.config, "id2label", None) or {}
        self._id2label = {int(k): str(v) for k, v in id2label.items()} if id2label else None

    def labels(self) -> Sequence[str]:
        if self._id2label:
            return [self._id2label[i] for i in sorted(self._id2label)]
        # Fallback: unknown label set (still fine for scoring).
        num = int(getattr(self._model.config, "num_labels", 0) or 0)
        return [f"LABEL_{i}" for i in range(num)]

    def tag_inputs(self, inputs: Sequence[str]) -> List[Act]:
        import torch

        texts = [(t or "").strip() for t in inputs]
        if not texts:
            return []

        acts: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device >= 0 and torch.cuda.is_available():
                enc = {k: v.to(f"cuda:{self.device}") for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
                logits = out.logits
                preds = torch.argmax(logits, dim=-1).tolist()
            for idx in preds:
                if self._id2label:
                    acts.append(self._id2label.get(int(idx), str(idx)))
                else:
                    acts.append(f"LABEL_{int(idx)}")
        return acts


def _normalize_seq2seq_act(text: str) -> str:
    """Normalize a seq2seq NLU output into a discrete act label.

    Many NLU models output one or more dialogue acts as a serialized string,
    e.g., "Inform(domain=..., slot=...); Request(slot=...)".

    For DAF we need a single label A(u). We extract act type tokens and join
    them into a compact label, e.g. "INFORM+REQUEST". If nothing is found,
    return "OTHER".
    """

    t = (text or "").strip()
    if not t:
        return "OTHER"

    # Extract leading act names like inform/request/confirm/deny/etc.
    # Common formats include: "inform(...)" / "Inform(...)" / "inform" / "inform ; request"
    # Match patterns like "inform(...)" or "Inform: ..."
    candidates = re.findall(r"([A-Za-z][A-Za-z0-9_-]{1,30})\s*(?:\(|:)", t)
    if not candidates:
        # fallback: split on separators and take the first alpha token
        candidates = re.findall(r"([A-Za-z][A-Za-z0-9_-]{1,30})", t)

    if not candidates:
        return "OTHER"

    keep: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        up = c.strip().upper()
        if not up or up in seen:
            continue
        seen.add(up)
        keep.append(up)

    return "+".join(keep[:4]) if keep else "OTHER"


class Seq2SeqDialogueActTagger(DialogueActTagger):
    """Seq2Seq NLU model used as a dialogue-act tagger (e.g., CrossWOZ NLU)."""

    def __init__(
        self,
        model_name_or_path: str = "ConvLab/mt5-small-nlu-all-crosswoz",
        *,
        local_files_only: bool = False,
        batch_size: int = 16,
        max_length: int = 256,
        max_new_tokens: int = 64,
        num_beams: int = 1,
        device: int = -1,
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        import torch

        self.model_name_or_path = model_name_or_path
        self.local_files_only = bool(local_files_only)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.device = int(device)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be > 0")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=self.local_files_only,
            use_fast=True,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            local_files_only=self.local_files_only,
        )
        if self.device >= 0 and torch.cuda.is_available():
            self._model.to(f"cuda:{self.device}")
        self._model.eval()

    def labels(self) -> Sequence[str]:
        # Unknown a priori; learned from data.
        return ()

    def tag_inputs(self, inputs: Sequence[str]) -> List[Act]:
        import torch

        texts = [(t or "").strip() for t in inputs]
        if not texts:
            return []

        outputs: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device >= 0 and torch.cuda.is_available():
                enc = {k: v.to(f"cuda:{self.device}") for k, v in enc.items()}
            with torch.no_grad():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                )
            decoded = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
            outputs.extend(decoded)

        return [_normalize_seq2seq_act(o) for o in outputs]


@dataclass
class DialogueActNgramModel:
    """Leave-one-dialogue-out n-gram model with backoff and add-alpha smoothing."""

    order: int = 4
    alpha: float = 0.1
    act_vocab: Sequence[Act] = ()

    # global_counts[h][a] = count
    global_counts: Mapping[History, Counter] = None  # type: ignore[assignment]
    # per_dialogue_counts[dialogue_id][h][a] = count
    per_dialogue_counts: Mapping[str, Mapping[History, Counter]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("order must be >= 0")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")

    def probability(self, dialogue_id: str, history: Sequence[Act], next_act: Act) -> float:
        """Compute LOO probability with backoff from full order down to unigram."""

        vocab = list(self.act_vocab) if self.act_vocab else []
        if next_act not in vocab:
            vocab.append(next_act)
        v = max(len(vocab), 1)

        per_counts = self.per_dialogue_counts.get(dialogue_id, {}) if self.per_dialogue_counts else {}

        hist_list = list(history)
        max_order = min(self.order, len(hist_list))
        for l in range(max_order, -1, -1):
            h = tuple(hist_list[-l:]) if l > 0 else tuple()
            global_counter = self.global_counts.get(h, Counter()) if self.global_counts else Counter()
            dialogue_counter = per_counts.get(h, Counter())

            total = sum(global_counter.values()) - sum(dialogue_counter.values())
            count = int(global_counter.get(next_act, 0)) - int(dialogue_counter.get(next_act, 0))
            if total <= 0 and l > 0:
                continue

            if total <= 0:
                # No evidence even at unigram level; uniform.
                return 1.0 / v

            alpha = float(self.alpha)
            return float((count + alpha) / (total + alpha * v))

        return 1.0 / v


def build_ngram_counts(
    dialogue_acts: Mapping[str, Sequence[Act]],
    *,
    order: int = 4,
) -> tuple[Dict[History, Counter], Dict[str, Dict[History, Counter]]]:
    """Build global + per-dialogue n-gram counts from act sequences.

    Uses training instances from each dialogue context by sliding windows of
    length order+1 over the act sequence. For each position, we also update
    backoff histories down to length 0.
    """

    global_counts: Dict[History, Counter] = defaultdict(Counter)
    per_dialogue: Dict[str, Dict[History, Counter]] = {}

    for did, acts in dialogue_acts.items():
        per_counts: Dict[History, Counter] = defaultdict(Counter)
        acts_list = [a for a in acts if a]
        if len(acts_list) <= order:
            per_dialogue[did] = per_counts
            continue
        for idx in range(order, len(acts_list)):
            history_full = tuple(acts_list[idx - order : idx])
            next_act = acts_list[idx]
            # Backoff histories: length 0..order
            for l in range(0, order + 1):
                h = history_full[-l:] if l > 0 else tuple()
                global_counts[h][next_act] += 1
                per_counts[h][next_act] += 1
        per_dialogue[did] = per_counts

    return global_counts, per_dialogue


def train_transition_model(
    dialogue_acts: Mapping[str, Sequence[Act]],
    *,
    order: int = 4,
    alpha: float = 0.1,
    act_vocab: Sequence[Act] | None = None,
) -> DialogueActNgramModel:
    global_counts, per_dialogue = build_ngram_counts(dialogue_acts, order=order)
    if act_vocab is None:
        observed: set[str] = set()
        for counter in global_counts.values():
            observed.update(counter.keys())
        act_vocab = sorted(observed)
    return DialogueActNgramModel(
        order=order,
        alpha=alpha,
        act_vocab=tuple(act_vocab),
        global_counts=global_counts,
        per_dialogue_counts=per_dialogue,
    )
