"""Log-likelihood-based flow metric for Sec. 4.2.2 (Eq. 20).

Flowing_LL = exp( (1/T) * Σ_{t=1..T} log p(u_{p,t} | u_{p,<t}, C) )

We estimate this using a causal LM with teacher-forcing: construct input_ids as
prompt (context) + continuation (predicted message), and set labels to -100 for
the prompt tokens so loss is computed only over the continuation tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def format_context_with_speakers(
    turns: Sequence[Mapping[str, Any]],
    *,
    next_tag: str = "Next:",
) -> str:
    """Render context turns into a single prompt string with speaker tags."""

    lines: list[str] = []
    for t in turns:
        speaker = str(t.get("speaker", "")).strip()
        utt = str(t.get("utterance", "")).strip()
        if not speaker or not utt:
            continue
        lines.append(f"{speaker}: {utt}")
    # Optional fixed next-turn tag as part of the prompt.
    #
    # Important: keep a trailing newline so the continuation starts cleanly
    # after the tag. Do not strip(), since that would remove the separator.
    if str(next_tag).strip():
        lines.append(str(next_tag))
    return "\n".join(lines) + "\n"


def build_ll_prompt_ids(
    scorer: "CausalLMScorer",
    *,
    context_turns: Sequence[Mapping[str, Any]],
    next_tag: str,
    prompt_format: str,
    system_prompt: str | None,
) -> list[int]:
    """Build prompt token ids for Flowing_LL.

    prompt_format:
      - "raw": a single raw string with speaker tags + trailing next tag
      - "chat": use tokenizer chat template; score u_p as assistant continuation
    """

    if prompt_format not in {"raw", "chat"}:
        raise ValueError("prompt_format must be 'raw' or 'chat'")

    transcript = format_context_with_speakers(context_turns, next_tag=next_tag)
    if prompt_format == "raw":
        return scorer.tokenize(transcript)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": transcript})
    return scorer.apply_chat_template(messages)


@dataclass
class CausalLMScorer:
    """Causal LM log-likelihood scorer for predicted next messages."""

    model_name_or_path: str
    device: int = -1
    local_files_only: bool = False
    max_length: int | None = None
    dtype: str = "auto"  # auto | bf16 | fp16
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "transformers + torch are required for Flowing_LL. "
                "Install via your environment (e.g., pip/uv)."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=bool(self.local_files_only),
            use_fast=True,
            trust_remote_code=bool(self.trust_remote_code),
        )

        torch_dtype: "torch.dtype | str | None" = None
        if self.dtype == "auto":
            # On GPU, let HF pick the checkpoint dtype (commonly bf16/fp16) to
            # avoid accidentally upcasting large models to fp32.
            if self.device >= 0 and torch.cuda.is_available():
                torch_dtype = "auto"
        elif self.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif self.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            raise ValueError("dtype must be one of: auto, bf16, fp16")

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            local_files_only=bool(self.local_files_only),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=bool(self.trust_remote_code),
        )
        if self.device >= 0 and torch.cuda.is_available():
            self._model.to(f"cuda:{self.device}")
        self._model.eval()
        try:
            # Caching is useful for generation but wastes memory for teacher-forcing scoring.
            self._model.config.use_cache = False
        except Exception:
            pass

        # Resolve a sensible max length.
        cfg_max = getattr(self._model.config, "max_position_embeddings", None)
        cfg_max_i = int(cfg_max) if isinstance(cfg_max, int) or (isinstance(cfg_max, str) and cfg_max.isdigit()) else 4096
        default_max_len = min(cfg_max_i, 4096)
        self._max_len = int(self.max_length) if self.max_length is not None else default_max_len

        # Ensure we have a PAD token for batching/padding if needed.
        if self._tokenizer.pad_token_id is None:
            # Many causal LMs set pad_token_id to eos_token_id.
            if self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def tokenizer_name_or_path(self) -> str:
        return str(getattr(self._tokenizer, "name_or_path", self.model_name_or_path))

    def tokenize(self, text: str) -> list[int]:
        enc = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc.get("input_ids", [])
        return [int(x) for x in ids]

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        """Tokenize a prompt using the tokenizer's chat template."""

        if not hasattr(self._tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not support chat templates (no apply_chat_template).")
        ids = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        return [int(x) for x in ids]

    def flowing_ll_from_token_ids(self, *, prompt_ids: Sequence[int], up_ids: Sequence[int]) -> float:
        """Compute Flowing_LL given tokenized prompt and predicted message ids."""

        torch = self._torch
        if not up_ids:
            return 0.0

        bos = self._tokenizer.bos_token_id
        eos = self._tokenizer.eos_token_id
        prefix: list[int] = [int(bos)] if bos is not None else []

        # Keep all u_p if possible; otherwise truncate u_p from the left as a last resort.
        max_total = self._max_len
        max_prompt = max_total - len(prefix) - len(up_ids)
        if max_prompt < 0:
            # u_p alone doesn't fit: keep the last tokens of u_p so labels align at end.
            up_ids = list(up_ids)[-max(1, max_total - len(prefix)) :]
            max_prompt = 0

        prompt_trunc = list(prompt_ids)[-max_prompt:] if max_prompt > 0 else []
        input_ids = [*prefix, *prompt_trunc, *up_ids]
        if eos is not None:
            # Optional EOS can stabilize termination; do not score EOS.
            input_ids = [*input_ids, int(eos)]

        # Labels: score only u_p tokens (exclude prompt and EOS).
        ignore = -100
        labels = [ignore] * (len(prefix) + len(prompt_trunc)) + list(up_ids)
        if eos is not None:
            labels = [*labels, ignore]

        inp = torch.tensor([input_ids], dtype=torch.long)
        lab = torch.tensor([labels], dtype=torch.long)
        if self.device >= 0 and torch.cuda.is_available():
            inp = inp.to(f"cuda:{self.device}")
            lab = lab.to(f"cuda:{self.device}")

        with torch.inference_mode():
            out = self._model(input_ids=inp, labels=lab, use_cache=False)
            loss = out.loss
        if loss is None:
            return 0.0
        # Eq. (20): exp(mean log p) = exp(-cross_entropy_loss).
        return float(torch.exp(-loss).detach().cpu().item())


def load_context_token_cache(cache_path: Path) -> Dict[str, List[int]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if isinstance(raw, dict) and "contexts" in raw and isinstance(raw["contexts"], dict):
        contexts = raw["contexts"]
    else:
        contexts = raw
    out: Dict[str, List[int]] = {}
    if isinstance(contexts, dict):
        for did, ids in contexts.items():
            if isinstance(ids, list):
                out[str(did)] = [int(x) for x in ids]
    return out


def load_context_token_cache_bundle(cache_path: Path) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Load contexts + metadata.

    Backward compatible:
      - old format: {"dialogue_id": [ids...], ...} with empty meta
      - new format: {"meta": {...}, "contexts": {...}}
    """

    if not cache_path.exists():
        return {}, {}
    with cache_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    if isinstance(raw, dict) and "contexts" in raw and isinstance(raw["contexts"], dict):
        meta = raw.get("meta", {}) if isinstance(raw.get("meta", {}), dict) else {}
        contexts_raw = raw["contexts"]
    else:
        meta = {}
        contexts_raw = raw

    out: Dict[str, List[int]] = {}
    if isinstance(contexts_raw, dict):
        for did, ids in contexts_raw.items():
            if isinstance(ids, list):
                out[str(did)] = [int(x) for x in ids]
    return out, dict(meta)


def save_context_token_cache(
    cache_path: Path,
    *,
    contexts: Dict[str, List[int]],
    meta: Dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": dict(meta), "contexts": contexts}
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)
