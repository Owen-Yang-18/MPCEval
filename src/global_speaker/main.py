"""CLI for global speaker diversity and dynamics metrics (Section 5.1.2)."""

from __future__ import annotations

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence

import numpy as np
import unicodedata

try:  # Optional dependency; GPU path uses torch when available.
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from src.global_speaker.metrics import (
    compute_contributions,
    normalized_speaker_entropy,
)


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", value).strip()


def normalize_name(value: str | None) -> str:
    return normalize_text(value).casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="JSON file or directory containing conversation JSON files.",
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="ground_truth",
        help="Field name containing the continuation turns (e.g., ground_truth or prediction).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Embedding device (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding.",
    )
    parser.add_argument(
        "--global-spread-method",
        type=str,
        choices=["kernel-logdet", "graph-mst"],
        default="graph-mst",
        help="Global spread method for marginal contribution.",
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=0.3,
        help="Weight for local novelty when combining contributions.",
    )
    parser.add_argument(
        "--radius-percentile",
        type=float,
        default=85.0,
        help="Percentile for the data-driven radius threshold.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=20,
        help="Maximum number of neighbors per message.",
    )
    parser.add_argument(
        "--kernel-eps",
        type=float,
        default=1e-6,
        help="Diagonal jitter for kernel log-det stability.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/global_speaker/output"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-conversation progress.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, 0), dtype=torch.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embeddings.float()


def build_embeddings(
    texts: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray | "torch.Tensor":
    try:
        if torch is None:
            raise RuntimeError("torch is required for sentence-transformers")
        embedder = SentenceTransformerEmbedder(model_name, device, batch_size)
        return embedder.encode(texts)
    except Exception:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for TF-IDF fallback") from exc
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), analyzer="word")
        matrix = vectorizer.fit_transform(list(texts)).toarray()
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-12, None)
        matrix = matrix / norm
        if torch is not None:
            return torch.from_numpy(matrix.astype(np.float32))
        return matrix.astype(np.float32)


def load_conversations(path: Path, target_field: str) -> List[Dict[str, Any]]:
    if path.is_dir():
        conversations: List[Dict[str, Any]] = []
        for file_path in sorted(path.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                data["_source_path"] = str(file_path)
                conversations.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item["_source_path"] = str(file_path)
                        conversations.append(item)
            else:
                raise ValueError(f"Unsupported JSON structure in {file_path}")
        return conversations

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "context" in data and target_field in data:
            return [data]
        if "conversations" in data:
            return list(data["conversations"])
    raise ValueError(f"Unsupported JSON structure in {path}")


def to_turn_records(raw_turns: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for turn in raw_turns:
        speaker = str(turn.get("speaker", "")).strip()
        utterance = str(turn.get("utterance", "")).strip()
        if not speaker or not utterance:
            continue
        records.append({"speaker": speaker, "utterance": utterance})
    return records


def extract_turns(record: Dict[str, Any], target_field: str) -> List[Dict[str, str]]:
    context = to_turn_records(record.get("context", []))
    target = to_turn_records(record.get(target_field, []))
    return context + target


def conversation_id(record: Dict[str, Any]) -> str | None:
    for key in ("dialogue_id", "conversation_id", "id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    source_path = record.get("_source_path")
    if source_path:
        return Path(str(source_path)).stem
    return None


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(pstdev(values)),
    }


def remap_keys(values: Dict[str, float], display_names: Dict[str, str]) -> Dict[str, float]:
    return {display_names.get(key, key): float(val) for key, val in values.items()}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(args.input_path, args.target_field)
    per_conv_path = args.output_dir / "conversation_scores.jsonl"
    summary_path = args.output_dir / "summary.json"

    nse_values: List[float] = []
    sgini_values: List[float] = []
    sgini_local_values: List[float] = []
    sgini_global_values: List[float] = []

    with per_conv_path.open("w", encoding="utf-8") as out_fp:
        for idx, record in enumerate(conversations):
            turns = extract_turns(record, args.target_field)
            if not turns:
                continue

            utterances = [turn["utterance"] for turn in turns]
            embeddings = build_embeddings(
                utterances,
                model_name=args.embedding_model,
                device=device,
                batch_size=args.batch_size,
            )

            if torch is not None and isinstance(embeddings, torch.Tensor):
                if embeddings.ndim == 1:
                    embeddings = embeddings.unsqueeze(0)
                similarity = embeddings @ embeddings.T
                similarity_np = similarity.detach().cpu().numpy()
                similarity_for_metrics = similarity
            else:
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                similarity_np = embeddings @ embeddings.T
                similarity_for_metrics = similarity_np

            speaker_ids: List[str] = []
            display_names: Dict[str, str] = {}
            for turn in turns:
                norm = normalize_name(turn["speaker"])
                if not norm:
                    continue
                speaker_ids.append(norm)
                display_names.setdefault(norm, turn["speaker"])

            speaker_counts: Dict[str, int] = defaultdict(int)
            for speaker_id in speaker_ids:
                speaker_counts[speaker_id] += 1

            nse = normalized_speaker_entropy(speaker_counts)
            contributions = compute_contributions(
                similarity_for_metrics,
                similarity_np,
                speaker_ids,
                max_neighbors=args.max_neighbors,
                percentile=args.radius_percentile,
                global_method=args.global_spread_method,
                lambda_weight=args.lambda_weight,
                eps=args.kernel_eps,
            )

            nse_values.append(nse)
            sgini_values.append(contributions["sgini"])
            sgini_local_values.append(contributions["sgini_local"])
            sgini_global_values.append(contributions["sgini_global"])

            result = {
                "conversation_id": conversation_id(record),
                "turns": len(turns),
                "unique_speakers": len(speaker_counts),
                "nse": nse,
                "sgini": contributions["sgini"],
                "sgini_local": contributions["sgini_local"],
                "sgini_global": contributions["sgini_global"],
                "global_spread_full": contributions["global_spread_full"],
                "global_spread_method": args.global_spread_method,
                "radius_tau": contributions["radius_tau"],
                "radius_percentile": args.radius_percentile,
                "max_neighbors": args.max_neighbors,
                "lambda_weight": args.lambda_weight,
                "local_novelty_totals": remap_keys(
                    contributions["local_novelty_totals"], display_names
                ),
                "global_spread_deltas": remap_keys(
                    contributions["global_spread_deltas"], display_names
                ),
                "combined_contribution": remap_keys(
                    contributions["combined_contribution"], display_names
                ),
            }
            out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")

            if args.verbose:
                print(f"Scored conversation {idx + 1}/{len(conversations)}")

    summary = {
        "conversations_scored": len(nse_values),
        "nse": summarize(nse_values),
        "sgini": summarize(sgini_values),
        "sgini_local": summarize(sgini_local_values),
        "sgini_global": summarize(sgini_global_values),
        "config": {
            "target_field": args.target_field,
            "embedding_model": args.embedding_model,
            "device": device,
            "global_spread_method": args.global_spread_method,
            "radius_percentile": args.radius_percentile,
            "max_neighbors": args.max_neighbors,
            "lambda_weight": args.lambda_weight,
            "kernel_eps": args.kernel_eps,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
