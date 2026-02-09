"""CLI for global centroid-based speaker-content consistency metrics."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from src.global_consistency.metrics import compute_speaker_scores, to_numpy_embeddings, weighted_aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="JSON file with context and generated turns.",
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="prediction",
        help="Field name containing the generated turns.",
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
        default="cuda",
        help="Embedding device (default: cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Upper bound for GMM components per speaker.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum messages required to consider multi-centroid clustering.",
    )
    parser.add_argument(
        "--min-k-large",
        type=int,
        default=2,
        help="Minimum k for speakers with many messages.",
    )
    parser.add_argument(
        "--large-threshold",
        type=int,
        default=25,
        help="Message count threshold for enforcing min-k.",
    )
    parser.add_argument(
        "--min-messages-per-cluster",
        type=int,
        default=5,
        help="Minimum messages required per cluster when selecting k.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for GMM initialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/global_consistency/output/run"),
        help="Output directory for the summary JSON.",
    )
    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", value).strip()


def normalize_name(value: str | None) -> str:
    return normalize_text(value).casefold()


def to_turn_records(raw_turns: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for turn in raw_turns:
        speaker = str(turn.get("speaker", "")).strip()
        utterance = str(turn.get("utterance", "")).strip()
        if not speaker or not utterance:
            continue
        records.append({"speaker": speaker, "utterance": utterance})
    return records


def load_conversations(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        conversations: List[Dict[str, Any]] = []
        for file_path in sorted(path.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                data["_source_path"] = str(file_path)
                conversations.append(data)
            else:
                raise ValueError(f"Expected a JSON object in {file_path}")
        return conversations

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Expected a JSON object in {path}")


def conversation_id(record: Dict[str, Any]) -> str | None:
    for key in ("dialogue_id", "conversation_id", "id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    source_path = record.get("_source_path")
    if source_path:
        return Path(str(source_path)).stem
    return None


def embed_utterances(
    model: SentenceTransformer,
    utterances: Sequence[str],
    batch_size: int,
) -> torch.Tensor:
    embeddings = model.encode(
        list(utterances),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return embeddings.float()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(args.input_path)
    per_conv_path = args.output_dir / "conversation_scores.jsonl"
    summary_path = args.output_dir / "summary.json"

    single_avg_values: List[float] = []
    single_max_values: List[float] = []
    multi_avg_values: List[float] = []
    multi_max_values: List[float] = []
    skipped = 0

    model = SentenceTransformer(args.embedding_model, device=args.device)

    with per_conv_path.open("w", encoding="utf-8") as out_fp:
        for record in conversations:
            generated = to_turn_records(record.get(args.target_field, []))
            convo_id = conversation_id(record)
            if not generated:
                skipped += 1
                out_fp.write(
                    json.dumps(
                        {
                            "conversation_id": convo_id,
                            "turns": 0,
                            "unique_speakers": 0,
                            "skipped": True,
                            "skipped_reason": "no_generated_turns",
                            "per_speaker": {},
                            "aggregate": {"single": {"avg": 0.0, "max": 0.0}, "multi": {"avg": 0.0, "max": 0.0}},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            utterances = [turn["utterance"] for turn in generated]
            embeddings = embed_utterances(
                model,
                utterances,
                batch_size=args.batch_size,
            )
            matrix = to_numpy_embeddings(embeddings)

            speaker_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
            display_names: Dict[str, str] = {}
            for turn, vector in zip(generated, matrix):
                norm = normalize_name(turn["speaker"])
                if not norm:
                    continue
                speaker_embeddings[norm].append(vector)
                display_names.setdefault(norm, turn["speaker"])

            speaker_arrays: Dict[str, np.ndarray] = {
                speaker: np.vstack(vectors) for speaker, vectors in speaker_embeddings.items()
            }
            scores = compute_speaker_scores(
                speaker_arrays,
                max_k=args.max_k,
                min_cluster_size=args.min_cluster_size,
                random_state=args.random_state,
                min_k_large=args.min_k_large,
                large_threshold=args.large_threshold,
                min_messages_per_cluster=args.min_messages_per_cluster,
            )
            aggregate = weighted_aggregate(scores)

            per_speaker: Dict[str, Dict[str, Any]] = {}
            for speaker, score in scores.items():
                name = display_names.get(speaker, speaker)
                per_speaker[name] = {
                    "count": score.count,
                    "single": {"avg": score.single_avg, "max": score.single_max},
                    "multi": {"avg": score.multi_avg, "max": score.multi_max, "k": score.k_selected},
                }

            single_avg_values.append(aggregate["single"]["avg"])
            single_max_values.append(aggregate["single"]["max"])
            multi_avg_values.append(aggregate["multi"]["avg"])
            multi_max_values.append(aggregate["multi"]["max"])

            out_fp.write(
                json.dumps(
                    {
                        "conversation_id": convo_id,
                        "turns": len(generated),
                        "unique_speakers": len(per_speaker),
                        "skipped": False,
                        "per_speaker": per_speaker,
                        "aggregate": aggregate,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def summarize(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "mean": float(mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "std": float(pstdev(values)),
        }

    summary = {
        "conversations_scored": len(conversations) - skipped,
        "conversations_skipped": skipped,
        "aggregate": {
            "single": {
                "avg": summarize(single_avg_values),
                "max": summarize(single_max_values),
            },
            "multi": {
                "avg": summarize(multi_avg_values),
                "max": summarize(multi_max_values),
            },
        },
        "config": {
            "target_field": args.target_field,
            "embedding_model": args.embedding_model,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_k": args.max_k,
            "min_cluster_size": args.min_cluster_size,
            "random_state": args.random_state,
            "min_k_large": args.min_k_large,
            "large_threshold": args.large_threshold,
            "min_messages_per_cluster": args.min_messages_per_cluster,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
