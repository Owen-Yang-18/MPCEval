"""CLI for global unsupervised content metrics (Section 5.2.2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence

import torch
from sentence_transformers import SentenceTransformer

from src.global_content_unsupervised.metrics import (
    harmonic_mean_progression,
    mean_step_distance,
    progression_distance,
    raw_displacement,
)


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
        default="prediction",
        help="Field name containing the generated turns (prediction, ground_truth, etc.).",
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
        "--epsilon",
        type=float,
        default=1e-2,
        help="Epsilon to stabilize harmonic mean progression.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/global_content_unsupervised/output/run"),
        help="Output directory for the summary JSON.",
    )
    return parser.parse_args()


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
    utterances: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(utterances),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    return embeddings.float()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(args.input_path)
    per_conv_path = args.output_dir / "conversation_scores.jsonl"
    summary_path = args.output_dir / "summary.json"

    pd_values: List[float] = []
    hmp_values: List[float] = []
    raw_disp_values: List[float] = []
    mean_step_values: List[float] = []
    skipped = 0

    with per_conv_path.open("w", encoding="utf-8") as out_fp:
        for record in conversations:
            generated = to_turn_records(record.get(args.target_field, []))
            utterances = [turn["utterance"] for turn in generated]
            convo_id = conversation_id(record)

            if len(utterances) < 2:
                skipped += 1
                result = {
                    "conversation_id": convo_id,
                    "turns": len(utterances),
                    "skipped": True,
                    "skipped_reason": "need_at_least_two_generated_turns",
                    "pd": None,
                    "hmp": None,
                    "raw_displacement": None,
                    "mean_step_distance": None,
                }
                out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue

            embeddings = embed_utterances(
                utterances,
                model_name=args.embedding_model,
                device=args.device,
                batch_size=args.batch_size,
            )

            pd_score = progression_distance(embeddings)
            hmp_score = harmonic_mean_progression(embeddings, epsilon=args.epsilon)
            raw_disp = raw_displacement(embeddings)
            mean_step = mean_step_distance(embeddings)

            if pd_score is not None:
                pd_values.append(pd_score)
            if hmp_score is not None:
                hmp_values.append(hmp_score)
            if raw_disp is not None:
                raw_disp_values.append(raw_disp)
            if mean_step is not None:
                mean_step_values.append(mean_step)

            result = {
                "conversation_id": convo_id,
                "turns": len(utterances),
                "skipped": False,
                "pd": pd_score,
                "hmp": hmp_score,
                "raw_displacement": raw_disp,
                "mean_step_distance": mean_step,
            }
            out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")

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
        "pd": summarize(pd_values),
        "hmp": summarize(hmp_values),
        "raw_displacement": summarize(raw_disp_values),
        "mean_step_distance": summarize(mean_step_values),
        "config": {
            "target_field": args.target_field,
            "embedding_model": args.embedding_model,
            "device": args.device,
            "batch_size": args.batch_size,
            "epsilon": args.epsilon,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
