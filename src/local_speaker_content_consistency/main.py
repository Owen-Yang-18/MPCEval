"""CLI to run Section 4.3 local speaker-content consistency metrics.

Expected input JSONL rows contain:
  - context: list[{speaker, utterance}, ...]
  - next_speaker: str
  - next_message: str

Outputs mirror the local speaker CLI:
  - pair_scores.jsonl (per-pair metric values)
  - summary.json (aggregated stats)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from tqdm import tqdm

from src.local_speaker.utils.types import TurnRecord, to_turn_records
from src.local_speaker_content_consistency.utils.scc import (
    CachedTextEmbedder,
    build_embedder,
    fit_tfidf_on_pairs,
    scc_avg,
    scc_max,
    scc_min,
)
from src.local_speaker.utils.embedding_similarity import TfidfEmbedder

DEFAULT_OUTPUT_DIR = Path("src/local_speaker_content_consistency/output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-path",
        type=Path,
        required=True,
        help="JSONL file with {context, next_speaker, next_message}.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Number of previous turns (k) to include when building C_sp.",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="auto",
        choices=["auto", "sentence-transformer", "tfidf"],
        help="Embedding backend selection (default: auto).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name for embedding-based metrics.",
    )
    parser.add_argument(
        "--embedding-cache",
        action="store_true",
        help="Enable an in-memory LRU cache for text embeddings.",
    )
    parser.add_argument(
        "--tfidf-fit-max-pairs",
        type=int,
        default=2000,
        help="When falling back to TF-IDF, fit on up to this many pairs first.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Evaluation mode: single sample or batch over all pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when sampling a single pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store batch evaluation outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages while computing metrics.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def sample_record(path: Path, seed: Optional[int]) -> Dict[str, Any]:
    rng = random.Random(seed)
    chosen: Optional[Dict[str, Any]] = None
    count = 0
    for record in iter_jsonl(path):
        count += 1
        if rng.randrange(count) == 0:
            chosen = record
    if chosen is None:
        raise RuntimeError(f"No pairs found in {path}")
    return chosen


def score_pair(
    *,
    context_records: List[TurnRecord],
    next_speaker: str,
    next_message: str,
    window_size: int,
    embedder,
) -> Dict[str, float | None]:
    return {
        "scc_avg": scc_avg(
            next_message, next_speaker, context_records, window_size=window_size, embedder=embedder
        ),
        "scc_max": scc_max(
            next_message, next_speaker, context_records, window_size=window_size, embedder=embedder
        ),
        "scc_min": scc_min(
            next_message, next_speaker, context_records, window_size=window_size, embedder=embedder
        ),
    }


def run_single(args: argparse.Namespace) -> None:
    record = sample_record(args.pairs_path, args.seed)
    context_records = to_turn_records(record.get("context", []))
    next_speaker = str(record.get("next_speaker", "")).strip()
    next_message = str(record.get("next_message", "")).strip()

    embedder = build_embedder(args.embedding_model, backend=args.embedding_backend)
    if args.embedding_cache:
        embedder = CachedTextEmbedder(embedder.encode).encode  # type: ignore[assignment]

    metrics = score_pair(
        context_records=context_records,
        next_speaker=next_speaker,
        next_message=next_message,
        window_size=args.window_size,
        embedder=embedder,
    )

    skipped = any(value is None for value in metrics.values())
    report = round_data({
        "dialogue_id": record.get("dialogue_id"),
        "window_size": args.window_size,
        "next_speaker": next_speaker,
        "next_message": next_message,
        "metrics": metrics,
        "skipped": skipped,
        "skipped_reason": "no_speaker_history_in_window" if skipped else None,
        "embedding_model": args.embedding_model,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_batch(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = args.output_dir / "pair_scores.jsonl"
    summary_path = args.output_dir / "summary.json"

    embedder_obj = build_embedder(args.embedding_model, backend=args.embedding_backend)
    if isinstance(embedder_obj, TfidfEmbedder):
        fit_tfidf_on_pairs(
            embedder_obj,
            pairs=iter_jsonl(args.pairs_path),
            window_size=args.window_size,
            max_pairs=args.tfidf_fit_max_pairs,
        )
    embedder: Any = embedder_obj
    if args.embedding_cache:
        embedder = CachedTextEmbedder(embedder_obj.encode).encode

    metric_values: Dict[str, List[float]] = defaultdict(list)
    total_seen = 0
    total_scored = 0
    skipped_no_history = 0

    with per_pair_path.open("w", encoding="utf-8") as out_fp:
        progress = tqdm(iter_jsonl(args.pairs_path), disable=not args.verbose, desc="Pairs", unit="pair")
        for record in progress:
            total_seen += 1
            context_records = to_turn_records(record.get("context", []))
            if not context_records:
                continue
            next_speaker = str(record.get("next_speaker", "")).strip()
            next_message = str(record.get("next_message", "")).strip()

            metrics = score_pair(
                context_records=context_records,
                next_speaker=next_speaker,
                next_message=next_message,
                window_size=args.window_size,
                embedder=embedder,
            )

            if any(value is None for value in metrics.values()):
                skipped_no_history += 1
                continue

            output_record = round_data({
                "dialogue_id": record.get("dialogue_id"),
                "context_size": len(context_records),
                "next_speaker": next_speaker,
                "metrics": metrics,
            })
            json.dump(output_record, out_fp, ensure_ascii=False)
            out_fp.write("\n")

            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    metric_values[metric_name].append(float(metric_value))
            total_scored += 1

    summary = round_data({
        "total_pairs_seen": total_seen,
        "total_pairs_scored": total_scored,
        "total_pairs_skipped_no_speaker_history": skipped_no_history,
        "metrics": {name: summarize_metric(values) for name, values in metric_values.items()},
        "embedding_model": args.embedding_model,
        "pairs_report": str(per_pair_path),
    })
    with summary_path.open("w", encoding="utf-8") as summary_fp:
        json.dump(summary, summary_fp, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"[local-scc] Wrote {per_pair_path} and {summary_path}")


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


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
