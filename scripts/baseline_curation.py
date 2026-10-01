#!/usr/bin/env python3
"""Generate a baseline report for the existing data-curation pipeline.

This script documents the current curation behavior *before* any improvements
are made, so that future changes can be compared against a measured baseline.

It loads the raw instruction dataset, applies the current concision-based
curation pipeline, and reports:
  - raw dataset statistics
  - how many examples survive each curation gate
  - the distribution of concision scores
  - any integration issues between formatting and curation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from src.data import build_dataset, load_instruction_dataset
from src.data_curation import curate_dataset
from src.utils import load_config


def load_raw_dataset(data_config: dict[str, Any]) -> Any:
    """Load the instruction dataset with original columns intact."""
    return load_instruction_dataset(
        dataset_name=data_config["dataset_name"],
        config_name=data_config.get("dataset_config"),
        split=data_config.get("split", "train"),
        cache_dir=data_config.get("cache_dir"),
        streaming=data_config.get("streaming", False),
        max_samples=data_config.get("max_samples"),
    )


def token_counts(texts: list[str], tokenizer: Any) -> list[int]:
    """Return token counts for a list of texts."""
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def word_counts(texts: list[str]) -> list[int]:
    """Return word counts for a list of texts."""
    return [len(t.split()) for t in texts]


def compute_bucket_stats(values: list[float], buckets: list[float]) -> dict[str, int]:
    """Count how many values fall into each bucket."""
    counts: dict[str, int] = {}
    for low, high in zip([0.0] + buckets, buckets + [float("inf")], strict=False):
        key = f"{low:.2f} - {high:.2f}" if high != float("inf") else f"> {low:.2f}"
        counts[key] = sum(1 for v in values if low <= v < high)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline curation report")
    parser.add_argument("--config", default="configs/qlora_config.yaml", help="Path to YAML config")
    parser.add_argument("--output", default="reports/baseline_curation_report.md", help="Report path")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name or path (defaults to config model base_model)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    curation_cfg = data_config.get("curation", {})
    response_field = data_config.get("output_field", "output")

    keep_top_fraction = curation_cfg.get("keep_top_fraction", 0.8)
    max_tokens_per_word = curation_cfg.get("max_tokens_per_word")
    min_tokens_per_example = curation_cfg.get("min_tokens_per_example", 4)

    tokenizer_name = args.tokenizer or config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Raw dataset
    # ------------------------------------------------------------------
    raw_dataset = load_raw_dataset(data_config)
    num_raw = len(raw_dataset)

    outputs = [str(ex.get(response_field, "")).strip() for ex in raw_dataset]
    tokens = token_counts(outputs, tokenizer)
    words = word_counts(outputs)
    concision_scores = [t / max(w, 1) for t, w in zip(tokens, words, strict=True)]

    raw_stats = {
        "num_examples": num_raw,
        "avg_output_tokens": float(np.mean(tokens)),
        "median_output_tokens": float(np.median(tokens)),
        "avg_output_words": float(np.mean(words)),
        "median_output_words": float(np.median(words)),
        "avg_concision_score": float(np.mean(concision_scores)),
        "median_concision_score": float(np.median(concision_scores)),
    }

    # ------------------------------------------------------------------
    # 2. Per-gate impact
    # ------------------------------------------------------------------
    min_token_mask = [t >= min_tokens_per_example for t in tokens]
    max_tpw_mask = [
        (max_tokens_per_word is None) or (c <= max_tokens_per_word) for c in concision_scores
    ]
    both_gates_mask = [a and b for a, b in zip(min_token_mask, max_tpw_mask, strict=True)]

    passed_both = sum(both_gates_mask)
    if passed_both > 0:
        kept_scores = [c for c, keep in zip(concision_scores, both_gates_mask, strict=True) if keep]
        threshold = float(np.percentile(kept_scores, 100 * (1 - keep_top_fraction)))
        final_mask = [keep and (c <= threshold) for c, keep in zip(concision_scores, both_gates_mask, strict=True)]
    else:
        threshold = float("nan")
        final_mask = [False] * num_raw

    num_after_min_token = sum(min_token_mask)
    num_after_max_tpw = sum(both_gates_mask)
    num_after_percentile = sum(final_mask)

    # ------------------------------------------------------------------
    # 3. Apply the actual curate_dataset function for comparison
    # ------------------------------------------------------------------
    curated_dataset = curate_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        response_field=response_field,
        keep_top_fraction=keep_top_fraction,
        max_tokens_per_word=max_tokens_per_word,
        min_tokens_per_example=min_tokens_per_example,
    )
    num_curated = len(curated_dataset)

    # ------------------------------------------------------------------
    # 4. Concision score distribution
    # ------------------------------------------------------------------
    score_buckets = compute_bucket_stats(concision_scores, [1.0, 1.5, 2.0, 2.5, 3.0, 5.0])

    # ------------------------------------------------------------------
    # 5. Integration check: build_dataset removes columns before curation
    # ------------------------------------------------------------------
    formatted_dataset = build_dataset(
        data_config=data_config,
        tokenizer=tokenizer,
        eval_size=0,
        seed=config["training"].get("seed", 42),
    )["train"]
    has_output_column = response_field in formatted_dataset.column_names

    # ------------------------------------------------------------------
    # 6. Write report
    # ------------------------------------------------------------------
    report_lines = [
        "# Data-Curation Baseline Report",
        "",
        f"**Config:** `{args.config}`  ",
        f"**Dataset:** `{data_config['dataset_name']}`  ",
        f"**Response field:** `{response_field}`  ",
        f"**Tokenizer:** `{tokenizer_name}`  ",
        "",
        "## Current curation settings",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| `enabled` | `{curation_cfg.get('enabled', False)}` |",
        f"| `keep_top_fraction` | `{keep_top_fraction}` |",
        f"| `max_tokens_per_word` | `{max_tokens_per_word}` |",
        f"| `min_tokens_per_example` | `{min_tokens_per_example}` |",
        "",
        "## Raw dataset statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Examples | {raw_stats['num_examples']:,} |",
        f"| Avg response tokens | {raw_stats['avg_output_tokens']:.2f} |",
        f"| Median response tokens | {raw_stats['median_output_tokens']:.2f} |",
        f"| Avg response words | {raw_stats['avg_output_words']:.2f} |",
        f"| Median response words | {raw_stats['median_output_words']:.2f} |",
        f"| Avg concision score | {raw_stats['avg_concision_score']:.4f} |",
        f"| Median concision score | {raw_stats['median_concision_score']:.4f} |",
        "",
        "## Per-gate impact",
        "",
        "| Stage | Examples retained | Fraction retained |",
        "|---|---|---|",
        f"| Raw | {num_raw:,} | 100.00% |",
        f"| After `min_tokens_per_example >= {min_tokens_per_example}` | {num_after_min_token:,} | {100*num_after_min_token/num_raw:.2f}% |",
    ]

    if max_tokens_per_word is not None:
        report_lines.append(
            f"| After `concision_score <= {max_tokens_per_word}` | {num_after_max_tpw:,} | {100*num_after_max_tpw/num_raw:.2f}% |"
        )
    else:
        report_lines.append("| After `max_tokens_per_word` | skipped (not set) | — |")

    report_lines.extend([
        f"| After `keep_top_fraction={keep_top_fraction}` percentile | {num_after_percentile:,} | {100*num_after_percentile/num_raw:.2f}% |",
        f"| Actual `curate_dataset` output | {num_curated:,} | {100*num_curated/num_raw:.2f}% |",
        "",
        "## Concision-score distribution",
        "",
        "| Tokens / word bucket | Count | Fraction |",
        "|---|---|---|",
    ])

    for bucket, count in score_buckets.items():
        report_lines.append(f"| {bucket} | {count:,} | {100*count/num_raw:.2f}% |")

    report_lines.extend([
        "",
        "## Integration issue check",
        "",
        f"- Formatted dataset still has `{response_field}` column: **{has_output_column}**",
        "- `build_dataset` removes original columns and creates `text` before `curate_dataset` is called in `src/train.py`.",
        "- Therefore, unless curation is run on the raw dataset first, the current `train.py` ordering will fail to access the response field.",
        "",
        "## JSON summary",
        "",
        "```json",
        json.dumps(
            {
                "config": args.config,
                "dataset": data_config["dataset_name"],
                "settings": {
                    "keep_top_fraction": keep_top_fraction,
                    "max_tokens_per_word": max_tokens_per_word,
                    "min_tokens_per_example": min_tokens_per_example,
                },
                "raw_stats": raw_stats,
                "retention": {
                    "raw": num_raw,
                    "after_min_token_gate": num_after_min_token,
                    "after_max_tpw_gate": num_after_max_tpw,
                    "after_percentile": num_after_percentile,
                    "curate_dataset_output": num_curated,
                },
                "score_distribution": score_buckets,
                "integration_issue": {
                    "formatted_dataset_has_response_column": has_output_column,
                },
            },
            indent=2,
        ),
        "```",
    ])

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Baseline report written to: {output_path}")


if __name__ == "__main__":
    main()
