#!/usr/bin/env python3
"""Run data-side curation ablations and produce a comparison report.

This script compares different curation strategies on the raw instruction
 dataset. It does **not** train models, so it can run on CPU; the resulting
report shows how each strategy affects dataset size and example statistics.

For quality-scoring ablations, pass ``--quality_model`` pointing to a small
reference model (e.g. ``Qwen/Qwen2.5-0.5B-Instruct``).
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_instruction_dataset
from src.data_curation import build_distillation_dataset
from src.utils import load_config


def build_variant_config(
    base_config: dict[str, Any],
    name: str,
    curation_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Create a config variant by overriding curation settings."""
    variant = deepcopy(base_config)
    variant["data"]["curation"] = {
        **variant["data"].get("curation", {}),
        "enabled": True,
        **curation_overrides,
    }
    return variant


def load_quality_model(quality_cfg: dict[str, Any]) -> Any:
    """Load a reference model for perplexity scoring."""
    model_name = quality_cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def compute_stats(dataset: Any, tokenizer: Any, response_field: str) -> dict[str, float]:
    """Compute basic statistics for a curated dataset."""
    outputs = [str(ex.get(response_field, "")).strip() for ex in dataset]
    tokens = [len(tokenizer.encode(text, add_special_tokens=False)) for text in outputs]
    words = [len(text.split()) for text in outputs]
    concision = [t / max(w, 1) for t, w in zip(tokens, words, strict=False)]

    stats = {
        "num_examples": len(dataset),
        "avg_tokens": float(np.mean(tokens)) if tokens else 0.0,
        "median_tokens": float(np.median(tokens)) if tokens else 0.0,
        "avg_words": float(np.mean(words)) if words else 0.0,
        "median_words": float(np.median(words)) if words else 0.0,
        "avg_concision": float(np.mean(concision)) if concision else 0.0,
        "median_concision": float(np.median(concision)) if concision else 0.0,
    }

    if "quality_score" in dataset.column_names:
        quality = dataset["quality_score"]
        stats["avg_quality_perplexity"] = float(np.mean(quality))
        stats["median_quality_perplexity"] = float(np.median(quality))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Curation ablation runner")
    parser.add_argument("--config", default="configs/qlora_config.yaml", help="Base config")
    parser.add_argument("--output", default="reports/ablation_results.md", help="Report path")
    parser.add_argument("--csv", default="reports/ablation_results.csv", help="CSV path")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for speed")
    parser.add_argument("--quality_model", default=None, help="Optional reference model for quality scoring")
    args = parser.parse_args()

    base_config = load_config(args.config)
    data_config = base_config["data"]
    response_field = data_config.get("output_field", "output")

    tokenizer = AutoTokenizer.from_pretrained(
        base_config["model"]["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define ablation variants.
    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline", {"enabled": False}),
        ("concision_only", {"deduplication": {"enabled": False}, "quality_scoring": {"enabled": False}}),
        ("concision_plus_exact_dedup", {"deduplication": {"enabled": True, "exact": True, "near": False}, "quality_scoring": {"enabled": False}}),
        ("concision_plus_near_dedup", {"deduplication": {"enabled": True, "exact": True, "near": True}, "quality_scoring": {"enabled": False}}),
    ]

    quality_model = None
    quality_tokenizer = None
    if args.quality_model:
        variants.extend([
            ("concision_plus_quality", {"deduplication": {"enabled": False}, "quality_scoring": {"enabled": True, "model": args.quality_model, "max_perplexity": None}}),
            ("full_pipeline", {"deduplication": {"enabled": True, "exact": True, "near": True}, "quality_scoring": {"enabled": True, "model": args.quality_model, "max_perplexity": None}}),
        ])
        quality_model, quality_tokenizer = load_quality_model({"model": args.quality_model})

    records: list[dict[str, Any]] = []

    for name, overrides in variants:
        config = build_variant_config(base_config, name, overrides)
        if args.max_samples is not None:
            config["data"]["max_samples"] = args.max_samples

        raw = load_instruction_dataset(
            dataset_name=config["data"]["dataset_name"],
            config_name=config["data"].get("dataset_config"),
            split=config["data"].get("split", "train"),
            cache_dir=config["data"].get("cache_dir"),
            streaming=config["data"].get("streaming", False),
            max_samples=config["data"].get("max_samples"),
        )
        raw_stats = compute_stats(raw, tokenizer, response_field)

        curated = build_distillation_dataset(
            dataset=raw,
            teacher_model=None,
            tokenizer=tokenizer,
            data_config=config["data"],
            quality_model=quality_model,
        )
        curated_stats = compute_stats(curated, tokenizer, response_field)

        records.append({
            "variant": name,
            "raw_examples": raw_stats["num_examples"],
            "curated_examples": curated_stats["num_examples"],
            "retention_fraction": round(curated_stats["num_examples"] / max(raw_stats["num_examples"], 1), 4),
            "avg_tokens": round(curated_stats["avg_tokens"], 2),
            "median_tokens": round(curated_stats["median_tokens"], 2),
            "avg_concision": round(curated_stats["avg_concision"], 4),
            "median_concision": round(curated_stats["median_concision"], 4),
            "avg_quality_perplexity": round(curated_stats.get("avg_quality_perplexity", 0.0), 4),
            "median_quality_perplexity": round(curated_stats.get("median_quality_perplexity", 0.0), 4),
        })

    # Write CSV.
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Write markdown report.
    report_lines = [
        "# Curation Ablation Results",
        "",
        f"**Base config:** `{args.config}`  ",
        f"**Dataset:** `{data_config['dataset_name']}`  ",
        f"**Max samples:** `{args.max_samples}`  ",
        f"**Quality model:** `{args.quality_model}`  ",
        "",
        "| Variant | Raw | Curated | Retention | Avg tokens | Median tokens | Avg concision | Median concision | Avg PPL | Median PPL |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        report_lines.append(
            f"| {r['variant']} | {r['raw_examples']:,} | {r['curated_examples']:,} | "
            f"{r['retention_fraction']:.2%} | {r['avg_tokens']} | {r['median_tokens']} | "
            f"{r['avg_concision']} | {r['median_concision']} | {r['avg_quality_perplexity']} | {r['median_quality_perplexity']} |"
        )

    report_lines.extend([
        "",
        "## Notes",
        "",
        "- `baseline` uses no curation.",
        "- `concision_only` applies the configured concision filtering.",
        "- `*_dedup` variants additionally remove exact and/or near duplicates.",
        "- `*_quality` variants include perplexity scoring with the reference model; ``max_perplexity`` is unset so filtering is not applied.",
        "",
        "## JSON summary",
        "",
        "```json",
        json.dumps(records, indent=2),
        "```",
    ])

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Ablation CSV written to: {csv_path}")
    print(f"Ablation report written to: {report_path}")


if __name__ == "__main__":
    main()
