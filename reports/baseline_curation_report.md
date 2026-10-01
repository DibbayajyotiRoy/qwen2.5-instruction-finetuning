# Data-Curation Baseline Report

**Config:** `configs/qlora_config.yaml`  
**Dataset:** `yahma/alpaca-cleaned`  
**Response field:** `output`  
**Tokenizer:** `Qwen/Qwen2.5-0.5B-Instruct`  

## Current curation settings

| Setting | Value |
|---|---|
| `enabled` | `False` |
| `keep_top_fraction` | `0.75` |
| `max_tokens_per_word` | `2.5` |
| `min_tokens_per_example` | `4` |

## Raw dataset statistics

| Metric | Value |
|---|---|
| Examples | 51,760 |
| Avg response tokens | 139.37 |
| Median response tokens | 103.00 |
| Avg response words | 109.94 |
| Median response words | 80.00 |
| Avg concision score | 1.3774 |
| Median concision score | 1.2319 |

## Per-gate impact

| Stage | Examples retained | Fraction retained |
|---|---|---|
| Raw | 51,760 | 100.00% |
| After `min_tokens_per_example >= 4` | 51,063 | 98.65% |
| After `concision_score <= 2.5` | 50,048 | 96.69% |
| After `keep_top_fraction=0.75` percentile | 12,957 | 25.03% |
| Actual `curate_dataset` output | 12,956 | 25.03% |

## Concision-score distribution

| Tokens / word bucket | Count | Fraction |
|---|---|---|
| 0.00 - 1.00 | 0 | 0.00% |
| 1.00 - 1.50 | 41,815 | 80.79% |
| 1.50 - 2.00 | 6,481 | 12.52% |
| 2.00 - 2.50 | 2,155 | 4.16% |
| 2.50 - 3.00 | 576 | 1.11% |
| 3.00 - 5.00 | 572 | 1.11% |
| > 5.00 | 161 | 0.31% |

## Integration issue check

- Formatted dataset still has `output` column: **False**
- `build_dataset` removes original columns and creates `text` before `curate_dataset` is called in `src/train.py`.
- Therefore, unless curation is run on the raw dataset first, the current `train.py` ordering will fail to access the response field.

## JSON summary

```json
{
  "config": "configs/qlora_config.yaml",
  "dataset": "yahma/alpaca-cleaned",
  "settings": {
    "keep_top_fraction": 0.75,
    "max_tokens_per_word": 2.5,
    "min_tokens_per_example": 4
  },
  "raw_stats": {
    "num_examples": 51760,
    "avg_output_tokens": 139.36765842349305,
    "median_output_tokens": 103.0,
    "avg_output_words": 109.9395285935085,
    "median_output_words": 80.0,
    "avg_concision_score": 1.3774026865293878,
    "median_concision_score": 1.2318840579710144
  },
  "retention": {
    "raw": 51760,
    "after_min_token_gate": 51063,
    "after_max_tpw_gate": 50048,
    "after_percentile": 12957,
    "curate_dataset_output": 12956
  },
  "score_distribution": {
    "0.00 - 1.00": 0,
    "1.00 - 1.50": 41815,
    "1.50 - 2.00": 6481,
    "2.00 - 2.50": 2155,
    "2.50 - 3.00": 576,
    "3.00 - 5.00": 572,
    "> 5.00": 161
  },
  "integration_issue": {
    "formatted_dataset_has_response_column": false
  }
}
```