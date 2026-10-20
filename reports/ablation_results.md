# Curation Ablation Results

**Base config:** `configs/qlora_config.yaml`  
**Dataset:** `yahma/alpaca-cleaned`  
**Max samples:** `None`  
**Quality model:** `None`  

| Variant | Raw | Curated | Retention | Avg tokens | Median tokens | Avg concision | Median concision | Avg PPL | Median PPL |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 51,760 | 51,760 | 100.00% | 139.37 | 103.0 | 1.3774 | 1.2319 | 0.0 | 0.0 |
| concision_only | 51,760 | 12,956 | 25.03% | 123.14 | 88.0 | 1.1306 | 1.1369 | 0.0 | 0.0 |
| concision_plus_exact_dedup | 51,760 | 12,955 | 25.03% | 123.15 | 88.0 | 1.1306 | 1.1369 | 0.0 | 0.0 |
| concision_plus_near_dedup | 51,760 | 12,955 | 25.03% | 123.15 | 88.0 | 1.1306 | 1.1369 | 0.0 | 0.0 |

## Notes

- `baseline` uses no curation.
- `concision_only` applies the configured concision filtering.
- `*_dedup` variants additionally remove exact and/or near duplicates.
- `*_quality` variants include perplexity scoring with the reference model; ``max_perplexity`` is unset so filtering is not applied.

## JSON summary

```json
[
  {
    "variant": "baseline",
    "raw_examples": 51760,
    "curated_examples": 51760,
    "retention_fraction": 1.0,
    "avg_tokens": 139.37,
    "median_tokens": 103.0,
    "avg_concision": 1.3774,
    "median_concision": 1.2319,
    "avg_quality_perplexity": 0.0,
    "median_quality_perplexity": 0.0
  },
  {
    "variant": "concision_only",
    "raw_examples": 51760,
    "curated_examples": 12956,
    "retention_fraction": 0.2503,
    "avg_tokens": 123.14,
    "median_tokens": 88.0,
    "avg_concision": 1.1306,
    "median_concision": 1.1369,
    "avg_quality_perplexity": 0.0,
    "median_quality_perplexity": 0.0
  },
  {
    "variant": "concision_plus_exact_dedup",
    "raw_examples": 51760,
    "curated_examples": 12955,
    "retention_fraction": 0.2503,
    "avg_tokens": 123.15,
    "median_tokens": 88.0,
    "avg_concision": 1.1306,
    "median_concision": 1.1369,
    "avg_quality_perplexity": 0.0,
    "median_quality_perplexity": 0.0
  },
  {
    "variant": "concision_plus_near_dedup",
    "raw_examples": 51760,
    "curated_examples": 12955,
    "retention_fraction": 0.2503,
    "avg_tokens": 123.15,
    "median_tokens": 88.0,
    "avg_concision": 1.1306,
    "median_concision": 1.1369,
    "avg_quality_perplexity": 0.0,
    "median_quality_perplexity": 0.0
  }
]
```