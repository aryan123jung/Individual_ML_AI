# Project Structure

## `src/`

- `run_full_pipeline.py`
  Main entry point for the end-to-end pipeline.

- `pipelines/engineering/`
  Data cleaning and feature engineering for historical IPL, IPL 2026, and SMAT.

- `pipelines/ml/`
  Supervised model training and candidate ML scoring.

- `pipelines/reporting/`
  Report generation and workbook export.

- `recommendation_engine/`
  IPL benchmark creation, similarity scoring, squad-gap detection, replacements, and final recommendations.

## Compatibility entry files

The old top-level script names are kept as thin wrappers so existing commands still run while the internal architecture stays clean.
