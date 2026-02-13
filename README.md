# AutoML Workbench (Streamlit)

A practical no-code/low-code ML app for:
- Uploading tabular data
- Exploring data quality and leakage risks
- Configuring preprocessing
- Training and comparing models
- Exporting predictions and model artifacts

<video src="https://github.com/user-attachments/assets/fbfc95c4-5682-438e-94ca-e35074f06c0a" controls width="100%"></video>

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## What This App Does

The UI follows a guided flow:
1. `Data Upload`: CSV, Excel, Parquet + validation and dataset summary
2. `Target & EDA`: choose target/features and inspect distributions/correlations
3. `Warnings`: automatic alerts (leakage, imbalance, missingness, high-cardinality)
4. `Preprocessing`: imputation, encoding, scaling, outlier handling, optional polynomial features
5. `Model Config`: task type + model family + CV/split + tuning controls
6. `Train & Evaluate`: launch background training jobs, compare results, inspect metrics
7. `Predict & Export`: download predictions, metadata, and trusted model artifacts

## Key Features

- Clean separation: UI layer, service/orchestration layer, training/evaluation utilities
- Background training via `ProcessPoolExecutor` (non-blocking Streamlit flow)
- Fingerprint-based caching for heavy dataset summaries
- Smart data warnings with configurable thresholds
- Optional hyperparameter tuning (`RandomizedSearchCV`)
- Safe project state import/export (`JSON`/`YAML`) with strict schema checks

## Project Structure

- `app.py`: app entry point and composition root
- `automl_gui/ui/`: Streamlit rendering modules (upload, EDA, training, export, sidebar)
- `automl_gui/services.py`: business orchestration (`DataService`, `WarningService`, `TrainingService`)
- `automl_gui/core/jobs.py`: background job manager
- `automl_gui/data_utils.py`: file loading, validation, dataset fingerprinting, cached summaries
- `automl_gui/preprocessing.py`: preprocessing config + factory
- `automl_gui/training.py`: model trainer + optimization hooks
- `automl_gui/evaluation.py`: metrics and evaluation helpers
- `automl_gui/models.py`: extensible model registry
- `automl_gui/state.py`: session facade + strict config schema validation
- `automl_gui/warnings_utils.py`: warning generation logic
- `automl_gui/visualization.py`: plotting helpers

## Configuration

Environment variables:

- `AUTOML_MAX_UPLOAD_MB`: max upload size in MB (default `200`)
- `AUTOML_MAX_N_JOBS`: max CPU jobs for supported models/tuning (default `2`)
- `AUTOML_BG_WORKERS`: background worker processes (default `1`)
- `AUTOML_LOG_LEVEL`: logger level (`INFO`, `DEBUG`, ...)
- `AUTOML_LOG_TO_FILE`: set `1` to enable rotating file logs
- `AUTOML_LOG_FILE`: log filename (default `automl_gui.log`)

## Security Notes

- Do not load `.joblib`/`.pkl` files from untrusted sources.
- Model loading in the UI requires explicit trust confirmation.
- Project settings import uses strict schema validation and safe parsing (`yaml.safe_load`).

## Testing

```bash
pytest -q
```

## Troubleshooting

- `ModuleNotFoundError`: verify venv is activated and `pip install -r requirements.txt` completed.
- App feels slow on very large data: enable sampling in Upload tab and reduce selected features/models.
- Training takes too long: disable tuning, lower iterations, and reduce model count.

## Deployment Notes

- Streamlit Community Cloud: push `app.py` + pinned `requirements.txt`.
- Self-hosting: run behind HTTPS reverse proxy and enforce auth at the edge.
