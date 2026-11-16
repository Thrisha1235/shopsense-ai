# models/

Written by `python -m ml.train`, `ml.calibration` and `ml.explain` (run from `backend/`).
Only this README and `model_manifest.json` are in git. The `.joblib` files are rebuilt by the
pipeline or downloaded from your artifact store (see [DEPLOYMENT.md](../../DEPLOYMENT.md)).

| File | Contents | In git? |
|---|---|---|
| `purchase_model_without_page_values.joblib` | **Served model**: `TunedThresholdClassifierCV` wrapping `Pipeline(preprocess -> RandomForest)`, 16 features | No |
| `purchase_model_with_page_values.joblib` | Benchmark model (17 features, includes PageValues) | No |
| `preprocessor_*.joblib` | The fitted `ColumnTransformer` of each model, saved separately | No |
| `calibrator_*.joblib` | Validated sigmoid calibrator (raw score → probability) | No |
| `model_manifest.json` | Default variant, thresholds, metrics, calibration and SHAP records, and a **SHA-256 for every artifact** | Yes |

## Integrity

joblib files are pickles: loading one can execute code. The API therefore verifies each
artifact's SHA-256 against the manifest **before** unpickling it, and rejects artifact names that
point outside this folder.

```powershell
python -m ml.integrity            # verify all artifacts (exit code 1 on any mismatch)
python -m ml.integrity --record   # re-record checksums after deliberately replacing artifacts
```

Retraining writes new checksums automatically. Only load artifacts that you trained, or that come
from storage you control.
