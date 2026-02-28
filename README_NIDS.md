# NIDS Hybrid (AdaBoost + C4.5-style)

This project trains a Network Intrusion Detection System with:
- `C4.5-style Decision Tree` (using entropy criterion)
- `Hybrid AdaBoost + C4.5-style weak learners`

Dataset expected: `Bruteforce-Tuesday-no-metadata.parquet`

## 1) Train

```powershell
python train_nids_hybrid.py
```

Outputs:
- `artifacts/nids_model.joblib`
- `artifacts/metrics.json`

## 2) Binary mode (Benign vs Attack)

```powershell
python train_nids_hybrid.py --binary
```

## 3) Quick smoke run on smaller sample

```powershell
python train_nids_hybrid.py --sample-frac 0.1 --ada-estimators 30
```

## 4) Predict on a parquet/csv file

```powershell
python predict_nids.py --input Bruteforce-Tuesday-no-metadata.parquet --drop-label
```

Output:
- `artifacts/predictions.csv`

## 5) Stage-2: Tune Hybrid Model (GridSearchCV)

```powershell
python tune_nids_hybrid.py --sample-frac 0.2 --cv-folds 5 --out-dir artifacts_tuning
```

Outputs:
- `artifacts_tuning/nids_model_tuned.joblib`
- `artifacts_tuning/tuning_summary.json`
- `artifacts_tuning/cv_results.csv`
- `artifacts_tuning/cv_top_results.csv`

## 6) Stage-2: Detailed Evaluation Artifacts

```powershell
python evaluate_nids_model.py --model artifacts_tuning/nids_model_tuned.joblib --data Bruteforce-Tuesday-no-metadata.parquet --out-dir artifacts_eval
```

Outputs:
- `artifacts_eval/evaluation_report.json`
- `artifacts_eval/predictions_detailed.csv`
- `artifacts_eval/confusion_matrix.csv`
- `artifacts_eval/per_class_metrics.csv`
- `artifacts_eval/roc_curve_points.csv` (if probabilities are available)

## 7) Stage-3: Real-time API (FastAPI)

Default model path used by the API:
- `artifacts_baseline/nids_model.joblib`

You can override it:
```powershell
$env:NIDS_MODEL_PATH="artifacts_tuning_smoke/nids_model_tuned.joblib"
```

Run server:
```powershell
uvicorn api_nids:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`

Single prediction example:
```powershell
$body = @{
  record = @{
    Protocol = 6
    "Flow Duration" = 12345
    "Total Fwd Packets" = 8
    "Total Backward Packets" = 6
  }
  include_probabilities = $true
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

Batch prediction example:
```powershell
$body = @{
  records = @(
    @{
      Protocol = 6
      "Flow Duration" = 12000
      "Total Fwd Packets" = 10
      "Total Backward Packets" = 4
    },
    @{
      Protocol = 17
      "Flow Duration" = 4000
      "Total Fwd Packets" = 3
      "Total Backward Packets" = 1
    }
  )
  include_probabilities = $false
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict-batch" -Method Post -ContentType "application/json" -Body $body
```

## 8) Automated API Integration Tests

Install dev dependencies (includes pytest):
```powershell
python -m pip install -r requirements-dev.txt
```

Run all API tests:
```powershell
python -m pytest test_api_nids.py -v
```

Fallback (without pytest):
```powershell
python test_api_nids.py
```

What these tests verify:
- `/health` endpoint
- `/model-info` endpoint
- `/predict` endpoint
- `/predict-batch` endpoint

## 9) PowerShell API Launcher Script

Default launch:
```powershell
.\start_api.ps1
```

If script execution is blocked on your system:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_api.ps1
```

Launch with a specific model and port:
```powershell
.\start_api.ps1 -ModelPath "artifacts_tuning_smoke/nids_model_tuned.joblib" -ApiHost "127.0.0.1" -Port 8010
```

## 10) Docker Deployment

Build image:
```powershell
docker build -t nids-api:latest .
```

Run container:
```powershell
docker run --rm -p 8000:8000 nids-api:latest
```

Test container health:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

## 11) Deploy Online (Render, from GitHub)

This repo now includes `render.yaml` for one-click deployment.

1. Push this project to a GitHub repository.
2. In Render, choose **New +** -> **Blueprint**.
3. Connect the GitHub repo and select this project.
4. Render reads `render.yaml`, builds the Docker image, and starts the API.
5. After deploy, open:
   - `https://<your-render-service>.onrender.com/health`
   - `https://<your-render-service>.onrender.com/docs`

Notes:
- The container now uses Render's dynamic `PORT` automatically.
- Default model is bundled in the image: `artifacts_baseline/nids_model.joblib`.
- To use a different bundled model, set `NIDS_MODEL_PATH` in Render environment settings.
