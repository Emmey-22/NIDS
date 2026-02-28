param(
  [string]$ModelPath = "artifacts_baseline/nids_model.joblib",
  [string]$ApiHost = "127.0.0.1",
  [int]$Port = 8000
)

Set-StrictMode -Version Latest
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not [System.IO.Path]::IsPathRooted($ModelPath)) {
  $ModelPath = Join-Path $scriptDir $ModelPath
}

if (-not (Test-Path $ModelPath)) {
  Write-Error "Model file not found: $ModelPath"
  exit 1
}

$env:NIDS_MODEL_PATH = $ModelPath
Write-Host "Starting NIDS API with model: $ModelPath"
Write-Host "Host: $ApiHost  Port: $Port"
python -m uvicorn api_nids:app --host $ApiHost --port $Port
