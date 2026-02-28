#!/usr/bin/env python
"""
Integration tests for NIDS FastAPI service.

These tests run a real uvicorn server process and call HTTP endpoints,
so they validate the API wiring end-to-end.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts_baseline" / "nids_model.joblib"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_health(base_url: str, timeout_s: int = 30) -> None:
    start = time.time()
    while time.time() - start <= timeout_s:
        try:
            result = _http_json("GET", f"{base_url}/health")
            if result.get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError("API did not become healthy in time.")


def _start_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["NIDS_MODEL_PATH"] = str(MODEL_PATH)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api_nids:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _sample_record() -> dict:
    # Minimal feature subset; missing fields are filled with NaN by API.
    return {
        "Protocol": 6,
        "Flow Duration": 12345,
        "Total Fwd Packets": 8,
        "Total Backward Packets": 6,
    }


def test_health_endpoint() -> None:
    assert MODEL_PATH.exists(), f"Missing model file for API tests: {MODEL_PATH}"
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _start_server(port)
    try:
        _wait_for_health(base_url)
        data = _http_json("GET", f"{base_url}/health")
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_model_info_endpoint() -> None:
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _start_server(port)
    try:
        _wait_for_health(base_url)
        data = _http_json("GET", f"{base_url}/model-info")
        assert data["feature_count"] > 0
        assert isinstance(data["classes"], list) and len(data["classes"]) >= 2
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_single_predict_endpoint() -> None:
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _start_server(port)
    try:
        _wait_for_health(base_url)
        payload = {
            "record": _sample_record(),
            "include_probabilities": True,
        }
        data = _http_json("POST", f"{base_url}/predict", payload)
        assert "prediction_label" in data
        assert "prediction_id" in data
        assert "probabilities" in data
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_batch_predict_endpoint() -> None:
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _start_server(port)
    try:
        _wait_for_health(base_url)
        payload = {
            "records": [
                _sample_record(),
                {
                    "Protocol": 17,
                    "Flow Duration": 4000,
                    "Total Fwd Packets": 3,
                    "Total Backward Packets": 1,
                },
            ],
            "include_probabilities": False,
        }
        data = _http_json("POST", f"{base_url}/predict-batch", payload)
        assert data["count"] == 2
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 2
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def run_all_tests() -> None:
    test_health_endpoint()
    test_model_info_endpoint()
    test_single_predict_endpoint()
    test_batch_predict_endpoint()


if __name__ == "__main__":
    run_all_tests()
    print("All API integration checks passed.")
