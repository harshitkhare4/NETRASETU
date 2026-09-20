"""
scratch/test_health_endpoint.py
Tests GET /health and GET /api/health using the Flask test client.
Measures latency and validates response structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
from app import app

def run_tests():
    client = app.test_client()
    
    print("\n--- Testing GET /health ---")
    t0 = time.perf_counter()
    r = client.get('/health')
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    print(f"Status Code: {r.status_code}")
    print(f"Elapsed Time: {elapsed_ms:.3f} ms")
    data = r.get_json()
    print("Response JSON:")
    print(json.dumps(data, indent=2))
    
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
    assert data.get("service") == "NetraSetu", f"Expected service 'NetraSetu', got {data.get('service')}"
    assert data.get("mode") == "production", f"Expected mode 'production', got {data.get('mode')}"
    assert data.get("model") == "NetraSetu_ResNet50_best.pth", f"Expected model 'NetraSetu_ResNet50_best.pth', got {data.get('model')}"
    assert data.get("referable_threshold") == 0.24, f"Expected threshold 0.24, got {data.get('referable_threshold')}"
    print(">>> GET /health: ALL ASSERTIONS PASSED!")
    
    print("\n--- Testing GET /api/health ---")
    t0 = time.perf_counter()
    r_api = client.get('/api/health')
    elapsed_api_ms = (time.perf_counter() - t0) * 1000
    
    print(f"Status Code: {r_api.status_code}")
    print(f"Elapsed Time: {elapsed_api_ms:.3f} ms")
    api_data = r_api.get_json()
    print("Response JSON:")
    print(json.dumps(api_data, indent=2))
    
    assert r_api.status_code == 200, f"Expected 200, got {r_api.status_code}"
    assert api_data.get("status") in ("ready", "ok"), f"Unexpected status {api_data.get('status')}"
    assert api_data.get("classifier_loaded") is True, "Expected classifier_loaded to be True"
    assert "NetraSetu_ResNet50_best.pth" in api_data.get("classifier_model_path", ""), "Expected baseline model path"
    assert api_data.get("locked_referable_threshold") == 0.24, "Expected locked_referable_threshold 0.24"
    print(">>> GET /api/health: ALL ASSERTIONS PASSED!")
    
    print("\n--- Testing GET / (UI Dashboard) ---")
    r_index = client.get('/')
    assert r_index.status_code == 200
    assert "NetraSetu" in r_index.data.decode('utf-8')
    print(">>> GET /: PASSED!")

if __name__ == '__main__':
    run_tests()
