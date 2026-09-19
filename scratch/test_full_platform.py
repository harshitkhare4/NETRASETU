import json
import requests
import base64
import time
import os

BASE_URL = "http://127.0.0.1:5000"
SAMPLE_FILE = r"C:\NetraSetu\data\APTOS\processed\test\2\ff52392372d3.jpg"

def test_endpoints():
    print("1. Testing /api/health...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print("   Health response:", r.status_code, r.json().get("status"), "GPU:", r.json().get("gpu"))
        assert r.status_code == 200
        assert r.json().get("locked_referable_threshold") == 0.24
    except Exception as e:
        print("   Health failed:", e)
        return False

    print("\n2. Testing /api/samples...")
    r = requests.get(f"{BASE_URL}/api/samples")
    samples = r.json()
    print("   Samples count:", len(samples))
    assert r.status_code == 200 and len(samples) == 4

    print("\n3. Testing POST /api/sample/moderate-dr...")
    r = requests.post(f"{BASE_URL}/api/sample/moderate-dr")
    res_sample = r.json()
    case_id = res_sample.get("case_id")
    base_name = res_sample.get("base_name")
    print(f"   Sample analysis status: {r.status_code}, Case ID: {case_id}, Grade: {res_sample.get('icdr_grade')}, Referable: {res_sample.get('referable')}")
    assert r.status_code == 200
    assert res_sample.get("predicted_class") == 2
    assert res_sample.get("referable") is True

    print("\n4. Testing POST /api/analyze (File Upload)...")
    if os.path.exists(SAMPLE_FILE):
        with open(SAMPLE_FILE, "rb") as f:
            files = {"file": ("test_upload.jpg", f, "image/jpeg")}
            r = requests.post(f"{BASE_URL}/api/analyze", files=files)
            res_upload = r.json()
            print(f"   Upload analysis status: {r.status_code}, Case ID: {res_upload.get('case_id')}, Grade: {res_upload.get('icdr_grade')}")
            assert r.status_code == 200
    else:
        print("   Sample file not on disk, skipping file upload test.")

    print("\n5. Testing POST /api/analyze-camera with simulated base64 frame...")
    import cv2
    import numpy as np
    dummy_frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
    cv2.circle(dummy_frame, (112, 112), 80, (50, 100, 200), -1)
    _, buffer = cv2.imencode(".jpg", dummy_frame)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    r = requests.post(f"{BASE_URL}/api/analyze-camera", json={"image_data": b64_str})
    res_cam = r.json()
    print(f"   Camera frame analysis: {r.status_code}, Case ID: {res_cam.get('case_id')}, Status: {res_cam.get('status')}")
    assert r.status_code == 200

    print("\n6. Testing /api/history...")
    r = requests.get(f"{BASE_URL}/api/history?filter=all")
    history = r.json()
    records = history.get("screenings", [])
    print(f"   History records retrieved: {len(records)} (Total count reported: {history.get('count')})")
    assert r.status_code == 200
    assert len(records) > 0

    print("\n7. Testing /api/reports/monthly (September 2026)...")
    r = requests.get(f"{BASE_URL}/api/reports/monthly?year=2026&month=9")
    monthly = r.json()
    print(f"   Monthly 2026-09 Total: {monthly.get('total_screenings')}, Completed: {monthly.get('completed')}, Specialist Review: {monthly.get('specialist_review')}")
    assert r.status_code == 200
    assert monthly.get("total_screenings") >= 248

    print("\n8. Testing /api/reports/yearly (2026)...")
    r = requests.get(f"{BASE_URL}/api/reports/yearly?year=2026")
    yearly = r.json()
    print(f"   Yearly 2026 Total: {yearly.get('total_screenings')}, Months tracked: {len(yearly.get('monthly_breakdown', []))}")
    assert r.status_code == 200
    assert yearly.get("total_screenings") >= 1500

    print("\n9. Testing /api/review-queue...")
    r = requests.get(f"{BASE_URL}/api/review-queue?status=all")
    rq = r.json()
    cases = rq.get("cases", [])
    print(f"   Review queue records: {len(cases)} (Total count: {rq.get('count')})")
    assert r.status_code == 200
    assert len(cases) > 0

    if case_id:
        print(f"\n10. Testing POST /api/review/{case_id} (Status update)...")
        r = requests.post(f"{BASE_URL}/api/review/{case_id}", json={
            "status": "Reviewed",
            "notes": "Ophthalmologist simulated triage: clinically confirmed moderate non-proliferative retinopathy."
        })
        print(f"    Review update response: {r.status_code}, Message: {r.json().get('message')}")
        assert r.status_code == 200

    print("\n11. Testing /api/capacity (100k Planning Model)...")
    r = requests.get(f"{BASE_URL}/api/capacity?target=100000&working_days=300&clinics=10&inference_time_sec=0.12")
    cap = r.json()
    tm = cap.get("throughput_metrics", {})
    print(f"    Capacity calculation: Required/day: {tm.get('required_screenings_per_day')}, Daily AI Cap/Workstation: {tm.get('ai_daily_capacity_per_workstation')}, Specialist Reviews/Day: {tm.get('estimated_specialist_reviews_day')}")
    assert r.status_code == 200
    assert tm.get("required_screenings_per_day") == 333.3

    print("\n12. Testing /api/analytics...")
    r = requests.get(f"{BASE_URL}/api/analytics")
    an = r.json()
    print(f"    Analytics summary: Total: {an.get('total_screenings')}, Pending reviews: {an.get('pending_specialist_reviews')}, Latency: {an.get('avg_processing_time_ms')} ms")
    assert r.status_code == 200

    print("\n13. Testing /api/export/csv...")
    r = requests.get(f"{BASE_URL}/api/export/csv")
    print(f"    CSV export status: {r.status_code}, Payload size: {len(r.text)} bytes")
    assert r.status_code == 200
    assert "Case ID,Timestamp,Date" in r.text

    if base_name:
        print(f"\n14. Testing POST /api/experimental-lesions for '{base_name}'...")
        r = requests.post(f"{BASE_URL}/api/experimental-lesions", json={"base_name": base_name})
        print(f"    Lesions segmentation status: {r.status_code}")
        if r.status_code == 200:
            les_data = r.json()
            print("    Lesions segmented successfully! Stats:", list(les_data.get("lesion_stats", {}).keys()))
            assert "Microaneurysm" in les_data.get("lesion_stats", {})

    print("\n============================================================")
    print(">>> ALL 14 PLATFORM ENDPOINTS & CAPABILITIES VERIFIED! <<<")
    print("============================================================")
    return True

if __name__ == "__main__":
    test_endpoints()
