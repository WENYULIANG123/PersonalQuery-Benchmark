"""
Test the Error Detection API
"""

import json
from api_service import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed")


def test_single_comment_with_error():
    response = client.post("/check", json={
        "comment": "Teh product is excellent",
        "threshold": 0.5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["analysis"]["num_errors"] == 1
    print("✓ Single comment test passed")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def test_batch_comments():
    response = client.post("/check-batch", json={
        "comments": [
            "This product is excellent",
            "Teh product is really good",
            "I likes this product very much",
            "Great quality good price",
            "I recieved the package quickly"
        ],
        "threshold": 0.5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_comments"] == 5
    print(f"✓ Batch test passed - Total errors: {data['total_errors']}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def test_no_errors():
    response = client.post("/check", json={
        "comment": "This product is absolutely excellent",
        "threshold": 0.5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["analysis"]["num_errors"] == 0
    print("✓ No-error test passed")


def test_multiple_errors():
    response = client.post("/check", json={
        "comment": "Teh product is excelent and I likes it very much",
        "threshold": 0.5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["analysis"]["num_errors"] >= 2
    print(f"✓ Multiple errors test passed - Found {data['analysis']['num_errors']} errors")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("API SERVICE TESTS")
    print("="*80 + "\n")
    
    test_health_check()
    print()
    
    print("\n--- Test 1: Single Comment with Error ---")
    test_single_comment_with_error()
    
    print("\n--- Test 2: No Errors ---")
    test_no_errors()
    
    print("\n--- Test 3: Multiple Errors ---")
    test_multiple_errors()
    
    print("\n--- Test 4: Batch Processing ---")
    test_batch_comments()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80 + "\n")
