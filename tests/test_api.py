import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200

def test_single_prediction():
    """Test single prediction."""
    payload = {
        "day_of_week": 2,
        "day_of_month": 15,
        "month": 8,
        "is_weekend": 0,
        "is_month_start": 0,
        "is_month_end": 0,
        "is_metro": 1,
        "pin_rto_rate": 0.25,
        "state_rto_rate": 0.35,
        "pin_order_count": 50,
        "state_order_count": 200,
        "quantity": 1,
        "meesho_price": 1200,
        "final_price": 1200,
        "shipping_charges_total": 50,
        "price_per_unit": 1200,
        "discount_amount": 0,
        "discount_pct": 0,
        "shipping_to_price_ratio": 0.042,
        "has_valid_pincode": 1,
        "has_state": 1,
        "address_quality_score": 1.0,
        "product_rto_rate": 0.3,
        "product_order_count": 10,
        "product_length": 50,
        "price_category": "medium",
        "state_clean": "Maharashtra"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("\nSingle Prediction:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200
    
    result = response.json()
    assert 0 <= result['rto_probability'] <= 1
    assert result['rto_prediction'] in [0, 1]
    assert result['risk_level'] in ['Low', 'Medium', 'High']

if __name__ == "__main__":
    print("="*60)
    print("API TESTING")
    print("="*60)
    
    test_health_check()
    test_single_prediction()
    
    print("\n✓ All tests passed!")