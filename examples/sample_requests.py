import requests

BASE_URL = "http://localhost:8000"


def chat_example():
    payload = {
        "user_state": {
            "stage_dropped": "VKYC",
            "error_codes": ["OCR_FAIL"],
            "device_type": "Android",
            "language": "english",
        },
        "query": "Why did my video KYC fail?",
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    print(res.json())


def predict_reason_example():
    payload = {
        "stage_dropped": "OTP",
        "error_codes": ["OTP_TIMEOUT"],
        "device_type": "iOS",
    }
    res = requests.post(f"{BASE_URL}/predict_reason", json=payload, timeout=30)
    print(res.json())


def nudge_example():
    payload = {
        "user_state": {
            "stage_dropped": "eKYC",
            "error_codes": [],
            "device_type": "Android",
            "language": "english",
        },
        "nudge_type": "cta_focused",
        "language": "english",
    }
    res = requests.post(f"{BASE_URL}/nudge_user", json=payload, timeout=30)
    print(res.json())


if __name__ == "__main__":
    chat_example()
    predict_reason_example()
    nudge_example()

