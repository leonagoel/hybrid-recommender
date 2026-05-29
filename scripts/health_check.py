import time

import requests


def check_backend_health() -> None:
    """Check whether the backend API is running."""
    url = "http://localhost:8000/api/status"

    try:
        start_time = time.time()

        response = requests.get(url, timeout=5)

        end_time = time.time()

        response_time = round((end_time - start_time) * 1000, 2)

        if response.status_code == 200:
            print("✅ Backend is running")
            print(f"⏱ Response time: {response_time} ms")
            print(f"📦 Response: {response.json()}")
        else:
            print(
                f"❌ Backend returned status "
                f"code {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")

    except requests.exceptions.Timeout:
        print("❌ Request timed out")


if __name__ == "__main__":
    check_backend_health()