import time
import math
import logging
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)

SLOW_RESPONSE_THRESHOLD_MS = 500.0
METRICS_SAMPLE_SIZE = 1000
METRICS_WINDOW_SECONDS = 600

response_time_samples = deque(maxlen=METRICS_SAMPLE_SIZE)
response_metrics = {
    "total_requests": 0,
    "error_requests": 0,
}
response_metrics_lock = Lock()

def _percentile(values, percentile):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]

def record_response_metric(endpoint, method, status_code, response_time_ms):
    with response_metrics_lock:
        response_metrics["total_requests"] += 1
        if status_code >= 400:
            response_metrics["error_requests"] += 1
        response_time_samples.append(
          (time.time(), response_time_ms)
        )

        current_time = time.time()

        while (
          response_time_samples
          and current_time - response_time_samples[0][0] > METRICS_WINDOW_SECONDS
        ):
          response_time_samples.popleft()
    
    log_level = logging.WARNING if response_time_ms > SLOW_RESPONSE_THRESHOLD_MS else logging.INFO
    if log_level == logging.WARNING:
        logger.warning("API request slow endpoint=%s method=%s status=%s time=%.2fms response_time_ms=%.2f endpoint=%s",
                       endpoint, method, status_code, response_time_ms, response_time_ms, endpoint)
    else:
        logger.info("API request endpoint=%s method=%s status=%s time=%.2fms",
                    endpoint, method, status_code, response_time_ms)

def reset_response_metrics():
    with response_metrics_lock:
        response_metrics["total_requests"] = 0
        response_metrics["error_requests"] = 0
        response_time_samples.clear()

def get_response_metrics_snapshot():
    with response_metrics_lock:
        samples = [value for _, value in response_time_samples]
        total_requests = response_metrics["total_requests"]
        error_requests = response_metrics["error_requests"]
    avg_response_time = sum(samples) / len(samples) if samples else 0.0
    error_rate = (error_requests / total_requests) * 100 if total_requests else 0.0
    return {
        "avg_response_time": round(avg_response_time, 2),
        "p95_response_time": round(_percentile(samples, 95), 2),
        "total_requests": total_requests,
        "error_rate": round(error_rate, 2),
    }

async def response_time_middleware_logic(request, call_next):
    start_time = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        response_time_ms = (time.perf_counter() - start_time) * 1000
        if response is not None:
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
        record_response_metric(request.url.path, request.method, status_code, response_time_ms)
