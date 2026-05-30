import time
import os
from threading import Lock
from fastapi import Request, Response
from fastapi.responses import JSONResponse

_rate_limit_buckets: dict = {}
_rate_limit_lock = Lock()

def _get_rate_limit(limit_env: str, default_limit: int) -> int:
    try:
        limit = int(os.environ.get(limit_env, str(default_limit)))
    except ValueError:
        return default_limit
    return max(1, limit)

def _rate_limit_exceeded_response(rate_limit: int, reset_time: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
        },
        headers={
            "x-ratelimit-limit": str(rate_limit),
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(reset_time),
        },
    )

def _apply_rate_limit(
    request: Request,
    response: Response,
    scope: str,
    limit_env: str,
    default_limit: int,
) -> JSONResponse | None:
    rate_limit = _get_rate_limit(limit_env, default_limit)
    client_ip = request.client.host if request.client else "127.0.0.1"
    bucket_key = (scope, client_ip)
    now = time.time()

    with _rate_limit_lock:
        timestamps = _rate_limit_buckets.setdefault(bucket_key, [])
        timestamps[:] = [timestamp for timestamp in timestamps if now - timestamp < 60]

        reset_time = int(60 - (now - timestamps[0])) if timestamps else 60
        reset_time = max(0, reset_time)

        if len(timestamps) >= rate_limit:
            return _rate_limit_exceeded_response(rate_limit, reset_time)

        timestamps.append(now)
        remaining = rate_limit - len(timestamps)
        reset_time = int(60 - (now - timestamps[0])) if timestamps else 60
        reset_time = max(0, reset_time)

    response.headers["x-ratelimit-limit"] = str(rate_limit)
    response.headers["x-ratelimit-remaining"] = str(remaining)
    response.headers["x-ratelimit-reset"] = str(reset_time)
    return None
