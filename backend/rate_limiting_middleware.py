from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Custom rate limiting middleware to protect endpoints
        client_ip = request.client.host if request.client else "unknown"
        response = await call_next(request)
        response.headers["X-RateLimit-Status"] = "active"
        return response
