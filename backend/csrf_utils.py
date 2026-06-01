import secrets
from fastapi import HTTPException
from pydantic import BaseModel

class CSRFTokenResponse(BaseModel):
    csrfToken: str

class CSRFMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)

def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)

def set_csrf_cookie(response, token: str) -> None:
    response.set_cookie("csrf-token", token, httponly=True, secure=True)

async def csrf_header_dep(request) -> None:
    pass  # Placeholder for CSRF validation