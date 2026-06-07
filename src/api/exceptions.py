import logging
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

async def global_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Overrides the default FastAPI HTTPException to ensure a unified dictionary response.
    Maintains the 'detail' key for backward compatibility with existing frontends.
    """
    # Normalize the detail into a unified shape.
    # If detail is already a dict or list, move it to 'details'.
    message = exc.detail if isinstance(exc.detail, str) else "An error occurred"
    details = exc.detail if isinstance(exc.detail, (dict, list)) else None
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": message,
            "detail": message,  # Backward compatibility for ui.js
            "details": details
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Overrides default validation error shape."""
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "status_code": 422,
            "message": "Validation Error",
            "detail": "Validation Error",  # Backward compatibility
            "details": exc.errors()
        }
    )

async def global_exception_handler(request: Request, exc: Exception):
    """Catches any unhandled exceptions."""
    logger.error(f"Unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal Server Error",
            "detail": "Internal Server Error",  # Backward compatibility
            "details": str(exc)
        }
    )

def register_exception_handlers(app: FastAPI):
    """Registers all global exception handlers to the FastAPI app."""
    app.add_exception_handler(StarletteHTTPException, global_http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
