from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from db import get_supabase

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    token = credentials.credentials

    sb = get_supabase()

    try:

        user_response = sb.auth.get_user(token)

        if not user_response.user:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

        if user_response.user.is_anonymous:
            raise HTTPException(
                status_code=401,
                detail="Please sign in to access personalized features"
            )

        return user_response.user

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )