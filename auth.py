"""
JWT Authentication module for Hybrid Recommender System.
Handles user authentication, token generation, and validation.
"""
import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Configuration
JWT_SECRET = os.environ.get("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.environ.get("JWT_EXPIRATION_HOURS", "24"))


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt(rounds=10)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False


def create_token(user_id: str, email: str, display_name: Optional[str] = None, expires_in_hours: Optional[int] = None) -> str:
    """
    Create a JWT token for a user.
    
    Args:
        user_id: Unique user identifier
        email: User's email address
        display_name: Optional display name to include in the token
        expires_in_hours: Token expiration time in hours (defaults to JWT_EXPIRATION_HOURS)
    
    Returns:
        Encoded JWT token as string
    """
    if expires_in_hours is None:
        expires_in_hours = JWT_EXPIRATION_HOURS
    
    payload = {
        'user_id': user_id,
        'email': email,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'type': 'access'
    }
    
    if display_name:
        payload['display_name'] = display_name
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


def extract_token_from_header(auth_header: str) -> Optional[str]:
    """
    Extract JWT token from Authorization header.
    Expects format: "Bearer <token>"
    
    Args:
        auth_header: Authorization header value
    
    Returns:
        Token string if valid format, None otherwise
    """
    if not auth_header:
        return None
    
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]
