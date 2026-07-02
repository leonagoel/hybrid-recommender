"""
Password hashing helpers for any repository-managed credential flow.

The public app currently delegates user authentication to Supabase Auth. If a
backend/admin flow ever stores local passwords, it must use these helpers
rather than fast hashes such as MD5 or SHA variants.
"""

from __future__ import annotations

import re

import bcrypt

BCRYPT_COST_FACTOR = 12
_BCRYPT_HASH_PATTERN = re.compile(r"^\$2[aby]\$(\d{2})\$[./A-Za-z0-9]{53}$")


def _to_password_bytes(password: str) -> bytes:
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    return password.encode("utf-8")


def hash_password(
    password: str,
    *,
    cost_factor: int = BCRYPT_COST_FACTOR,
) -> str:
    """
    Hash a plaintext password with bcrypt.

    bcrypt cost 12 is the project baseline. Higher values may be passed by
    callers after performance testing, but lower values are rejected.
    """

    if cost_factor < BCRYPT_COST_FACTOR:
        raise ValueError(
            f"bcrypt cost factor must be at least {BCRYPT_COST_FACTOR}."
        )

    password_bytes = _to_password_bytes(password)
    return bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt(rounds=cost_factor),
    ).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.

    Malformed hashes, legacy MD5/SHA digests, and empty inputs fail closed.
    """

    if not isinstance(password_hash, str) or not is_bcrypt_hash(password_hash):
        return False

    try:
        return bcrypt.checkpw(
            _to_password_bytes(password),
            password_hash.encode("ascii"),
        )
    except (TypeError, ValueError):
        return False


def is_bcrypt_hash(password_hash: str) -> bool:
    """Return whether a value has bcrypt's expected modular crypt format."""

    return isinstance(password_hash, str) and bool(
        _BCRYPT_HASH_PATTERN.fullmatch(password_hash)
    )


def password_hash_needs_rehash(
    password_hash: str,
    *,
    cost_factor: int = BCRYPT_COST_FACTOR,
) -> bool:
    """Return true when a bcrypt hash is missing or uses a weaker cost."""

    match = _BCRYPT_HASH_PATTERN.fullmatch(password_hash or "")
    if not match:
        return True

    return int(match.group(1)) < cost_factor
