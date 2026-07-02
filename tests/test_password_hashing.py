import hashlib

import pytest

from backend.password_hashing import (
    BCRYPT_COST_FACTOR,
    hash_password,
    is_bcrypt_hash,
    password_hash_needs_rehash,
    verify_password,
)


def test_hash_password_uses_bcrypt_cost_factor_12():
    password_hash = hash_password("Correct Horse Battery Staple")

    assert is_bcrypt_hash(password_hash)
    assert password_hash.startswith(f"$2b${BCRYPT_COST_FACTOR}$")
    assert "Correct Horse Battery Staple" not in password_hash


def test_verify_password_accepts_valid_password_and_rejects_invalid_password():
    password_hash = hash_password("s3cure-pa55word")

    assert verify_password("s3cure-pa55word", password_hash) is True
    assert verify_password("wrong-password", password_hash) is False


def test_verify_password_fails_closed_for_md5_and_malformed_hashes():
    legacy_md5_hash = hashlib.md5(b"s3cure-pa55word").hexdigest()

    assert verify_password("s3cure-pa55word", legacy_md5_hash) is False
    assert verify_password("s3cure-pa55word", "not-a-bcrypt-hash") is False
    assert verify_password("", legacy_md5_hash) is False


def test_hash_password_rejects_weak_cost_factors_and_empty_passwords():
    with pytest.raises(ValueError):
        hash_password("s3cure-pa55word", cost_factor=BCRYPT_COST_FACTOR - 1)

    with pytest.raises(ValueError):
        hash_password("")


def test_password_hash_needs_rehash_detects_missing_or_weak_hashes():
    current_hash = hash_password("s3cure-pa55word")
    weak_hash = hash_password(
        "s3cure-pa55word",
        cost_factor=BCRYPT_COST_FACTOR,
    )

    assert password_hash_needs_rehash(current_hash) is False
    assert password_hash_needs_rehash(
        weak_hash,
        cost_factor=BCRYPT_COST_FACTOR + 1,
    ) is True
    assert password_hash_needs_rehash(
        "5f4dcc3b5aa765d61d8327deb882cf99"
    ) is True
