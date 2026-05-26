import re

import pytest

from scripts.seed_mock_data import generate_mock_password


def test_generate_mock_password_uses_strong_default_shape():
    password = generate_mock_password()

    assert len(password) == 24
    assert re.search(r"[a-z]", password)
    assert re.search(r"[A-Z]", password)
    assert re.search(r"\d", password)
    assert re.search(r"[!@#$%^&*()\-_=+]", password)
    assert not password.startswith("MockUser")


def test_generate_mock_password_rejects_short_lengths():
    with pytest.raises(ValueError):
        generate_mock_password(12)


def test_generate_mock_password_is_not_deterministic():
    passwords = {generate_mock_password() for _ in range(10)}

    assert len(passwords) == 10
