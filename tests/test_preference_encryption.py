"""
Tests for user preference data encryption.
Validates encryption, decryption, and GDPR compliance.
"""

import pytest
import json
from src.data.preference_encryption import PreferenceEncryption


def test_encrypt_decrypt_roundtrip():
    """Verify encrypted data can be decrypted correctly."""
    encryption = PreferenceEncryption(master_secret='test-secret-key-12345')

    user_id = 'user123'
    preferences = {
        'genres': ['action', 'drama'],
        'languages': ['en', 'es'],
        'rating_preference': 'high',
    }

    encrypted = encryption.encrypt_preferences(user_id, preferences)
    assert isinstance(encrypted, str)
    assert encrypted != json.dumps(preferences), "Encrypted data should not equal plaintext"

    decrypted = encryption.decrypt_preferences(user_id, encrypted)
    assert decrypted == preferences, "Decrypted data should match original"


def test_different_users_different_encryption():
    """Verify different users produce different ciphertexts for same preferences."""
    encryption = PreferenceEncryption(master_secret='test-secret-key-12345')

    preferences = {'genre': 'action', 'language': 'en'}

    encrypted_user1 = encryption.encrypt_preferences('user1', preferences)
    encrypted_user2 = encryption.encrypt_preferences('user2', preferences)

    assert encrypted_user1 != encrypted_user2, "Different users should have different encryption"

    assert encryption.decrypt_preferences('user1', encrypted_user1) == preferences
    assert encryption.decrypt_preferences('user2', encrypted_user2) == preferences


def test_encryption_disabled_without_secret():
    """Verify plaintext mode when secret not configured."""
    encryption = PreferenceEncryption(master_secret='')

    assert not encryption.is_encrypted_enabled()
    preferences = {'genre': 'action'}

    encrypted = encryption.encrypt_preferences('user123', preferences)
    assert json.loads(encrypted) == preferences, "Should return plaintext when encryption disabled"


def test_complex_preference_structure():
    """Verify encryption works with nested preference structures."""
    encryption = PreferenceEncryption(master_secret='test-secret-key-12345')

    preferences = {
        'genres': {
            'preferred': ['action', 'drama'],
            'disliked': ['horror'],
            'weight': 0.8,
        },
        'demographics': {
            'age_group': '25-34',
            'region': 'US',
        },
        'history': ['item1', 'item2', 'item3'],
    }

    encrypted = encryption.encrypt_preferences('user123', preferences)
    decrypted = encryption.decrypt_preferences('user123', encrypted)

    assert decrypted == preferences


def test_encryption_secret_validation():
    """Verify encryption fails gracefully without proper secret."""
    encryption = PreferenceEncryption(master_secret=None)

    with pytest.raises(ValueError, match='Encryption not configured'):
        encryption._derive_key('user123')


def test_tenant_isolation():
    """Verify preferences cannot be decrypted with wrong user ID."""
    encryption = PreferenceEncryption(master_secret='test-secret-key-12345')

    preferences = {'sensitive': 'data'}
    encrypted = encryption.encrypt_preferences('user1', preferences)

    with pytest.raises(Exception):
        encryption.decrypt_preferences('user2', encrypted)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
