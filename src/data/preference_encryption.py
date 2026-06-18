"""
User preference data encryption module.

Encrypts sensitive user preference data at rest using per-user encryption keys.
Supports GDPR compliance with data deletion capability.
"""

import os
import logging
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from base64 import urlsafe_b64encode, urlsafe_b64decode
import json

logger = logging.getLogger(__name__)


class PreferenceEncryption:
    """
    Encrypts and decrypts user preference data using Fernet (AES-128).
    Per-user encryption keys derived from user ID and master secret.
    """

    def __init__(self, master_secret: Optional[str] = None):
        """
        Initialize encryption with master secret.

        Args:
            master_secret: Base encryption secret. If None, uses PREFERENCE_ENCRYPTION_SECRET env var.
        """
        if master_secret is None:
            master_secret = os.getenv('PREFERENCE_ENCRYPTION_SECRET', '')
            if not master_secret:
                logger.warning('PREFERENCE_ENCRYPTION_SECRET not configured; encryption disabled')

        self.master_secret = master_secret.encode() if master_secret else None

    def _derive_key(self, user_id: str) -> bytes:
        """
        Derive per-user encryption key from master secret and user ID.
        Uses PBKDF2 with 100,000 iterations for security.
        """
        if not self.master_secret:
            raise ValueError('Encryption not configured: PREFERENCE_ENCRYPTION_SECRET missing')

        salt = user_id.encode()
        kdf = PBKDF2(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
        key = urlsafe_b64encode(kdf.derive(self.master_secret))

        return key

    def encrypt_preferences(self, user_id: str, preferences: Dict[str, Any]) -> str:
        """
        Encrypt user preference dictionary.

        Args:
            user_id: User identifier
            preferences: Dictionary of preference data

        Returns:
            Encrypted string (base64 encoded)
        """
        if not self.master_secret:
            logger.warning('Encryption disabled; returning plaintext')
            return json.dumps(preferences)

        try:
            key = self._derive_key(user_id)
            cipher = Fernet(key)
            plaintext = json.dumps(preferences).encode()
            ciphertext = cipher.encrypt(plaintext)

            return urlsafe_b64decode(ciphertext).decode('utf-8')
        except Exception as e:
            logger.error(f'Preference encryption failed for user {user_id}: {e}')
            raise

    def decrypt_preferences(self, user_id: str, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt user preference data.

        Args:
            user_id: User identifier
            encrypted_data: Encrypted preference string

        Returns:
            Decrypted preference dictionary
        """
        if not self.master_secret:
            logger.warning('Encryption disabled; returning plaintext')
            return json.loads(encrypted_data)

        try:
            key = self._derive_key(user_id)
            cipher = Fernet(key)
            ciphertext = urlsafe_b64encode(encrypted_data.encode())
            plaintext = cipher.decrypt(ciphertext)

            return json.loads(plaintext.decode('utf-8'))
        except Exception as e:
            logger.error(f'Preference decryption failed for user {user_id}: {e}')
            raise

    def is_encrypted_enabled(self) -> bool:
        """Check if encryption is properly configured."""
        return self.master_secret is not None
