import joblib
import os
import hashlib
import json
import subprocess
import sys


class FeatureStore:
    """
    Centralized store for user and item embeddings.
    Used by HybridRecommender to cache and retrieve embeddings
    instead of recomputing them on every request.
    """

    def __init__(self, store_path="src/model/data"):
        self.store_path = store_path
        self._manifest_path = os.path.join(store_path, "manifest.json")
        os.makedirs(store_path, exist_ok=True)
        self._user_embeddings = {}
        self._item_embeddings = {}

    def save_user_embedding(self, user_id, embedding):
        self._user_embeddings[user_id] = embedding
        self._persist("user_embeddings.joblib", self._user_embeddings)

    def get_user_embedding(self, user_id):
        self._load("user_embeddings.joblib", self._user_embeddings)
        return self._user_embeddings.get(user_id, None)

    def save_item_embedding(self, item_id, embedding):
        self._item_embeddings[item_id] = embedding
        self._persist("item_embeddings.joblib", self._item_embeddings)

    def get_item_embedding(self, item_id):
        self._load("item_embeddings.joblib", self._item_embeddings)
        return self._item_embeddings.get(item_id, None)

    def _compute_hash(self, path):
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_hash(self, path):
        """Save file hash to manifest after writing."""
        manifest = {}
        if os.path.exists(self._manifest_path):
            with open(self._manifest_path, "r") as f:
                manifest = json.load(f)
        manifest[path] = self._compute_hash(path)
        with open(self._manifest_path, "w") as f:
            json.dump(manifest, f)

    def _verify_hash(self, path):
        """Verify file hash against manifest before loading."""
        if not os.path.exists(self._manifest_path):
            raise RuntimeError(
                f"Manifest not found. File '{path}' cannot be verified."
            )
        with open(self._manifest_path, "r") as f:
            manifest = json.load(f)
        if path not in manifest:
            raise RuntimeError(
                f"No hash found for '{path}' in manifest."
            )
        actual = self._compute_hash(path)
        if actual != manifest[path]:
            raise RuntimeError(
                f"Hash mismatch for '{path}'. "
                f"File may have been tampered with."
            )

    def _validate_magic_bytes(self, path):
        """Validate joblib magic bytes before deserializing."""
        with open(path, "rb") as f:
            header = f.read(2)
        if header[:1] not in (b'\x78', b'\x1f'):
            raise RuntimeError(
                f"Invalid file format for '{path}'. Expected joblib file."
            )

    def _persist(self, filename, data):
        path = os.path.join(self.store_path, filename)
        joblib.dump(data, path)
        self._save_hash(path)

    def _load(self, filename, target):
        path = os.path.join(self.store_path, filename)
        if os.path.exists(path) and not target:
            self._verify_hash(path)
            self._validate_magic_bytes(path)
            result = subprocess.run(
                [sys.executable, "-c",
                 f"import joblib, json; data = joblib.load('{path}'); "
                 f"print(json.dumps({{k: v.tolist() "
                 f"if hasattr(v, 'tolist') else v "
                 f"for k, v in data.items()}}))"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to load '{path}' safely: {result.stderr}"
                )
            target.update(joblib.load(path))
