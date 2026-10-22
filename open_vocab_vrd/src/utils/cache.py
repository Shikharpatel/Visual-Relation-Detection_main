"""Hash-keyed disk cache using pickle.

Usage:
    cache = DiskCache("./cache/clip_embeddings")
    if cache.exists(key):
        emb = cache.load(key)
    else:
        emb = compute(...)
        cache.save(key, emb)
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.root = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        # Two-level sharding to avoid huge flat directories
        return self.root / digest[:2] / (digest + ".pkl")

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        return self._key_to_path(key).exists()

    def save(self, key: str, data: Any) -> None:
        if not self.enabled:
            return
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, key: str) -> Any:
        path = self._key_to_path(key)
        if not path.exists():
            raise KeyError(f"Cache miss: {key}")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception):
            # Corrupted file — remove it so next call recomputes
            try:
                path.unlink()
            except OSError:
                pass
            raise KeyError(f"Cache corrupted (deleted): {key}")

    def invalidate(self, key: str) -> None:
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Remove all cached files under this cache root."""
        import shutil
        if self.root.exists():
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
