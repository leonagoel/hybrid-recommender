import os
import json
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.warning(f"Redis not available, caching will fall back to in-memory: {e}")
    redis_client = None

class RedisCache:
    @staticmethod
    def get(key: str) -> Optional[Any]:
        if not redis_client:
            return None
        try:
            data = redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    @staticmethod
    def set(key: str, value: Any, ttl: int = 300) -> None:
        if not redis_client:
            return
        try:
            redis_client.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
