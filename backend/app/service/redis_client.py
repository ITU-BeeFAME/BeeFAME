import os
import redis

def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def get_redis_client() -> redis.Redis:
    if os.getenv("REDIS_HOST"):
        host = os.getenv("REDIS_HOST")
    else:
        host = "redis" if os.path.exists("/.dockerenv") else "127.0.0.1"
    port = _get_env_int("REDIS_PORT", 6379)
    db = _get_env_int("REDIS_DB", 0)
    decode_responses = _get_env_bool("REDIS_DECODE_RESPONSES", True)
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=decode_responses,
    )

def cache_set(client: redis.Redis, key: str, value: str) -> None:
    """Store a cache entry. CACHE_TTL_SECONDS > 0 sets an expiry; unset or 0
    keeps the entry permanently, which precomputed results rely on."""
    ttl = _get_env_int("CACHE_TTL_SECONDS", 0)
    if ttl > 0:
        client.setex(key, ttl, value)
    else:
        client.set(key, value)
