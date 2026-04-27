import redis

from core.config import REDIS_URL

redis_client = redis.asyncio.from_url(REDIS_URL, decode_responses=True)
model_cache = {}

async def close_redis():
    await redis_client.close()