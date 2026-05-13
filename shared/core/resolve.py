from fastapi import HTTPException
from sqlalchemy import select

from core.redis import model_cache
from inference.registry import get_model_class
from models.models import MLModel


async def resolve_model(model_id, session, redis_client, load=True):
    model = None
    model_name = None
    model_version = None
    backend_type = None
    weights_path = None

    if model_id in model_cache:
        return model_cache[model_id]
    
    else:
        cached_data = await redis_client.hgetall(model_id)
        if cached_data:
            model_name = cached_data["model_name"]
            model_version = cached_data["model_version"]
            backend_type = cached_data["backend_type"]
            weights_path = cached_data["weights_path"]
            
        
        else:
            query = select(MLModel).where(MLModel.id == model_id)
            result = await session.execute(query)
            model_db = result.scalar_one_or_none()

            if model_db is None:
                raise HTTPException(status_code=404, detail=f"Model with id {model_id} does not exist!")

            model_name = model_db.name
            model_version = model_db.version
            backend_type = model_db.backend_type
            weights_path = model_db.weights_path
            
            await redis_client.hset(model_id, mapping={
                "model_name": model_name,
                "model_version": model_version,
                "backend_type": backend_type,
                "weights_path": weights_path
            })
            await redis_client.expire(model_id, 3600)

    ret_data = {
            "model": model,
            "model_name": model_name,
            "model_version": model_version,
            "backend_type": backend_type,
            "weights_path": weights_path
        }
    if load:
        backend = get_model_class(backend_type)
        model = backend()
        model.load(weights_path)

        ret_data["model"] = model
        model_cache[model_id] = ret_data
    return ret_data