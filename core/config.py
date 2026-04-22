import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://mlserve:mlserve@localhost:5432/mlserve")
MODEL_DIR = os.getenv("MODEL_DIR", "temp_dir")
os.makedirs(MODEL_DIR, exist_ok=True)
ALLOWED_BACKENDS = os.getenv("ALLOWED_BACKENDS", "onnx,joblib").split(",")
