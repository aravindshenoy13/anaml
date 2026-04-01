from typing import AsyncGenerator

import onnxruntime as ort
from base import BaseEngine

class OnnxModel(BaseEngine):
    def __init__(self):
        self.model_path = None
        self.model = None
    
    def load(self, weight_path: str):
        self.model_path = weight_path
        self.model = ort.InferenceSession(self.model_path)
    
    def predict(self, input_data: dict) -> dict:
        pass

    async def stream(self, input_data: dict) -> AsyncGenerator[dict, None]:
        pass