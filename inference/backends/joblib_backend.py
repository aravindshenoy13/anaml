import joblib
from inference.base import BaseEngine
from typing import AsyncGenerator

class JoblibModel(BaseEngine):
    def __init__(self):
        self.model = None

    def load(self, weight_path: str):
        self.model_path = weight_path
        self.model = joblib.load(self.model_path)
    
    def predict(self, input_data: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded, call load() first")

        if "features" in input_data:
            modified_input_data = [input_data["features"]]
        elif "text" in input_data:
            modified_input_data = [input_data["text"]]
        elif "instances" in input_data:
            modified_input_data = input_data["instances"]
        else:
            raise ValueError("Invalid input data format")
        
        try:
            #Prediction
            result = self.model.predict(modified_input_data)

            #Confidence Score
            confidence = None
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(modified_input_data)
                confidence = probs.max(axis=1).tolist()
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(modified_input_data)
                confidence = scores.tolist()
        except Exception as e:
            raise ValueError(f"Inference failed: {e}")
        
        #Return as a dictionary
        return {"predictions": result.tolist(), "confidence": confidence}
    
    async def stream(self, input_data: dict) -> AsyncGenerator[dict, None]:
        result = self.predict(input_data)
        yield result
    

