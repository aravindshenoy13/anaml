import joblib

from inference.base import BaseEngine


class JoblibModel(BaseEngine):
    def __init__(self):
        self.model = None

    def load(self, weight_path: str):
        self.model_path = weight_path
        self.model = joblib.load(self.model_path)
    
    def predict(self, input_data: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded, call load() first")

        preprocessed_input_data = input_data["features"]
        
        try:
            #Prediction
            result = self.model.predict(preprocessed_input_data)

            #Confidence Score
            confidence = None
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(preprocessed_input_data)
                confidence = probs.max(axis=1).tolist()
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(preprocessed_input_data)
                confidence = scores.tolist()
        except Exception as e:
            raise ValueError(f"Inference failed: {e}")
        
        return {"predictions": result.tolist(), "confidence": confidence}
    
    async def stream(self, input_data: dict):
        result = self.predict(input_data)
        yield result

    def metadata(self):
        if self.model is None:
            raise RuntimeError("Model not loaded, call load() first")

        input_shape = getattr(self.model, "n_features_in_", None)

        outputs = getattr(self.model, "classes_", None)
        if outputs is not None:
            outputs = outputs.tolist()
        
        return {
            "inputs": "features",
            "input_shapes": [None, input_shape],
            "input_types": "float64",
            "labels": outputs
        }
    

