import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import joblib

from abc import ABC, abstractmethod


class BaseEngine(ABC):
    @abstractmethod
    def load(self, weight_path: str):
        pass
    
    @abstractmethod
    def predict(self, input_data: dict) -> dict:
        pass

    @abstractmethod
    async def stream(self, input_data: dict):
        pass

    @abstractmethod
    def metadata(self):
        pass


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

        labels = getattr(self.model, "classes_", None)
        if labels is not None:
            labels = labels.tolist()
        
        return {
            "inputs": "features",
            "input_shapes": [None, input_shape],
            "input_types": "float64",
            "labels": labels
        }
    

dset = load_iris()
model = RandomForestClassifier()
model.fit(dset['data'], dset['target'])
joblib.dump(model, "test_model.pkl")
print("Training done and saved")


# dset = load_iris()
# model = RandomForestClassifier()
# model.fit(dset['data'], dset['target'])
# model = skl2onnx.convert_sklearn(model,initial_types=[("X",FloatTensorType([None, 4]))])
# model_bytes = model.SerializeToString()
model = JoblibModel()
model.load("test_model.pkl")
print(model.metadata())