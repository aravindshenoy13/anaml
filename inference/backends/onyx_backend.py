import onnxruntime as ort

class OnyxModel:
    def __init__(self, weight_path):
        self.model_path = weight_path
        self.model = None
    
    def load(self):
        self.model = ort.InferenceSession(self.model_path)
        return self.model