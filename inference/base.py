from backends.pickle_backend import PickleModel
from backends.onyx_backend import OnyxModel

class InferenceEngine:
    def __init__(self, model_id, model_type):
        self.model_id = model_id
        self.model_type = model_type
        self.model = None

    def load(self):
        weight_path, model_type = database_get(self.model_id)
        if model_type == 'pickle':
            self.model = PickleModel(weight_path)
        elif model_type == 'onyx':
            self.model = OnyxModel(weight_path)
        elif model_type == 'remote':
            pass
    
    def predict(self, input):
        return self.model.predict(input)
    
    async def stream(self, input):
        async for output in self.model.stream(input):
            yield output