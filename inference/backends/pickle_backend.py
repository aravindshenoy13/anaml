import joblib

class PickleModel:
    def __init__(self, weight_path):
        self.model_path = weight_path
        self.model = self.load()

    def load(self):
        self.model = joblib.load(self.model_path)
        return self.model
    
    def predict(self, data):
        #Expects input data in list(list) form -> [[values], [values], ...]
        return self.model.predict(data)
    
    async def stream(self, data):
        #Expects input data in list form -> [values, values, ...] (streamed one by one)
        for v in data:
            yield self.model.predict([v])
    

