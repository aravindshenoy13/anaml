from backends.pickle_backend import pickle_load_model

def load(model_id, model_type):
    weights_path = None
    if model_type == 'pickle':
        model = pickle_load_model(weights_path)
    elif model_type == 'onyx':
        pass
    else:
        pass

def predict(input):
    
