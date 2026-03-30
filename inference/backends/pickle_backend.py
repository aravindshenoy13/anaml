import joblib

def pickle_load_model(file_path):
    return joblib.load(file_path)