from backends.joblib_backend import JoblibModel
from backends.onnx_backend import OnnxModel

reg = {
    "joblib": JoblibModel,
    "onnx": OnnxModel
}

def get_model_class(class_type: str):
    if class_type in reg:
        return reg[class_type]
    else:
        raise ValueError(f"Model {class_type} not found in registry")