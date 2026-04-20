from typing import AsyncGenerator

import numpy as np
import onnxruntime as ort

from inference.base import BaseEngine


def parse_inputs(session):
    onnx_to_numpy = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
        "tensor(float16)": np.float16,
    }
    return {
        inp.name: {
            "shape": inp.shape,
            "dtype": onnx_to_numpy[inp.type],
        }
        for inp in session.get_inputs()
    }

class OnnxModel(BaseEngine):
    def __init__(self):
        self.model_path = None
        self.model = None
        
        self.input_format = None

        self.output_format = dict()
    
    def load(self, weight_path: str):
        self.model_path = weight_path
        self.model = ort.InferenceSession(self.model_path)
        self.input_format = parse_inputs(self.model)

    
    def predict(self, input_data: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded, call load() first")
        
        preprocessed_input_data = []
        for inp_name, inp_vals in self.input_format.items():
            #Check if input exists in input data
            if inp_name not in input_data:
                raise ValueError(f"Missing input: {inp_name}")
            arr = np.asarray(input_data, dtype = inp_vals["dtype"])
            
            #Check if dimensions match
            for i, dim in enumerate(inp_vals["shape"]):
                if dim is not None and arr.shape[i]!=dim:
                    raise ValueError(f"{inp_name} dim {i}: expected {dim}, got {arr.shape[i]}")
            
            preprocessed_input_data[inp_name] = arr



    async def stream(self, input_data: dict) -> AsyncGenerator[dict, None]:
        pass