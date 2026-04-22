from io import BytesIO

import joblib
import pytest
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def joblib_test_data():
    data_dict = {
        "name":"test-model",
        "version":"1.0",
        "backend_type":"joblib",
        "description":"cicd test model for joblib",
        "accuracy":"97.79"
    }
    buffer = BytesIO()
    dset = load_iris()
    model = RandomForestClassifier()
    model.fit(dset['data'], dset['target'])
    joblib.dump(model, buffer)

    file_dict = {"file": 
            ("test.pkl",
            buffer.getvalue(),
            "application/octet-stream"
        )}
    return data_dict, file_dict

def onnx_test_data():
    data_dict = {
        "name":"test-model",
        "version":"1.0",
        "backend_type":"onnx",
        "description":"cicd test model for onnx",
        "accuracy":"97.79"
    }
    dset = load_iris()
    model = RandomForestClassifier()
    model.fit(dset['data'], dset['target'])
    model = skl2onnx.convert_sklearn(model,initial_types=[("X",FloatTensorType([None, 4]))])
    model_bytes = model.SerializeToString()

    file_dict = {"file": 
            ("test.onnx",
            model_bytes,
            "application/octet-stream"
        )}
    
    return data_dict, file_dict

@pytest.mark.asyncio
async def test_health_live(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"

@pytest.mark.asyncio
async def test_joblib_workflow(client):
    #create_model fixture
    data_dict, file_dict = joblib_test_data()
    create_response = await client.post("/models/register", data = data_dict, files=file_dict)
    data = create_response.json()
    assert create_response.status_code == 201, "Model creation failed: " + data["detail"]
    model_id = data["id"]
    assert data["id"], "Model creation id not returned"

    #get_model fixture
    get_response = await client.get(f"/models/{model_id}")
    data = get_response.json()
    assert get_response.status_code == 200, "Model retreival failed: " + data["detail"]
    assert data["id"] == model_id, "Model retreival id not matching/not returned"
    assert data["model_metadata"], "Model metadata not returned: " + str(type(data["model_metadata"]))

    #predict_model fixture
    payload= {
    "input_data":{
    "features": [[5.1, 3.5, 1.4, 0.2]]
    }}
    predict_response = await client.post(f"/models/{model_id}/predict", json=payload)
    assert predict_response.status_code == 200, "Model prediction failed"
    data = predict_response.json()
    assert data["output_data"]['predictions'], "Prediction output values not returned: " + data["detail"]

    #delete_model fixture
    delete_response = await client.delete(f"/models/{model_id}")
    assert delete_response.status_code == 204, "Model deletion failed"
    
    #verifying delete
    verify_response = await client.get(f"/models/{model_id}")
    assert verify_response.status_code == 404, "Model exists after deletion" 


@pytest.mark.asyncio
async def test_onnx_workflow(client):
    #create_model fixture
    data_dict, file_dict = onnx_test_data()
    create_response = await client.post("/models/register", data = data_dict, files=file_dict)
    data = create_response.json()
    assert create_response.status_code == 201, "Model creation failed: " + data["detail"]
    model_id = data["id"]
    assert data["id"], "Model creation id not returned"

    #get_model fixture
    get_response = await client.get(f"/models/{model_id}")
    data = get_response.json()
    assert get_response.status_code == 200, "Model retreival failed: " + data["detail"]
    assert data["id"] == model_id, "Model retreival id not matching/not returned"
    assert data["model_metadata"], "Model metadata not returned"

    #predict_model fixture
    payload= {
    "input_data":{
    "X": [[5.1, 3.5, 1.4, 0.2]]
    }}
    predict_response = await client.post(f"/models/{model_id}/predict", json=payload)
    assert predict_response.status_code == 200, "Model prediction failed"
    data = predict_response.json()
    assert data["output_data"]['predictions'], "Prediction output values not returned: " + data["detail"]

    #delete_model fixture
    delete_response = await client.delete(f"/models/{model_id}")
    assert delete_response.status_code == 204, "Model deletion failed"
    
    #verifying delete
    verify_response = await client.get(f"/models/{model_id}")
    assert verify_response.status_code == 404, "Model exists after deletion"

