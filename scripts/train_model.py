from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

dset = load_iris()
model = RandomForestClassifier()
model.fit(dset['data'], dset['target'])
joblib.dump(model, "test_model.pkl")
print("Training done and saved")