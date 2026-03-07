import os
import lightgbm as lgb
import pandas as pd

def model_fn(model_dir):
    return lgb.Booster(model_file=os.path.join(model_dir, "model.pkl"))

def input_fn(request_body, content_type):
    if content_type == "text/csv":
        return pd.read_csv(request_body)
    raise ValueError("Unsupported content type")

def predict_fn(data, model):
    data["prediction"] = model.predict(data)
    return data

def output_fn(prediction, accept):
    return prediction.to_csv(index=False)
