import lightgbm as lgb
import pandas as pd
import mlflow

def train_model(train_path):
    df = pd.read_csv(train_path)
    X = df.drop(columns=["yield"])
    y = df["yield"]

    model = lgb.LGBMRegressor()
    model.fit(X, y)

    mlflow.lightgbm.log_model(model, "model")
    return model
