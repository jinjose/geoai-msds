import pandas as pd
from pathlib import Path
import numpy as np
import joblib
import lightgbm as lgb

from geoai.inference.predict import predict_csv

def test_predict_csv_smoke(tmp_path: Path):
    # Train tiny model
    X = pd.DataFrame({"a":[1,2,3,4], "b":[0.1,0.2,0.3,0.4]})
    y = np.array([10,12,13,15])
    train = lgb.Dataset(X, label=y)
    booster = lgb.train({"objective":"regression", "verbose":-1}, train, num_boost_round=5)

    model_dir = tmp_path/"model"
    model_dir.mkdir()
    joblib.dump(booster, model_dir/"model.pkl")

    inp = tmp_path/"input.csv"
    X.assign(county=["x"]*4, year=[2020]*4, cutoff=["jun01"]*4).to_csv(inp, index=False)
    out = tmp_path/"pred.csv"

    res = predict_csv(inp, out, id_columns=["county","year","cutoff"], model_dir=model_dir)
    assert out.exists()
    df = pd.read_csv(out)
    assert "y_pred" in df.columns
    assert len(df) == 4
