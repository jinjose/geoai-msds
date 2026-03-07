import os, json, tempfile
import pandas as pd

def test_missing_feature_columns_detectable():
    # This unit test validates the feature-order concept used in batch_runner.
    feature_order = {"feature_order": ["a","b"]}
    df = pd.DataFrame({"a":[1,2]})
    missing = [c for c in feature_order["feature_order"] if c not in df.columns]
    assert missing == ["b"]
