import pandas as pd
import numpy as np

def normalize_county(x):
    if pd.isna(x):
        return x
    x = str(x).lower().replace("county", "").replace("-", " ")
    return " ".join(x.split())

def cutoff_mask(dates, month, day):
    return (dates.dt.month < month) | (
        (dates.dt.month == month) & (dates.dt.day <= day)
    )

def trapezoid_auc(dates, values):
    if len(values) < 2:
        return np.nan
    x = dates.map(pd.Timestamp.toordinal).to_numpy()
    y = values.to_numpy()
    return float(np.trapz(y, x))
