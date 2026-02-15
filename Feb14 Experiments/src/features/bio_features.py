import numpy as np
import pandas as pd
from utils import cutoff_mask, trapezoid_auc


def ndvi_features(ndvi_year, month, day, hist_mean_ndvi=None):
    """
    NDVI features using smoothed signal for peak detection.
    """
    ndvi_col = "NDVI_smooth" if "NDVI_smooth" in ndvi_year.columns else "NDVI"
    nd = ndvi_year[cutoff_mask(ndvi_year["date"], month, day)]

    if nd.empty: return {}

    nd = nd.sort_values("date")
    peak_row = nd.loc[nd[ndvi_col].idxmax()]

    feats = {
        "ndvi_at_cutoff": nd.iloc[-1][ndvi_col],
        "ndvi_auc": trapezoid_auc(nd["date"], nd[ndvi_col]),
        "ndvi_peak": peak_row[ndvi_col],
        "day_of_peak_ndvi": peak_row["date"].dayofyear,
    }

    if hist_mean_ndvi is not None:
        feats["ndvi_anomaly"] = feats["ndvi_at_cutoff"] - hist_mean_ndvi

    if len(nd) >= 2:
        x = nd["date"].map(pd.Timestamp.toordinal).to_numpy()
        y = nd[ndvi_col].to_numpy()
        feats["ndvi_slope"] = np.polyfit(x, y, 1)[0]
        feats["ndvi_drop_rate"] = min(0, feats["ndvi_at_cutoff"] - feats["ndvi_peak"])
    else:
        feats["ndvi_slope"] = np.nan
        feats["ndvi_drop_rate"] = 0.0

    return feats


def weather_features(wx_year, month, day, hist_mean_temp=None):
    """
    Weather features updated with VPD and Water Balance to reduce MAE.
    """
    # wd = wx_year[cutoff_mask(wx_year["date"], month, day)]
    # if wd.empty: return {}

    mask = cutoff_mask(wx_year["date"], month, day)
    wd = wx_year[
        mask & (wx_year["date"].dt.month >= 4)
        ]
    if wd.empty: return {}

    # Standard metrics
    gdd = wd["temperature"].apply(lambda x: max(min(x, 30) - 10, 0)).sum()

    feats = {
        "heat_days_gt32": int((wd["temperature"] > 29).sum()),
        "rain_sum_mm": wd["rain_mm"].sum(),
        "rain_days": int((wd["rain_mm"] > 0.1).sum()),
        "cumulative_gdd": gdd,
        "temp_anomaly": wd["temperature"].mean() - hist_mean_temp if hist_mean_temp else 0.0
    }

    # IMPROVISATION 1: Vapor Pressure Deficit (VPD)
    # Measures 'atmospheric thirst' - if air is dry, corn stops growing even with rain.
    if "vpd_kpa" in wd.columns:
        feats["avg_vpd_kpa"] = wd["vpd_kpa"].mean()

    # IMPROVISATION 2: Water Balance (Supply vs. Demand)
    # A negative balance (more ET than rain) is a smoking gun for yield loss.
    if "water_balance_mm" in wd.columns:
        feats["water_balance_total_mm"] = wd["water_balance_mm"].sum()

    # IMPROVISATION 3: ET-to-Rain Ratio
    # Provides a normalized stress index that helps prevent Ridge model spikes.
    if "et_mm" in wd.columns:
        feats["et_rain_ratio"] = wd["et_mm"].sum() / (feats["rain_sum_mm"] + 1e-6)

    # Legacy Heat-Rain Interaction for backward compatibility
    feats["heat_rain_stress_idx"] = feats["heat_days_gt32"] / (feats["rain_sum_mm"] + 1e-6)

    return feats