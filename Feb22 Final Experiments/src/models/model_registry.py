from .lag1_baseline_mlflow import run_lag1_baseline
from .ridge_mlflow import run_ridge
from .lightbgm_withlimited_withstorm import run_lightgbm_limited_features_storm

# ============================================================
# MODEL EXPERIMENT REGISTRY
# Each model below answers a specific statistical question.
# This is not just model training — it is controlled experimentation.
# ============================================================

MODEL_CONFIGS = [

    # ------------------------------------------------------------
    # 1. Lag-1 Baseline (Pure Persistence Model)
    # ------------------------------------------------------------
    # Purpose:
    #   Establish the minimal benchmark.
    #   Tests how well last year's yield alone predicts current yield.
    #
    # Hypothesis:
    #   Yield persistence explains a large portion of structural variance.
    #
    # Interpretation:
    #   If complex models barely beat this, then seasonal signals
    #   are adding limited incremental value.
    # ------------------------------------------------------------
    {
        "name": "Lag-1 Baseline",
        "func": run_lag1_baseline,
        "enabled": True,
        "metric_cols": ["mae", "rmse", "mape", "r2"],
    },

    # # ------------------------------------------------------------


    # ------------------------------------------------------------
    # 3. Ridge Regression (Linear Benchmark)
    # ------------------------------------------------------------
    # Purpose:
    #   Test linear relationships only.
    #
    # Hypothesis:
    #   If Ridge performs similarly to LightGBM,
    #   then nonlinear tree complexity may not be necessary.
    #
    # Interpretation:
    #   Serves as bias-controlled linear baseline.
    # ------------------------------------------------------------
    {
        "name": "Ridge",
        "func": lambda df, cutoff: run_ridge(df, cutoff_key=cutoff, alpha=1.0),
        "enabled": True,
        "metric_cols": ["mae", "rmse", "mape", "r2"],
    },


    # # ------------------------------------------------------------
    # # 6. LightGBM Limited Features (Orthogonal Signal Model)
    # # ------------------------------------------------------------
    # # Purpose:
    # #   Remove redundant correlated signals.
    # #   Keep only statistically orthogonal drivers.
    # #
    # # Features:
    # #   - rolling_3yr_mean (structural memory)
    # #   - ndvi_peak (biomass proxy)
    # #   - ndvi_slope (growth velocity)
    # #   - temp_anomaly (heat stress)
    # #   - net_moisture_stress (water imbalance)
    # #   - heat_days_gt32
    # #   - wind_severe_days_58_cutoff
    # #
    # # Hypothesis:
    # #   Cleaner signals → better generalization and lower overfitting.
    # #
    # # Interpretation:
    # #   Tests whether complexity reduction improves robustness.
    # # ------------------------------------------------------------
{
        "name": "LightGBM-limited_withstorm",
        "func": run_lightgbm_limited_features_storm,
        "enabled": True,
        "metric_cols": ["mae", "rmse", "mape", "r2"],
    },
]
