from .lag1_baseline_mlflow import run_lag1_baseline_mlflow
from .ridge_mlflow import run_ridge_mlflow
from .lightbgm_withlimited_withstorm import run_lightgbm_limited_features_storm
from .lightbgm_tuned_withoutstorm import run_lightgbm_tuned_withoutstorm

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
        "func": run_lag1_baseline_mlflow,
        "enabled": True,
        "metric_cols": ["mae", "rmse", "mape", "r2"],
    },

    # # ------------------------------------------------------------
    # # 2. LightGBM with Lag (Full Structural + Seasonal Model)
    # # ------------------------------------------------------------
    # # Purpose:
    # #   Combines persistence (lag1_yield) with NDVI + weather signals.
    # #
    # # Hypothesis:
    # #   Tree-based model captures nonlinear seasonal interactions
    # #   beyond pure persistence.
    # #
    # # Interpretation:
    # #   This is the strongest predictive model but may lean heavily
    # #   on lag dominance.
    # # ------------------------------------------------------------
    # {
    #     "name": "LightGBM-lag",
    #     "func": run_lightgbm_mlflow,
    #     "enabled": False,
    #     "metric_cols": ["mae", "rmse", "mape", "r2"],
    # },

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
        "func": lambda df, cutoff: run_ridge_mlflow(
            df,
            cutoff,
            alpha=1.0,
        ),
        "enabled": True,
        "metric_cols": ["mae", "rmse", "mape", "r2"],
    },

    # # ------------------------------------------------------------
    # # 4. LightGBM WITHOUT Lag
    # # ------------------------------------------------------------
    # # Purpose:
    # #   Remove persistence entirely.
    # #
    # # Hypothesis:
    # #   NDVI + weather alone can explain meaningful seasonal variance.
    # #
    # # Interpretation:
    # #   Directly tests whether model is just a disguised lag model.
    # # ------------------------------------------------------------
    # {
    #     "name": "LightGBM-No-Lag",
    #     "func": run_lightgbm_no_lag_mlflow,
    #     "enabled": False,
    #     "metric_cols": ["mae", "rmse", "mape", "r2"],
    # },

    # ------------------------------------------------------------
    # 5. LightGBM WITHOUT Lag + County as Categorical
    # ------------------------------------------------------------
    # Purpose:
    #   Remove lag but allow model to learn structural county baseline.
    #
    # Hypothesis:
    #   Structural spatial differences can replace explicit lag signal.
    #
    # Interpretation:
    #   Cleaner decomposition:
    #     Structural component → county
    #     Seasonal component → NDVI + weather
    # # ------------------------------------------------------------
    # {
    #     "name": "LightGBM-nolag1_withcountyascategory",
    #     "func": run_lightgbm_no_lag_mlflow_county,
    #     "enabled": False,
    #     "metric_cols": ["mae", "rmse", "mape", "r2"],
    # },

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
    # #   - heat_days_gt32 It is actually 29 will be renamed later
    # #
    # # Hypothesis:
    # #   Cleaner signals → better generalization and lower overfitting.
    # #
    # # Interpretation:
    # #   Tests whether complexity reduction improves robustness.
    # # ------------------------------------------------------------
    # {
    #     "name": "LightGBM-limited_withoutstorm",
    #     "func": run_lightgbm_limited_features_mlflow,
    #     "enabled": True,
    #     "metric_cols": ["mae", "rmse", "mape", "r2"],
    # },
{
        "name": "LightGBM-limited_withstorm",
        "func": run_lightgbm_limited_features_storm,
        "enabled": True,
        "metric_cols": ["mae", "rmse", "mape", "r2"],
    },
{
        "name": "LightGBM-tuned_withoutstorm",
        "func": run_lightgbm_tuned_withoutstorm,
        "enabled": True,
        "metric_cols": ["mae", "rmse", "mape", "r2"],
    },
]
