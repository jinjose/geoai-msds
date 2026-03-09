import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os

# ==========================================
# PAGE SETUP
# ==========================================
# ==========================================
# PATHS
# ==========================================
BASE_PATH = "Feb22 Final Experiments/inference-dataset/intermediate"
FEATURE_DIR = "Feb22 Final Experiments/inference-dataset/features_frozen"

MODEL_CONFIG = {
    "Planting-Vegetative Stage (Jun 01)": {
        "file": "Feb22 Final Experiments/exported_models/jun01/LightGBM-limited_withstorm/model.pkl",
        "cutoff": "jun01"
    },
    "Reproductive Stage (Jul 01)": {
        "file": "Feb22 Final Experiments/exported_models/jul01/LightGBM-limited_withstorm/model.pkl",
        "cutoff": "jul01"
    },
    "Pre-Harvest Stage (Aug 01)": {
        "file": "Feb22 Final Experiments/exported_models/aug01/LightGBM-limited_withstorm/model.pkl",
        "cutoff": "aug01"
    }
}

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    for name, cfg in MODEL_CONFIG.items():
        path = cfg["file"]
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_temporal_data(cutoff):

    ndvi = pd.read_csv(os.path.join(BASE_PATH, f"ndvi_until_{cutoff}.csv"))
    wx = pd.read_csv(os.path.join(BASE_PATH, f"weather_until_{cutoff}.csv"))
    storm = pd.read_csv(os.path.join(BASE_PATH, f"storm_until_{cutoff}.csv"))
    h_yield = pd.read_csv(os.path.join(BASE_PATH, "yield_history.csv"))
    augset = pd.read_csv(os.path.join(FEATURE_DIR, f"features_{cutoff}.csv"))

    ndvi["date"] = pd.to_datetime(ndvi["date"])
    wx["date"] = pd.to_datetime(wx["date"])

    t_col = "datetime" if "datetime" in storm.columns else "time"
    storm[t_col] = pd.to_datetime(storm[t_col])

    return ndvi, wx, storm, h_yield, augset


models_dict = load_models()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("2025 GEO AI Annual Yield Prediction")
st.sidebar.info("Forecast Year: 2025")

selected_stage = st.sidebar.radio(
    "Select Forecast Stage",
    list(MODEL_CONFIG.keys())
)

current_cutoff = MODEL_CONFIG[selected_stage]["cutoff"]

ndvi_all, wx_all, storm_all, h_yield_all, augset_all = load_temporal_data(current_cutoff)

all_counties = sorted(ndvi_all["county"].unique())
default_county = "benton" if "benton" in all_counties else all_counties[0]

COUNTY = st.sidebar.selectbox(
    "Select County",
    all_counties,
    index=all_counties.index(default_county)
)

# ==========================================
# FILTER COUNTY DATA
# ==========================================
ndvi_c = ndvi_all[ndvi_all["county"] == COUNTY]
wx_c = wx_all[wx_all["county"] == COUNTY]
storm_c = storm_all[storm_all["county"] == COUNTY]
yield_c = h_yield_all[h_yield_all["county"] == COUNTY]
augset_c = augset_all[augset_all["county"].astype(str).str.lower() == COUNTY.lower()]

# ==========================================
# PAGE TITLE
# ==========================================
st.title("2025 GEOAI Yield Intelligence Hub")

# ==========================================
# ROW 1 : MODEL INFERENCE
# ==========================================
st.subheader(f"Yield Forecast — {selected_stage}")

model = models_dict.get(selected_stage)

if model is None:
    st.error("Model not found. Check deployment paths.")
    st.stop()

if not augset_c.empty:

    FEATURES = [
        "county",
        "rolling_3yr_mean",
        "ndvi_peak",
        "ndvi_slope",
        "temp_anomaly",
        "net_moisture_stress",
        "heat_days_gt32",
        "wind_severe_days_58_cutoff"
    ]

    X = augset_c[FEATURES].copy()
    X["county"] = X["county"].astype("category")

    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(0)

    if hasattr(model, "booster_"):
        pred = model.booster_.predict(X)[0]
    else:
        pred = model.predict(X)[0]

    if not yield_c.empty:
        last_actual = yield_c.sort_values("year").iloc[-1]["yield_bu_acre"]
        delta = pred - last_actual
    else:
        last_actual = None
        delta = None

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Predicted 2025 Yield",
            f"{pred:.2f} bu/ac"
        )

    with c2:
        if delta is not None:
            st.metric(
                "vs 2024 Actual",
                f"{last_actual:.2f} bu/ac",
                delta=f"{delta:+.2f} bu/ac"
            )

    with c3:
        st.metric(
            "Forecast Stage",
            selected_stage
        )

else:
    st.error("Incomplete feature data for this county.")

# ==========================================
# ROW 2 : STRESSORS
# ==========================================
st.subheader(f"Crop Stressors — {COUNTY.upper()} (through {current_cutoff.upper()})")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ndvi_c["date"],
    y=ndvi_c["NDVI"],
    name="NDVI",
    line=dict(color="forestgreen", width=4)
))

fig.add_trace(go.Scatter(
    x=wx_c["date"],
    y=wx_c["temperature"],
    name="Temperature (°C)",
    line=dict(color="orange", dash="dot"),
    yaxis="y2"
))

if "wind_mph" in storm_c.columns:

    severe = storm_c[storm_c["wind_mph"] >= 58]

    if not severe.empty:

        t_col = "datetime" if "datetime" in severe.columns else "time"

        fig.add_trace(go.Scatter(
            x=severe[t_col],
            y=[ndvi_c["NDVI"].max()] * len(severe),
            mode="markers",
            name="Severe Wind ≥58mph",
            marker=dict(color="red", size=12, symbol="x")
        ))

fig.update_layout(
    hovermode="x unified",
    yaxis=dict(title="NDVI Index"),
    yaxis2=dict(
        title="Temperature (°C)",
        overlaying="y",
        side="right"
    ),
    xaxis=dict(title=f"2025 Timeline (Cutoff: {current_cutoff.upper()})")
)

st.plotly_chart(fig, width="stretch")

# ==========================================
# ROW 3 : FEATURES
# ==========================================
st.subheader("Model Input Features")

if not augset_c.empty:
    st.dataframe(augset_c[FEATURES], width="stretch")

# ==========================================
# ROW 4 : DATA EXPLORER
# ==========================================
st.subheader("Raw Data Explorer")

dataset_choice = st.radio(
    "Inspect Dataset",
    ["NDVI", "Weather", "Storm", "History"],
    horizontal=True
)

if dataset_choice == "NDVI":
    st.dataframe(ndvi_c, width="stretch")

elif dataset_choice == "Weather":
    st.dataframe(wx_c, width="stretch")

elif dataset_choice == "Storm":
    st.dataframe(storm_c, width="stretch")

elif dataset_choice == "History":
    st.dataframe(
        yield_c.sort_values("year", ascending=False),
        width="stretch"
    )
