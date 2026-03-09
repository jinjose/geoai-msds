import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# ==========================================
# 1. PAGE SETUP (Explicitly 2025)
# ==========================================
st.set_page_config(page_title="2025 GEOAI Yield Intelligence Hub", layout="wide")

# ==========================================
# 2. DEFINITIONS & PATHS
# ==========================================
BASE_PATH = "Feb22 Final Experiments/inference-dataset/intermediate"
FEATURE_DIR = "Feb22 Final Experiments/inference-dataset/features_frozen"
MODEL_DIR = "Feb22 Final Experiments/inference-demo"

# Map Models to their specific 2025 data cutoffs
MODEL_CONFIG = {
    "Early Stage (Jun 01)": {"file": "EFeb22 Final Experiments/exported_models/jun01/LightGBM-limited_withstorm/model.pkl", "cutoff": "jun01"},
    "Reproductive Stage (Jul 01)": {"file": "Feb22 Final Experiments/exported_models/jul01/LightGBM-limited_withstorm/model.pkl", "cutoff": "jul01"},
    "Pre-Harvest Stage (Aug 01)": {"file": "Feb22 Final Experiments/exported_models/aug01/LightGBM-limited_withstorm/model.pkl", "cutoff": "aug01"}
}


# ==========================================
# 3. CACHED DATA & MODELS
# ==========================================
@st.cache_resource
def load_models():
    loaded = {}
    for name, cfg in MODEL_CONFIG.items():
        path = os.path.join(MODEL_DIR, cfg["file"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
    return loaded


@st.cache_data
def load_temporal_data(cutoff):
    """Loads specific intermediate files for a given 2025 date cutoff."""
    ndvi = pd.read_csv(os.path.join(BASE_PATH, f"ndvi_until_{cutoff}.csv"))
    wx = pd.read_csv(os.path.join(BASE_PATH, f"weather_until_{cutoff}.csv"))
    storm = pd.read_csv(os.path.join(BASE_PATH, f"storm_until_{cutoff}.csv"))
    h_yield = pd.read_csv(os.path.join(BASE_PATH, "yield_history.csv"))
    augset = pd.read_csv(os.path.join(FEATURE_DIR, f"features_{cutoff}.csv"))

    # Convert dates for plotting
    ndvi['date'] = pd.to_datetime(ndvi['date'])
    wx['date'] = pd.to_datetime(wx['date'])
    t_col = 'datetime' if 'datetime' in storm.columns else 'time'
    storm[t_col] = pd.to_datetime(storm[t_col])

    return ndvi, wx, storm, h_yield, augset


# Execute Loading
models_dict = load_models()

# ==========================================
# 4. SIDEBAR - 2025 FORECAST CONTROLS
# ==========================================
st.sidebar.title("2025 GEO AI Annual Yield Prediction ")
st.sidebar.info("📅 **Current Forecast Year: 2025**")

# Stage Selection (Drives the Cutoff)
selected_stage = st.sidebar.radio(
    "Select Forecasting Point:",
    options=list(MODEL_CONFIG.keys()),
    help="Selects the model and filters ALL data to the corresponding 2025 cutoff."
)
current_cutoff = MODEL_CONFIG[selected_stage]["cutoff"]

# Load Data based on Cutoff
ndvi_all, wx_all, storm_all, h_yield_all, augset_all = load_temporal_data(current_cutoff)

# County Selection
all_counties = sorted(ndvi_all['county'].unique())
default_county = 'benton' if 'benton' in all_counties else all_counties[0]
COUNTY = st.sidebar.selectbox("Select County", all_counties, index=all_counties.index(default_county))

# Filters for the specific county
ndvi_c = ndvi_all[ndvi_all["county"] == COUNTY]
wx_c = wx_all[wx_all["county"] == COUNTY]
storm_c = storm_all[storm_all["county"] == COUNTY]
yield_c = h_yield_all[h_yield_all["county"] == COUNTY]
augset_c = augset_all[augset_all["county"].astype(str).str.lower() == COUNTY.lower()]

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 2025 Visual Analysis",
    "🧠 Model Inference",
    "📋 Input Features",
    "📁 Data Source Explorer"
])

# --- TAB 1: VISUAL ANALYSIS (Synced to 2025 Cutoff) ---
with tab1:
    st.title(f"2025 Stressors: {COUNTY.upper()} (through {current_cutoff.upper()})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ndvi_c['date'], y=ndvi_c['NDVI'], name="NDVI",
                             line=dict(color='forestgreen', width=4)))
    fig.add_trace(go.Scatter(x=wx_c['date'], y=wx_c['temperature'], name="Temp (°C)",
                             line=dict(color='orange', dash='dot'), yaxis="y2"))

    # Storm Logic (>= 58 mph)
    if 'wind_mph' in storm_c.columns:
        severe = storm_c[storm_c['wind_mph'] >= 58]
        t_col = 'datetime' if 'datetime' in severe.columns else 'time'
        if not severe.empty:
            fig.add_trace(go.Scatter(x=severe[t_col], y=[ndvi_c['NDVI'].max()] * len(severe),
                                     mode='markers', name='Severe Wind (≥58mph)',
                                     marker=dict(color='red', size=12, symbol='x')))

    fig.update_layout(
        hovermode="x unified",
        yaxis=dict(title=dict(text="NDVI Index", font=dict(color="forestgreen")), tickfont=dict(color="forestgreen")),
        yaxis2=dict(title=dict(text="Temp (°C)", font=dict(color="orange")), tickfont=dict(color="orange"),
                    overlaying="y", side="right"),
        xaxis=dict(title=dict(text=f"2025 Timeline (Cutoff: {current_cutoff.upper()})"))
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: INFERENCE ENGINE ---
with tab2:
    st.title(f"2025 Inference Engine: {selected_stage}")

    if not augset_c.empty:
        FEATURES = ["county", "rolling_3yr_mean", "ndvi_peak", "ndvi_slope",
                    "temp_anomaly", "net_moisture_stress", "heat_days_gt32", "wind_severe_days_58_cutoff"]

        X = augset_c[FEATURES].copy()
        X["county"] = X["county"].astype("category")

        # Fill numeric NaNs
        num_cols = X.select_dtypes(include=['number']).columns
        X[num_cols] = X[num_cols].fillna(0)

        # Run Prediction
        model = models_dict[selected_stage]
        pred = model.booster_.predict(X)[0] if hasattr(model, "booster_") else model.predict(X)[0]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted 2025 Yield", f"{pred:.2f} bu/ac")
        with c2:
            last_actual = yield_c.sort_values('year').iloc[-1]['yield_bu_acre']
            st.metric("vs 2024 Actual", f"{pred - last_actual:.2f} bu/ac", delta=f"{pred - last_actual:.2f}")
    else:
        st.error(f"Incomplete feature data for {COUNTY} at {current_cutoff}.")

# --- TAB 3: MODEL FEATURES ---
with tab3:
    st.subheader(f"2025 Engineered Features (Cutoff: {current_cutoff})")
    st.dataframe(augset_c[FEATURES], use_container_width=True)

# --- TAB 4: DATA SOURCE EXPLORER (Synced to Cutoff) ---
with tab4:
    st.header(f"2025 Raw Data Explorer (Cutoff: {current_cutoff})")
    choice = st.radio("Inspect Dataset:", ["NDVI", "Weather", "Storm", "History"], horizontal=True)

    if choice == "NDVI":
        st.dataframe(ndvi_c, use_container_width=True)
    elif choice == "Weather":
        st.dataframe(wx_c, use_container_width=True)
    elif choice == "Storm":
        st.dataframe(storm_c, use_container_width=True)
    elif choice == "History":
        st.dataframe(yield_c.sort_values('year', ascending=False), use_container_width=True)
