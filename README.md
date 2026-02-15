# GEOAI corn field prediction

## Project Structure

    project_root/
    │
    ├── data/
    │   ├── raw/              # Raw input data (copied from Google Drive)
    │   └── frozen/           # Feature-engineered datasets (feature_frozen)
    │
    ├── src/
    │   ├── models/           # All model definitions and experiments
    │   ├── build_new_features.py
    │   └── train.py
    │
    └── mlruns/               # MLflow experiment tracking

------------------------------------------------------------------------

## Data Setup

1.  The dataset is stored in **Google Drive**.
2.  Download the data from Google Drive.
3.  Copy raw files into: data/raw/

4.  After running feature engineering, the processed datasets will be
    saved in: data/frozen/

------------------------------------------------------------------------

## Feature Engineering

To generate new feature sets:

``` bash
python src/build_new_features.py
```

This will:

-   Read data from `data/raw/`
-   Generate NDVI and weather features
-   Save feature tables into `data/frozen/`

------------------------------------------------------------------------

## Model Training

To train models:

``` bash
python src/train.py
```

This will:

-   Load features from `data/frozen/`
-   Run walk-forward validation
-   Log metrics and artifacts to MLflow

------------------------------------------------------------------------

## Viewing Results in MLflow

Start MLflow UI:

``` bash
mlflow ui
```

If you need a different port:

``` bash
mlflow ui --port 5050
```

Then open in browser:

    http://127.0.0.1:5000
 

(or the port you specified)

MLflow will show:

-   Model parameters
-   MAE, RMSE, MAPE, R²
-   Comparison plots
-   Artifacts and logs

------------------------------------------------------------------------

## Key Directories Summary

-   `data/raw/` → Raw Google Drive data\
-   `data/frozen/` → Final feature-engineered datasets\
-   `src/models/` → Model implementations\
-   `mlruns/` → MLflow tracking
