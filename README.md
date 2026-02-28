# GEOAI corn field prediction

 ``` Download this shape file from internet tl_2023_us_county.shp``` and place it under raw_dataprep
### Important  Commands
```mlflow ui --port 5050 --backend-store-uri "./Feb22 Final Experiments/mlruns"```

```python dataprep/weather_era5.py --mode live, historical ```

```python dataprep/ndvi_croplevel.py --mode live, historical```

```python src/build_new_features.py --mode live, historical ``` 

## Project Structure

```
.
├── exported_models/        # Final selected models per cutoff
├── mlruns/                 # MLflow experiment tracking directory
├── plots/                  # Auto-generated comparison plots
├── raw_dataprep/           # Raw preprocessing outputs
├── src/
│   ├── analysis/           # Plotting, SHAP, comparison utilities
│   ├── features/           # Feature engineering logic
│   ├── models/             # All model training functions
│   ├── build_new_features.py
│   ├── config.py
│   ├── train.py            # Main training + model selection script
│   └── utils.py
│   └── tests               # Unittest
├── training-dataset/
│   ├── features_frozen/    # Cutoff-specific frozen feature files
│   └── raw/                # Raw source data
```

------------------------------------------------------------------------

## Data Setup

1.  The dataset is stored in **Google Drive**.
2.  Download the data from Google Drive.
3.  Copy raw files into: training-dataset/raw/
4.  After running feature engineering, the processed datasets will be
    saved in: training-dataset/features-frozen/

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


Results:
![img.png](img.png)