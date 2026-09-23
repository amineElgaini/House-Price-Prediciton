# House Price Prediction

Estimate the sale price of a house from its characteristics (surface, quality, location, age, etc.), using supervised regression, and serve the model through an interactive Streamlit app.

## Business problem

A real estate agency wants to automatically estimate a home's sale price from its features, in order to:
- estimate the price of new listings
- identify which characteristics matter most
- compare several regression models
- measure prediction quality
- offer a simple interface for new predictions

## Dataset

- Source: House Prices - Advanced Regression Techniques (Kaggle)
- 1,460 houses with a known sale price, 79 explanatory features (numeric + categorical)
- Target variable: `SalePrice` (US dollars)

Key numeric target stats: mean ≈ $180,921, median $163,000, min $34,900, max $755,000. The distribution is right-skewed (skewness ≈ 1.88), so the model is trained on `log(SalePrice)` and predictions are converted back to dollars.

## Data preparation

1. **Filtered to the true training rows** (`Id <= 1460`) — the raw CSV also contained rows with placeholder/synthetic prices, which were excluded to avoid training on fake targets.
2. **Missing values**, handled by type:
   - Columns where `NaN` means "doesn't have it" (`PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `MasVnrType`, garage and basement quality/type columns) → filled with `"None"`.
   - Numeric columns where `NaN` means 0 (`MasVnrArea`, basement/garage areas and counts) → filled with `0`.
   - Truly missing values (`LotFrontage`, `GarageYrBlt`, `Electrical`, `MSZoning`, etc.) → filled inside the pipeline with the median (numeric) or most frequent value (categorical), fitted on the training set only, to avoid data leakage.
3. **`MSSubClass`** converted from an integer code to a string, since it represents a category, not a quantity.
4. **Outliers removed**: 2 houses with `GrLivArea > 4000` and `SalePrice < 300000` (Id 524 and 1299) — unusually large but cheap properties that broke the area/price trend, most likely partial or non-market sales.
5. **No duplicate rows** found.
6. **Encoding**: categorical columns one-hot encoded; numeric columns standardized (mean 0, std 1).
7. **Train/test split**: 80/20, `random_state=42`. All learned steps (imputation medians, encoding categories, scaling) are fitted on the training set only, inside a scikit-learn `Pipeline` + `ColumnTransformer`, then applied to the test set.

## Feature engineering

| Feature | Formula | Why |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Total living space including the basement; correlates more strongly with price (0.83) than `GrLivArea` alone (0.72) |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | Bathrooms were split across 4 columns; combines them into one comparable count |
| `HouseAge` | `YrSold - YearBuilt` | Age at the time of sale, more meaningful than the raw construction year |
| `YearsSinceRemod` | `YrSold - YearRemodAdd` | Captures the effect of a recent renovation |

## Exploratory analysis — key findings

- **Overall quality** and **living area** both rise steadily with price; the effect of quality accelerates at the top end (8→9→10 adds more than 4→5).
- **Neighborhood** matters strongly: `NoRidge`, `NridgHt`, and `StoneBr` are the most expensive; `MeadowV`, `IDOTRR`, and `BrDale` are the cheapest.
- **Year built** has only a moderate relationship with price (houses built after ~1990 tend to be pricier, but with a lot of spread).
- Several original numeric features are highly correlated with each other (`GarageCars`/`GarageArea`, `TotalBsmtSF`/`1stFlrSF`), i.e. multicollinearity, which mainly affects linear models.

## Models

Three regression models were trained and compared, each inside a `Pipeline` (preprocessing → model), predicting on `log(SalePrice)` via `TransformedTargetRegressor`:

| Model | Type |
|---|---|
| Ridge | Linear, regularized |
| Random Forest | Tree-based ensemble |
| SVR | Non-linear (kernel-based) |

## Cross-validation

5-fold `KFold` cross-validation on the training set (`scoring="r2"`):

| Model | Mean R² | Std |
|---|---|---|
| Ridge | 0.937 | 0.010 |
| Random Forest | 0.885 | 0.006 |
| SVR (default params) | 0.831 | 0.074 |

SVR's high standard deviation flagged it as under-tuned and unstable across folds, which motivated hyperparameter search.

## Hyperparameter optimization (GridSearchCV)

| Model | Grid searched | Best params | Best CV R² |
|---|---|---|---|
| Ridge | `alpha`: [0.1, 1, 10, 30, 50, 100] | `alpha=10` | 0.937 (unchanged) |
| SVR | `C`: [1,10,50], `epsilon`: [0.01,0.05,0.1], `kernel`: [rbf, linear] | `C=1, epsilon=0.01, kernel=linear` | 0.926 (+0.095) |

SVR's best kernel turned out to be `linear`, not `rbf` — evidence that, once the features are well engineered and encoded, the relationship between features and price is largely linear for this dataset. This also explains why Ridge remains the strongest model overall.

## Final evaluation (held-out test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Ridge (tuned)** | **$14,295** | **$20,005** | **0.928** |
| SVR (tuned) | $14,639 | $21,003 | 0.920 |
| Random Forest | $16,324 | $23,545 | 0.900 |

**Ridge (tuned) was selected as the final model** — best score on every metric, on cross-validation, and on the untouched test set, with low variance across folds.

Actual vs predicted prices hug the diagonal closely at lower price ranges, with more spread at higher prices — the model is more confident on typical homes than on expensive/unusual ones. Residuals are centered on 0 and roughly symmetric, indicating no systematic over- or under-pricing bias.

### Largest prediction errors

The 10 worst errors cluster in the $190K–$440K range, mostly on houses with a large `TotalSF` (big basements) relative to their `OverallQual`, and several in the `Crawfor` neighborhood — suggesting the model would benefit from features that interact size with quality, or from neighborhood-specific effects not fully captured by a single linear coefficient per neighborhood.

## Model interpretation

Ridge's largest positive coefficients (features that increase price the most): specific high-value neighborhoods (`Crawfor`, `StoneBr`), premium exterior material (`BrkFace`), `OverallQual`, `GrLivArea`, and `TotalSF`.

Largest negative coefficients: commercial zoning (`MSZoning_C`), the cheapest neighborhood (`MeadowV`), gravity heating, and abnormal sale conditions.

These findings are consistent with the exploratory analysis: `OverallQual`, `GrLivArea`, and `TotalSF` were already the strongest numeric correlates of price, and the categorical effects (neighborhood, zoning) confirm what the boxplots showed — location and zoning matter as much as physical size.

## Streamlit application

The app (`dashboard/app.py`) lets a user enter a subset of the most important features (quality, basement area, garage capacity, bathrooms, year built, neighborhood, 1st/2nd floor area) — the remaining columns are filled with training-set defaults (median for numeric, most frequent for categorical) so the model always receives a complete, valid input row.

The model, default-value template, and neighborhood list are loaded once at startup (`@st.cache_resource`) and never retrained on a prediction.

Features:
- Interactive form for house characteristics
- Instant price estimate
- Percentile: where the estimate falls relative to the training market
- Histogram of training prices with the estimate marked

### Screenshots

![Form input](dashboard/screenshots/form_input.png)
![Prediction result](dashboard/screenshots/prediction_result.png)
![Actual vs predicted](dashboard/screenshots/actual_vs_predicted.png)

### Example predictions

| Inputs | Estimated price | Percentile |
|---|---|---|
| Quality 6, 1000 sq ft, Blmngtn, built 1990 | $142,925 | 36% |
| Quality 9, 3000 sq ft, StoneBr, built 2005 | $318,835 | 94% |

## Architecture

![Pipeline diagram](docs/pipeline_diagram.png)

Raw data → cleaning & feature engineering → preprocessing pipeline → model training & comparison → Streamlit app → Docker container.

## Project structure

```
house-price-prediction/
├── data/                # train.csv
├── notebooks/           # exploration, EDA, feature engineering, modeling
├── models/              # ridge_model.pkl, template_row.pkl, neighborhoods.pkl, y_train.pkl
├── dashboard/            # app.py + screenshots/
├── docs/                # pipeline_diagram.png
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd house-price-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebooks

```bash
jupyter notebook
```
Open the notebooks in `notebooks/` in order (exploration → EDA/feature engineering → modeling) to reproduce the analysis and retrain the model.

## Running the Streamlit app locally

```bash
cd dashboard
streamlit run app.py
```
Open `http://localhost:8501`.

## Running with Docker

From the project root (where the `Dockerfile` is):

```bash
docker build -t house-price-app .
docker run -p 8501:8501 house-price-app
```
Open `http://localhost:8501`.

## Key takeaways

- A carefully engineered feature set (`TotalSF`, `TotalBath`, `HouseAge`, `YearsSinceRemod`) improved correlation with price beyond the raw columns.
- A well-tuned linear model (Ridge) outperformed both a tree ensemble and a kernel method on this dataset, likely because the price relationship is largely linear once features are properly encoded and scaled.
- Cross-validation was essential to catch SVR's instability, which a single train/test split would have missed.
- The largest prediction errors are concentrated on large, unusual, or specific-neighborhood houses — a natural next step would be neighborhood-aware interactions or a stacked/tree-based ensemble for that segment.