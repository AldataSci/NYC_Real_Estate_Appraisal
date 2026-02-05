## NYC Real Estate AI Appraiser (Condo Market Value)

Predicting NYC condominium **Full Market Value** using building characteristics and neighborhood, with a focus on avoiding data leakage and building an interpretable regression model.

---

## 🧾 Project Overview

- Built a regression model to estimate **Full Market Value** of NYC condos.
- Uses **physical + location features only** (no income/expense leakage).
- End-to-end pipeline: data cleaning → feature engineering → encoding → modeling → evaluation → interpretability.

---

## 📊 Data & Features

**Target**

- `Full_Market_Value` (modeled in log-space, then converted back to dollars).

**Final Features (Physical-Only Model)**

- Numeric: `Total_Units`, `Gross_SqFt`, `Year_Built`, `Building_Age`, `Report_Year`
- Categorical: `Neighborhood`, `Building_Classification`

**Preprocessing**

- Removed invalid rows and extreme outliers (kept values ≤ 99th percentile of `Full_Market_Value`).
- Log-transform on target: `log(Full_Market_Value)` to handle skew.
- `ColumnTransformer` with:
  - `OneHotEncoder(handle_unknown="ignore")` for categoricals
  - `remainder="passthrough"` for numeric features

---

## ⚠️ Data Leakage Handling

**Data Source:** [NYC Condo Dataset](<https://www.openml.org/search?type=data&status=active&id=43361>)

Early versions included:

- `Net_Operating_Income`
- `Estimated_Gross_Income`
- `Estimated_Expense`
- Income-per-square-foot features

These produced **R² ≈ 0.98**, but they leak information because market value is often derived directly from income:

> Value ≈ NOI / Cap Rate

Final model **removes all income/expense-derived features**, so it can appraise buildings using only observable physical traits and neighborhood.

---

## 🤖 Model & Metrics (Test Set, Physical-Only)

- Model: `RandomForestRegressor`
- **R²:** `0.964`
- **MAE:** `≈ $1,375,807`
- **RMSE:** `≈ $3,679,073`

Interpretation:  
Model explains ~96% of variance in condo values using only size, age, units, and location. Errors are reasonable relative to multi-million-dollar property values.

---

## 🔍 Interpretability

**Feature Importance (Top Drivers)**

1. `Gross_SqFt` – dominant driver (~84% importance)
2. `Total_Units`, `Year_Built`, `Building_Age`, `Report_Year`
3. `Neighborhood indicators` (e.g., Chelsea, Upper East Side, Flushing, Morrisania/Longwood)

**Residual Analysis**

- Residuals mostly positive → model tends to be **conservative** (underestimates some properties, likely missing amenities / views / micro-location signals).
- Higher volatility for lower-priced properties; tighter band for very high-value buildings.

---

## 🧪 How to Run

1. Clone the repo and open the notebook (e.g. in Colab).
2. Install dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
3. Run notebook cells top-to-bottom to:
   - Load & clean data
   - Transform features
   - Train Random Forest model
   - Evaluate and plot feature importance + residuals

---

## 🚀 Possible Extensions

- Add geospatial features (distance to subway/parks/CBD).
- Try gradient boosting models (XGBoost / LightGBM / CatBoost).
- Neighborhood-specific or hierarchical models.
- More advanced cross-validation (e.g., grouped by neighborhood).
