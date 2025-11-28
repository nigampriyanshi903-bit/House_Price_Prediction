# House_Price_Prediction
Advanced House Price Prediction using XGBoost Regressor and extensive Feature Engineering. Achieved R² score of 0.9055 and MAE of $67.7k. Includes Log Transformation, Zipcode Encoding, Linear Regression, and Ridge Regularization

##  Project Goal and Objective

The primary goal of this project was to **develop a highly accurate regression model** to predict the sale price of houses in King County, Washington (Seattle area). We aimed to test whether advanced **non-linear models** (XGBoost) could significantly outperform traditional **linear models** (Linear Regression/Ridge) after extensive data preparation.

##  Key Performance Summary

The **XGBoost Regressor** was selected as the final, most robust model based on its superior performance across all metrics.

| Metric | Linear Regression (Baseline) | **XGBoost Regressor (Final Model)** | Improvement |
| :--- | :--- | :--- | :--- |
| **R-squared Score (R²)** | 0.8779 | **0.9055** | **+2.76%** |
| **Mean Absolute Error (MAE)** | $80,922.85 | **$67,742.95** | **$13,179 reduction** |

---

##  Detailed Methodology and Steps

Our approach followed a structured Machine Learning pipeline, focusing heavily on feature optimization.

### 1. Data Cleaning and EDA

* **Missing Values:** Confirmed **zero missing values** in key columns.
* **Initial Analysis:** Observed that the target variable (`price`) was **highly right-skewed** , indicating a high concentration of outliers (expensive homes) which requires transformation.
* **Correlation:** Confirmed strong correlation between price and structural features like `sqft_living`, `grade`, and `sqft_above`.

### 2. Feature Engineering and Preprocessing

* **Log Transformation (Skewness Handling):** Applied `np.log()` to the target variable (`price`) to convert its distribution to near-normal, which is essential for linear model accuracy.
* **Feature Creation:** Created predictive features:
    * `house_age` from `yr_built`.
    * `is_renovated` from `yr_renovated`.
* **Categorical Encoding:** Converted the crucial geographical feature (`zipcode`) into model-usable numerical format using **One-Hot Encoding**.

### 3. Model Training and Evaluation

We used a **80/20 train-test split** and evaluated models on the log-transformed data, converting errors back to the original price scale for easy interpretation.

* **Baseline Model (Linear Regression):** Achieved a strong $R^2$ of **0.8779**.
* **Regularization (Ridge Regression):** Used **GridSearchCV** to apply L2 regularization ($\alpha=0.1$) to stabilize coefficients, especially for the high number of encoded zipcodes.
* **Advanced Modeling (XGBoost Regressor):** The non-linear gradient boosting algorithm was employed and resulted in the **highest performance** ($R^2=0.9055$).

---

##  Repository Contents

| File Name | Description |
| :--- | :--- |
| `House_Price_Prediction.ipynb` | The complete, executable Jupyter Notebook containing all code from Step 1 through Step 8 (XGBoost). |
| `README.md` | This detailed project summary and methodology. |
| `requirements.txt` | Lists all necessary Python dependencies, including `pandas`, `scikit-learn`, and `xgboost`. |

##  Dependencies (`requirements.txt`)
