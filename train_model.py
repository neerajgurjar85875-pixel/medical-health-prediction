# train_model.py
# AI-Based Medical Health Insurance Cost Prediction
# Dataset columns:
# age, sex, bmi, children, smoker, region, charges

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
DATA_FILE = "insurance(in).csv"

df = pd.read_csv(DATA_FILE)

print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing Values:")
print(df.isna().sum())


# ---------------------------------------------------------
# 2. Features and Target
# ---------------------------------------------------------
X = df.drop("charges", axis=1)
y = df["charges"]


# ---------------------------------------------------------
# 3. Train-Test Split
# Same 80/20 approach used in the notebook
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=121
)


# ---------------------------------------------------------
# 4. Column Types
# ---------------------------------------------------------
numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]


# ---------------------------------------------------------
# 5. Preprocessing
# Numerical:
#   Median imputation + StandardScaler
#
# Categorical:
#   Most-frequent imputation + OneHotEncoder
# ---------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ---------------------------------------------------------
# 6. Models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),

    "SGD Regressor": SGDRegressor(
        loss="squared_error",
        max_iter=20000,
        eta0=0.00001,
        n_iter_no_change=10,
        random_state=42
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=8
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=7
    )
}


# ---------------------------------------------------------
# 7. Train and Compare Models
# ---------------------------------------------------------
results = []
trained_pipelines = {}

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        "Model": name,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2
    })

    trained_pipelines[name] = pipeline

    print(f"\n{name}")
    print(f"  Train MAE  : {train_mae:.2f}")
    print(f"  Test MAE   : {test_mae:.2f}")
    print(f"  Train RMSE : {train_rmse:.2f}")
    print(f"  Test RMSE  : {test_rmse:.2f}")
    print(f"  Train R2   : {train_r2:.4f}")
    print(f"  Test R2    : {test_r2:.4f}")


# ---------------------------------------------------------
# 8. Results Table
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Test R2",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))


# ---------------------------------------------------------
# 9. Select Best Model
# Higher Test R2 = better regression performance
# ---------------------------------------------------------
best_model_name = results_df.loc[0, "Model"]
best_pipeline = trained_pipelines[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print("Best Model:", best_model_name)
print("Best Test R2:", round(results_df.loc[0, "Test R2"], 4))


# ---------------------------------------------------------
# 10. Save Complete Pipeline
# The saved pipeline includes preprocessing + model.
# Flask can directly use this file for prediction.
# ---------------------------------------------------------
joblib.dump(best_pipeline, "model.pkl")

# Save comparison results for the future website
results_df.to_csv("model_comparison.csv", index=False)

print("\nSaved files:")
print("1. model.pkl")
print("2. model_comparison.csv")
print("\nTraining completed successfully!")
