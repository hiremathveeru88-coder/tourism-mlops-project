
import os
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# Load train/test data
Xtrain = pd.read_csv(
    "tourism_project/model_building/Xtrain.csv"
)

Xtest = pd.read_csv(
    "tourism_project/model_building/Xtest.csv"
)

ytrain = pd.read_csv(
    "tourism_project/model_building/ytrain.csv"
).squeeze()

ytest = pd.read_csv(
    "tourism_project/model_building/ytest.csv"
).squeeze()


# Numeric features
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]


# Categorical features
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]


# Handle class imbalance
class_weight = (
    ytrain.value_counts()[0]
    / ytrain.value_counts()[1]
)

print("Class weight:", class_weight)


# Preprocessing pipeline
preprocessor = make_column_transformer(
    (
        StandardScaler(),
        numeric_features
    ),
    (
        OneHotEncoder(
            handle_unknown="ignore"
        ),
        categorical_features
    )
)


# XGBoost model
model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)


# Small grid for faster GitHub Actions execution
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}


# Complete ML pipeline
pipeline = make_pipeline(
    preprocessor,
    model
)


# Grid search
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)


print("Starting model training...")

grid.fit(
    Xtrain,
    ytrain
)


# Best model
best_model = grid.best_estimator_


print("\nBest parameters:")
print(grid.best_params_)


# Evaluate model
predictions = best_model.predict(Xtest)

print("\nClassification Report:")
print(
    classification_report(
        ytest,
        predictions
    )
)


# Create deployment directory
os.makedirs(
    "tourism_project/deployment",
    exist_ok=True
)


# Save model
model_path = (
    "tourism_project/deployment/"
    "best_tourism_model_v1.joblib"
)


joblib.dump(
    best_model,
    model_path
)


print(
    f"\nModel saved to: {model_path}"
)
