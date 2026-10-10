"""
train.py
Trains and evaluates multiple machine learning models for Cricket Score Prediction.
Selects the best performing model based on Mean Absolute Error (MAE) and saves the pipeline.

"""

import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime

# Ensure src is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_preprocessor():
    """Builds categorical and numerical ColumnTransformer."""
    categorical_features = ["batting_team", "bowling_team", "venue"]
    numerical_features = [
        "current_score",
        "wickets",
        "overs",
        "runs_last_5",
        "wickets_last_5",
        "balls_remaining",
        "wickets_in_hand",
        "current_run_rate",
        "last_5_run_rate"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )
    return preprocessor


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Evaluates Ridge, Random Forest, and Gradient Boosting Regressors.
    Returns evaluation metrics and the best performing model pipeline.
    """
    preprocessor = get_preprocessor()

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42)
    }

    results = {}
    best_name = None
    best_mae = float("inf")
    best_pipeline = None

    for name, regressor in candidates.items():
        print(f"  --> Training {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "MAE": round(float(mae), 3),
            "RMSE": round(float(rmse), 3),
            "R2": round(float(r2), 4)
        }
        print(f"      {name} -> MAE: {mae:.2f} runs | RMSE: {rmse:.2f} | R2: {r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_pipeline = pipeline

    print(f"  [+] Best Model: {best_name} with MAE {best_mae:.2f} runs")
    return best_name, best_pipeline, results


def run_training():
    """Trains models for both T20 and ODI formats and saves artifacts."""
    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "formats": {}
    }

    formats = [
        ("T20", os.path.join(DATA_DIR, "t20_matches.csv")),
        ("ODI", os.path.join(DATA_DIR, "odi_matches.csv"))
    ]

    for format_type, csv_path in formats:
        print(f"\n==========================================")
        print(f"Starting Training for {format_type} Cricket Model")
        print(f"==========================================")

        X, y, df = load_dataset(csv_path, format_type=format_type)
        print(f"Total dataset shape: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_name, best_pipeline, results = train_and_evaluate_models(
            X_train, X_test, y_train, y_test
        )

        # Save model pipeline
        model_filename = f"{format_type.lower()}_model.joblib"
        model_save_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(best_pipeline, model_save_path)
        print(f"Saved champion pipeline to {model_save_path}")

        # Unique teams and venues
        batting_teams = sorted(df["batting_team"].unique().tolist())
        venues = sorted(df["venue"].unique().tolist())

        metadata["formats"][format_type] = {
            "model_file": model_filename,
            "best_algorithm": best_name,
            "metrics": results[best_name],
            "all_model_benchmarks": results,
            "total_samples": len(df),
            "teams": batting_teams,
            "venues": venues,
            "features": list(X.columns)
        }

    # Save metadata JSON
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata successfully saved to {meta_path}")


if __name__ == "__main__":
    run_training()

