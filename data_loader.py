"""
data_loader.py
Module for loading, validating, and engineering cricket match features.
"""

import os
import pandas as pd
import numpy as np


def convert_overs_to_balls(overs_series):
    """
    Converts cricket overs notation (e.g. 5.3 = 5 overs + 3 balls) to total legal balls bowled.
    """
    overs_int = np.floor(overs_series).astype(int)
    balls = np.round((overs_series - overs_int) * 10).astype(int)
    return (overs_int * 6) + balls


def engineer_features(df, total_overs=20):
    """
    Computes derived cricket analytics features:
    - balls_bowled
    - balls_remaining
    - wickets_in_hand
    - current_run_rate (exact)
    - last_5_run_rate
    """
    df = df.copy()

    # Total legal balls bowled
    df["balls_bowled"] = convert_overs_to_balls(df["overs"])
    
    # Avoid zero division
    df["balls_bowled"] = df["balls_bowled"].clip(lower=1)
    
    total_balls = total_overs * 6
    df["balls_remaining"] = np.maximum(0, total_balls - df["balls_bowled"])
    df["wickets_in_hand"] = np.maximum(0, 10 - df["wickets"])
    
    # Exact current run rate
    df["current_run_rate"] = np.round(df["current_score"] / (df["balls_bowled"] / 6.0), 2)
    
    # Run rate in the last 5 overs (30 balls)
    df["last_5_run_rate"] = np.round(df["runs_last_5"] / 5.0, 2)

    return df


def load_dataset(csv_path, format_type="T20"):
    """
    Loads dataset from CSV, performs feature engineering, and splits into X and y.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    total_overs = 20 if format_type == "T20" else 50
    df = engineer_features(df, total_overs=total_overs)

    feature_cols = [
        "batting_team",
        "bowling_team",
        "venue",
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

    target_col = "final_score"

    X = df[feature_cols]
    y = df[target_col]

    return X, y, df

