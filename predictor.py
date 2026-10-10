"""
predictor.py
Unified prediction engine for T20 and ODI Cricket Score Prediction.
Performs domain validation, feature engineering, ML inference,
and cricket analytics (CRR, projected score, momentum index, confidence interval).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")


class CricketScorePredictor:
    """Predicts expected cricket score using trained machine learning pipelines."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.metadata = {}
        self._load_artifacts()

    def _load_artifacts(self):
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        for fmt in ["T20", "ODI"]:
            model_path = os.path.join(self.models_dir, f"{fmt.lower()}_model.joblib")
            if os.path.exists(model_path):
                self.models[fmt] = joblib.load(model_path)

    def get_metadata(self):
        """Returns loaded model metadata."""
        return self.metadata

    def validate_inputs(self, params):
        """
        Validates match parameters against cricket rules and constraints.
        Returns (is_valid, error_message).
        """
        format_type = params.get("format", "T20").upper()
        if format_type not in ["T20", "ODI"]:
            return False, f"Invalid format '{format_type}'. Must be T20 or ODI."

        batting_team = params.get("batting_team")
        bowling_team = params.get("bowling_team")
        venue = params.get("venue")

        if not batting_team or not bowling_team or not venue:
            return False, "Batting team, bowling team, and venue are all required."

        if batting_team == bowling_team:
            return False, "Batting team and bowling team cannot be the same."

        max_overs = 20.0 if format_type == "T20" else 50.0

        try:
            overs = float(params.get("overs", 0))
            current_score = int(params.get("current_score", 0))
            wickets = int(params.get("wickets", 0))
            runs_last_5 = int(params.get("runs_last_5", 0))
            wickets_last_5 = int(params.get("wickets_last_5", 0))
        except (ValueError, TypeError) as e:
            return False, f"Invalid numeric input: {e}"

        if overs <= 0.0 or overs >= max_overs:
            return False, f"Overs must be greater than 0 and less than {max_overs}."

        over_int = int(overs)
        balls_part = round((overs - over_int) * 10)
        if balls_part > 5:
            return False, f"Invalid overs notation '{overs}'. Legal balls in an over are 0 to 5 (e.g. 10.5)."

        if current_score < 0:
            return False, "Current score cannot be negative."

        if wickets < 0 or wickets >= 10:
            return False, "Wickets fallen must be between 0 and 9 (at 10 wickets, innings has ended)."

        if runs_last_5 < 0:
            return False, "Runs in last 5 overs cannot be negative."

        if runs_last_5 > current_score and overs <= 5.0:
            return False, "Runs in last 5 overs cannot exceed total score when overs <= 5."

        if wickets_last_5 < 0:
            return False, "Wickets in last 5 overs cannot be negative."

        if wickets_last_5 > wickets:
            return False, f"Wickets in last 5 overs ({wickets_last_5}) cannot exceed total wickets lost ({wickets})."

        return True, ""

    def predict(self, params):
        """
        Executes prediction and cricket analytics.
        """
        is_valid, err = self.validate_inputs(params)
        if not is_valid:
            return {"success": False, "error": err}

        format_type = params["format"].upper()
        if format_type not in self.models:
            return {"success": False, "error": f"Model for format {format_type} is not loaded."}

        model = self.models[format_type]
        total_overs = 20.0 if format_type == "T20" else 50.0
        total_balls = int(total_overs * 6)

        overs = float(params["overs"])
        over_int = int(overs)
        balls_in_over = round((overs - over_int) * 10)
        balls_bowled = max(1, (over_int * 6) + balls_in_over)
        balls_remaining = max(0, total_balls - balls_bowled)

        current_score = int(params["current_score"])
        wickets = int(params["wickets"])
        wickets_in_hand = max(0, 10 - wickets)

        runs_last_5 = int(params["runs_last_5"])
        wickets_last_5 = int(params["wickets_last_5"])

        overs_decimal = balls_bowled / 6.0
        current_run_rate = round(current_score / overs_decimal, 2)
        last_5_run_rate = round(runs_last_5 / 5.0, 2)
        crr_projected_score = int(round(current_run_rate * total_overs))

        # Build feature DataFrame matching pipeline features
        features_dict = {
            "batting_team": [params["batting_team"]],
            "bowling_team": [params["bowling_team"]],
            "venue": [params["venue"]],
            "current_score": [current_score],
            "wickets": [wickets],
            "overs": [overs],
            "runs_last_5": [runs_last_5],
            "wickets_last_5": [wickets_last_5],
            "balls_remaining": [balls_remaining],
            "wickets_in_hand": [wickets_in_hand],
            "current_run_rate": [current_run_rate],
            "last_5_run_rate": [last_5_run_rate]
        }
        df_features = pd.DataFrame(features_dict)

        raw_pred = model.predict(df_features)[0]
        # Final score cannot be less than current score
        predicted_score = max(current_score, int(round(raw_pred)))

        # Format MAE tolerance
        fmt_meta = self.metadata.get("formats", {}).get(format_type, {})
        mae = fmt_meta.get("metrics", {}).get("MAE", 10.0)

        lower_bound = max(current_score, int(round(predicted_score - mae)))
        upper_bound = int(round(predicted_score + mae))

        # Momentum indicator
        if wickets_last_5 >= 3:
            momentum = "Under Pressure (Recent Collapse)"
            momentum_color = "#ef4444"
        elif last_5_run_rate > current_run_rate * 1.15:
            momentum = "Accelerating (High Scoring Phase)"
            momentum_color = "#10b981"
        elif last_5_run_rate < current_run_rate * 0.85:
            momentum = "Slowing Down (Tight Bowling)"
            momentum_color = "#f59e0b"
        else:
            momentum = "Steady Progression"
            momentum_color = "#3b82f6"

        return {
            "success": True,
            "format": format_type,
            "batting_team": params["batting_team"],
            "bowling_team": params["bowling_team"],
            "venue": params["venue"],
            "match_state": {
                "overs": overs,
                "current_score": current_score,
                "wickets": wickets,
                "balls_remaining": balls_remaining,
                "wickets_in_hand": wickets_in_hand,
                "current_run_rate": current_run_rate,
                "last_5_run_rate": last_5_run_rate
            },
            "prediction": {
                "final_score": predicted_score,
                "score_range": f"{lower_bound} - {upper_bound}",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "crr_projected": crr_projected_score,
                "model_margin_error": round(mae, 1),
                "model_algorithm": fmt_meta.get("best_algorithm", "Gradient Boosting")
            },
            "analytics": {
                "momentum": momentum,
                "momentum_color": momentum_color,
                "required_future_run_rate": round(
                    max(0, predicted_score - current_score) / (balls_remaining / 6.0) if balls_remaining > 0 else 0,
                    2
                )
            }
        }

