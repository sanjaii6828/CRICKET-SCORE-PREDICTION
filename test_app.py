"""
test_app.py
Comprehensive test suite for Cricket Score Prediction.
Tests predictor logic, data validation, domain constraints, and Flask API endpoints.
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app import app
from src.predictor import CricketScorePredictor


class TestCricketScorePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = CricketScorePredictor()
        cls.client = app.test_client()

    def test_predictor_loaded(self):
        """Verify models and metadata are successfully loaded."""
        self.assertIn("T20", self.predictor.models)
        self.assertIn("ODI", self.predictor.models)
        meta = self.predictor.get_metadata()
        self.assertIn("formats", meta)
        self.assertIn("T20", meta["formats"])
        self.assertIn("ODI", meta["formats"])

    def test_t20_prediction_valid(self):
        """Test standard valid T20 match prediction."""
        params = {
            "format": "T20",
            "batting_team": "India",
            "bowling_team": "Australia",
            "venue": "Wankhede Stadium, Mumbai",
            "current_score": 115,
            "wickets": 2,
            "overs": 13.0,
            "runs_last_5": 48,
            "wickets_last_5": 1
        }
        res = self.predictor.predict(params)
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["prediction"]["final_score"], 115)
        self.assertGreaterEqual(res["prediction"]["upper_bound"], res["prediction"]["lower_bound"])
        self.assertIn("analytics", res)
        self.assertIn("momentum", res["analytics"])

    def test_odi_prediction_valid(self):
        """Test standard valid ODI match prediction."""
        params = {
            "format": "ODI",
            "batting_team": "India",
            "bowling_team": "England",
            "venue": "Lord's Cricket Ground, London",
            "current_score": 210,
            "wickets": 3,
            "overs": 38.0,
            "runs_last_5": 40,
            "wickets_last_5": 0
        }
        res = self.predictor.predict(params)
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["prediction"]["final_score"], 210)
        self.assertEqual(res["format"], "ODI")

    def test_invalid_team_matchup(self):
        """Batting and bowling team cannot be identical."""
        params = {
            "format": "T20",
            "batting_team": "India",
            "bowling_team": "India",
            "venue": "Eden Gardens, Kolkata",
            "current_score": 80,
            "wickets": 2,
            "overs": 10.0,
            "runs_last_5": 35,
            "wickets_last_5": 1
        }
        res = self.predictor.predict(params)
        self.assertFalse(res["success"])
        self.assertIn("cannot be the same", res["error"])

    def test_invalid_overs(self):
        """Overs cannot exceed format limit or have invalid ball counts."""
        # Ball 7 in over (10.7 is illegal cricket overs)
        params_illegal_ball = {
            "format": "T20",
            "batting_team": "India",
            "bowling_team": "Pakistan",
            "venue": "Dubai International Cricket Stadium, Dubai",
            "current_score": 80,
            "wickets": 2,
            "overs": 10.7,
            "runs_last_5": 30,
            "wickets_last_5": 1
        }
        res1 = self.predictor.predict(params_illegal_ball)
        self.assertFalse(res1["success"])

        # Overs > 20 in T20
        params_over_limit = {
            "format": "T20",
            "batting_team": "India",
            "bowling_team": "Pakistan",
            "venue": "Dubai International Cricket Stadium, Dubai",
            "current_score": 80,
            "wickets": 2,
            "overs": 21.0,
            "runs_last_5": 30,
            "wickets_last_5": 1
        }
        res2 = self.predictor.predict(params_over_limit)
        self.assertFalse(res2["success"])

    def test_flask_index_page(self):
        """Verify home route renders with status 200."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CricPredict ML", response.data)
        self.assertIn(b"Live Match State", response.data)
        self.assertNotIn(b"<script", response.data.lower())

    def test_form_format_change_uses_odi_metadata(self):
        """Format switching uses a regular form post and renders ODI choices."""
        response = self.client.post(
            "/predict",
            data={"action": "change-format", "format": "ODI"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="format" value="ODI" checked', response.data)
        self.assertIn(b'max="49.5"', response.data)
        self.assertIn(b"Lord&#39;s Cricket Ground, London", response.data)

    def test_form_preset_fills_match_values(self):
        """A preset fills the form without relying on browser-side scripts."""
        response = self.client.post(
            "/predict",
            data={"action": "preset-t20-powerplay", "format": "T20"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'value="55"', response.data)
        self.assertIn(b'value="6.0"', response.data)
        self.assertNotIn(b"<script", response.data.lower())

    def test_form_prediction_renders_server_side_result(self):
        """Predictions are rendered by Flask after a normal form submission."""
        response = self.client.post(
            "/predict",
            data={
                "action": "predict",
                "format": "T20",
                "batting_team": "India",
                "bowling_team": "Australia",
                "venue": "Wankhede Stadium, Mumbai",
                "current_score": "115",
                "wickets": "2",
                "overs": "13.0",
                "runs_last_5": "48",
                "wickets_last_5": "1",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"EXPECTED FINAL SCORE", response.data)
        self.assertNotIn(b'class="alert alert-error"', response.data)

    def test_flask_api_meta(self):
        """Verify API metadata endpoint returns teams and venues."""
        response = self.client.get("/api/meta?format=T20")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data["format"], "T20")
        self.assertIn("India", json_data["teams"])
        self.assertIn("Wankhede Stadium, Mumbai", json_data["venues"])

    def test_flask_api_predict(self):
        """Verify API predict endpoint processes JSON requests."""
        payload = {
            "format": "T20",
            "batting_team": "Chennai Super Kings",
            "bowling_team": "Mumbai Indians",
            "venue": "Wankhede Stadium, Mumbai",
            "current_score": 130,
            "wickets": 3,
            "overs": 15.0,
            "runs_last_5": 45,
            "wickets_last_5": 1
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertGreaterEqual(json_data["prediction"]["final_score"], 130)


if __name__ == "__main__":
    unittest.main()
