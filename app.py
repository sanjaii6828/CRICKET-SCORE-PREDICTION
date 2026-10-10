"""
app.py
Flask Web Application for Cricket Score Prediction.
Serves interactive dashboard, REST API for real-time predictions,
and server-side form submission fallback.

"""

import os
import sys
from flask import Flask, jsonify, render_template, request

# Ensure src is on sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predictor import CricketScorePredictor

app = Flask(__name__)
predictor = CricketScorePredictor()


PRESETS = {
    "preset-t20-powerplay": {
        "format": "T20",
        "batting_team": "India",
        "bowling_team": "Australia",
        "current_score": "55",
        "wickets": "1",
        "overs": "6.0",
        "runs_last_5": "46",
        "wickets_last_5": "1",
    },
    "preset-t20-death": {
        "format": "T20",
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "current_score": "138",
        "wickets": "3",
        "overs": "15.0",
        "runs_last_5": "52",
        "wickets_last_5": "1",
    },
    "preset-odi-middle": {
        "format": "ODI",
        "batting_team": "India",
        "bowling_team": "England",
        "current_score": "180",
        "wickets": "3",
        "overs": "34.0",
        "runs_last_5": "36",
        "wickets_last_5": "0",
    },
}


def _dashboard_context(format_type="T20", form_data=None, prediction_result=None):
    """Builds the server-rendered dashboard data for a cricket format."""
    metadata = predictor.get_metadata()
    format_type = format_type.upper()
    if format_type not in ("T20", "ODI"):
        format_type = "T20"

    format_metadata = metadata.get("formats", {}).get(format_type, {})
    teams = format_metadata.get("teams", [])
    venues = format_metadata.get("venues", [])
    defaults = {
        "format": format_type,
        "batting_team": "India" if "India" in teams else (teams[0] if teams else ""),
        "bowling_team": "Australia" if "Australia" in teams else "",
        "venue": "",
        "current_score": "112",
        "wickets": "3",
        "overs": "13.2" if format_type == "T20" else "32.0",
        "runs_last_5": "45",
        "wickets_last_5": "1",
    }
    if not defaults["bowling_team"] or defaults["bowling_team"] == defaults["batting_team"]:
        defaults["bowling_team"] = next(
            (team for team in teams if team != defaults["batting_team"]), ""
        )
    if venues:
        defaults["venue"] = next(
            (venue for venue in venues if format_type == "T20" and "Wankhede" in venue),
            venues[0],
        )

    form_data = {**defaults, **(form_data or {})}
    form_data["format"] = format_type
    if form_data.get("batting_team") not in teams:
        form_data["batting_team"] = defaults["batting_team"]
    if form_data.get("bowling_team") not in teams or form_data.get("bowling_team") == form_data.get("batting_team"):
        form_data["bowling_team"] = defaults["bowling_team"]
    if form_data.get("venue") not in venues:
        form_data["venue"] = defaults["venue"]

    hud = {"crr": "--", "balls_remaining": "--", "wickets_in_hand": "--", "projected_score": "--"}
    try:
        score = int(form_data["current_score"])
        wickets = int(form_data["wickets"])
        overs = float(form_data["overs"])
        completed_overs = int(overs)
        balls_in_over = round((overs - completed_overs) * 10)
        balls_bowled = completed_overs * 6 + balls_in_over
        if balls_bowled > 0:
            total_overs = 20 if format_type == "T20" else 50
            run_rate = score / (balls_bowled / 6)
            hud = {
                "crr": f"{run_rate:.2f}",
                "balls_remaining": max(0, total_overs * 6 - balls_bowled),
                "wickets_in_hand": max(0, 10 - wickets),
                "projected_score": round(run_rate * total_overs),
            }
    except (TypeError, ValueError):
        pass

    has_prediction = bool(prediction_result and prediction_result.get("success"))
    result_progress = 0
    if has_prediction:
        match_state = prediction_result["match_state"]
        final_score = max(1, prediction_result["prediction"]["final_score"])
        result_progress = min(
            100,
            round(match_state["current_score"] / final_score * 100),
        )

    return {
        "metadata": metadata,
        "teams": teams,
        "venues": venues,
        "t20_metrics": metadata.get("formats", {}).get("T20", {}).get("metrics", {}),
        "odi_metrics": metadata.get("formats", {}).get("ODI", {}).get("metrics", {}),
        "form_data": form_data,
        "prediction_result": prediction_result,
        "has_prediction": has_prediction,
        "result_progress": result_progress,
        "hud": hud,
        "total_overs": 20 if format_type == "T20" else 50,
    }


@app.route("/")
def index():
    """Serves the main Cricket Score Prediction dashboard."""
    format_type = request.args.get("format", "T20").upper()
    return render_template("index.html", **_dashboard_context(format_type))


@app.route("/api/meta", methods=["GET"])
def get_meta():
    """Returns format-specific metadata (teams, venues, metrics)."""
    fmt = request.args.get("format", "T20").upper()
    metadata = predictor.get_metadata()
    fmt_data = metadata.get("formats", {}).get(fmt, {})
    return jsonify({
        "format": fmt,
        "teams": fmt_data.get("teams", []),
        "venues": fmt_data.get("venues", []),
        "metrics": fmt_data.get("metrics", {}),
        "best_algorithm": fmt_data.get("best_algorithm", "")
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """AJAX endpoint for score prediction."""
    if not request.is_json:
        # Fallback to form data if json header wasn't set
        data = request.form.to_dict()
    else:
        data = request.get_json()

    result = predictor.predict(data)
    if not result.get("success"):
        return jsonify(result), 400

    return jsonify(result), 200


@app.route("/predict", methods=["POST"])
def form_predict():
    """Handles format changes, presets, and predictions with standard HTML forms."""
    data = request.form.to_dict()
    action = data.pop("action", "predict")

    if action in PRESETS:
        data = PRESETS[action].copy()
        action = "preset"

    format_type = data.get("format", "T20").upper()
    if action == "change-format":
        try:
            if format_type == "T20" and float(data.get("overs", 0)) >= 20:
                data["overs"] = "13.2"
        except (TypeError, ValueError):
            pass
        return render_template("index.html", **_dashboard_context(format_type, data))
    if action == "preset":
        return render_template("index.html", **_dashboard_context(format_type, data))

    if action != "predict":
        result = {"success": False, "error": "Unknown form action."}
    else:
        result = predictor.predict(data)

    return render_template(
        "index.html",
        **_dashboard_context(format_type, data, prediction_result=result),
    )


if __name__ == "__main__":
    # Start local development server
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Cricket Score Prediction Web App on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
