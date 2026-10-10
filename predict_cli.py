"""
predict_cli.py
Command-line interface for the Cricket Score Prediction machine learning model.

Usage example:
    python predict_cli.py --format T20 --batting India --bowling Australia --venue "Melbourne Cricket Ground (MCG), Melbourne" --score 95 --wickets 2 --overs 11.2 --runs5 46 --wkts5 1
"""

import sys
import os
import argparse

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predictor import CricketScorePredictor


def main():
    parser = argparse.ArgumentParser(description="Predict Final Cricket Score using Trained ML Models.")
    parser.add_argument("--format", default="T20", choices=["T20", "ODI"], help="Match format (T20 or ODI)")
    parser.add_argument("--batting", default="India", help="Batting team name")
    parser.add_argument("--bowling", default="Australia", help="Bowling team name")
    parser.add_argument("--venue", default="Wankhede Stadium, Mumbai", help="Venue name")
    parser.add_argument("--score", type=int, default=105, help="Current score (runs)")
    parser.add_argument("--wickets", type=int, default=3, help="Wickets fallen (0-9)")
    parser.add_argument("--overs", type=float, default=12.4, help="Overs bowled (e.g. 12.4)")
    parser.add_argument("--runs5", type=int, default=42, help="Runs scored in last 5 overs")
    parser.add_argument("--wkts5", type=int, default=1, help="Wickets lost in last 5 overs")

    args = parser.parse_args()

    predictor = CricketScorePredictor()

    params = {
        "format": args.format,
        "batting_team": args.batting,
        "bowling_team": args.bowling,
        "venue": args.venue,
        "current_score": args.score,
        "wickets": args.wickets,
        "overs": args.overs,
        "runs_last_5": args.runs5,
        "wickets_last_5": args.wkts5
    }

    result = predictor.predict(params)

    if not result["success"]:
        print(f"[!] Error: {result['error']}")
        sys.exit(1)

    state = result["match_state"]
    pred = result["prediction"]
    analytics = result["analytics"]

    print("\n" + "=" * 50)
    print(f"      CRICKET SCORE PREDICTION ({result['format']})")
    print("=" * 50)
    print(f"Match:       {result['batting_team']} vs {result['bowling_team']}")
    print(f"Venue:       {result['venue']}")
    print(f"State:       {state['current_score']}/{state['wickets']} in {state['overs']} overs")
    print(f"Current RR:  {state['current_run_rate']} rpo | Balls Left: {state['balls_remaining']}")
    print(f"Last 5 Ov:   {args.runs5} runs, {args.wkts5} wkts (RR: {state['last_5_run_rate']})")
    print("-" * 50)
    print(f"[*] PREDICTED FINAL SCORE: {pred['final_score']} runs")
    print(f"[*] Estimated Range:       {pred['score_range']} runs (+/- {pred['model_margin_error']})")
    print(f"[*] Simple CRR Projected:  {pred['crr_projected']} runs")
    print(f"[*] Momentum:              {analytics['momentum']}")
    print(f"[*] Future RR Needed:      {analytics['required_future_run_rate']} rpo")
    print(f"[*] Model:                 {pred['model_algorithm']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

