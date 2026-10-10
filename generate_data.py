"""
generate_data.py
Synthesizes realistic cricket match progression data for both T20 and ODI formats.
Reflects historical scoring rates, phase transitions (Powerplay, Middle, Death),
wickets fallen, venue pitch characteristics, and team batting/bowling strength ratings.
"""

import os
import numpy as np
import pandas as pd

# Fix random seed for reproducible datasets
np.random.seed(42)

# Directory paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATA_DIR, exist_ok=True)

# Team lists
T20_TEAMS = [
    "India", "Australia", "England", "South Africa", "Pakistan",
    "New Zealand", "West Indies", "Sri Lanka", "Afghanistan", "Bangladesh",
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants", "Punjab Kings"
]

ODI_TEAMS = [
    "India", "Australia", "England", "South Africa", "Pakistan",
    "New Zealand", "West Indies", "Sri Lanka", "Afghanistan", "Bangladesh",
    "Netherlands", "Ireland"
]

# Team strength indices (batting strength, bowling economy impact)
TEAM_STRENGTH = {
    "India": (1.10, 0.92),
    "Australia": (1.08, 0.93),
    "England": (1.07, 0.95),
    "South Africa": (1.05, 0.95),
    "Pakistan": (1.00, 0.92),
    "New Zealand": (1.04, 0.94),
    "West Indies": (1.02, 1.05),
    "Sri Lanka": (0.96, 1.00),
    "Afghanistan": (0.95, 0.94),
    "Bangladesh": (0.92, 1.02),
    "Netherlands": (0.86, 1.08),
    "Ireland": (0.88, 1.06),
    "Chennai Super Kings": (1.06, 0.94),
    "Mumbai Indians": (1.06, 0.95),
    "Royal Challengers Bengaluru": (1.07, 1.02),
    "Kolkata Knight Riders": (1.05, 0.96),
    "Rajasthan Royals": (1.03, 0.96),
    "Delhi Capitals": (1.01, 1.00),
    "Sunrisers Hyderabad": (1.08, 1.01),
    "Gujarat Titans": (1.03, 0.94),
    "Lucknow Super Giants": (1.02, 0.97),
    "Punjab Kings": (1.00, 1.02)
}

# Venues and pitch run-rate adjustment factors (1.0 = average)
VENUES = {
    "Wankhede Stadium, Mumbai": 1.08,
    "M. Chinnaswamy Stadium, Bengaluru": 1.12,
    "Eden Gardens, Kolkata": 1.04,
    "Melbourne Cricket Ground (MCG), Melbourne": 0.95,
    "Sydney Cricket Ground (SCG), Sydney": 1.02,
    "Lord's Cricket Ground, London": 0.97,
    "The Oval, London": 1.03,
    "Dubai International Cricket Stadium, Dubai": 0.96,
    "Narendra Modi Stadium, Ahmedabad": 1.01,
    "SuperSport Park, Centurion": 1.05,
    "Kensington Oval, Bridgetown": 0.98,
    "Gaddafi Stadium, Lahore": 1.03
}


def simulate_innings(format_type="T20"):
    """
    Simulates a full innings ball-by-ball and extracts a random realistic mid-innings snapshot
    along with the final score.
    """
    if format_type == "T20":
        total_overs = 20
        teams = T20_TEAMS
        base_rr = 8.2
        min_snapshot_over = 5.0
        max_snapshot_over = 19.2
    else:  # ODI
        total_overs = 50
        teams = ODI_TEAMS
        base_rr = 5.6
        min_snapshot_over = 10.0
        max_snapshot_over = 48.0

    batting_team = np.random.choice(teams)
    bowling_candidates = [t for t in teams if t != batting_team]
    bowling_team = np.random.choice(bowling_candidates)
    venue = np.random.choice(list(VENUES.keys()))

    bat_factor, _ = TEAM_STRENGTH.get(batting_team, (1.0, 1.0))
    _, bowl_factor = TEAM_STRENGTH.get(bowling_team, (1.0, 1.0))
    venue_factor = VENUES.get(venue, 1.0)

    # Combined base scoring rate per over
    overall_match_factor = bat_factor * bowl_factor * venue_factor

    # Simulate over by over
    overs_data = []
    current_runs = 0
    current_wickets = 0

    for over_num in range(1, total_overs + 1):
        if current_wickets >= 10:
            # All out, innings ended
            break

        # Phase factor
        if format_type == "T20":
            if over_num <= 6:
                phase_mult = 1.05  # Powerplay
                wkt_prob = 0.16
            elif over_num <= 15:
                phase_mult = 0.95  # Middle consolidation
                wkt_prob = 0.18
            else:
                # Death overs: higher run rate, higher wicket chance
                wickets_in_hand = 10 - current_wickets
                phase_mult = 1.30 * (0.6 + 0.04 * wickets_in_hand)
                wkt_prob = 0.28
        else:
            # ODI
            if over_num <= 10:
                phase_mult = 0.95  # Initial Powerplay
                wkt_prob = 0.12
            elif over_num <= 40:
                phase_mult = 0.92  # Middle accumulation
                wkt_prob = 0.14
            else:
                # Death overs
                wickets_in_hand = 10 - current_wickets
                phase_mult = 1.35 * (0.55 + 0.045 * wickets_in_hand)
                wkt_prob = 0.26

        # Expected runs this over
        exp_runs = base_rr * overall_match_factor * phase_mult
        # Add random noise
        over_runs = max(0, int(np.round(np.random.normal(loc=exp_runs, scale=3.5))))
        current_runs += over_runs

        # Wickets in this over
        # Adjust wicket probability based on wickets in hand
        prob = wkt_prob * (1.0 + (10 - current_wickets) * 0.02)
        wkts_fallen = np.random.binomial(n=1, p=min(0.7, prob))
        if wkts_fallen and current_wickets < 10:
            # Small chance of double wicket in an over
            if np.random.rand() < 0.15 and current_wickets + 1 < 10:
                wkts_fallen = 2
            current_wickets = min(10, current_wickets + wkts_fallen)

        overs_data.append({
            "over": over_num,
            "runs_cum": current_runs,
            "wkts_cum": current_wickets,
            "over_runs": over_runs
        })

    final_score = current_runs
    final_wickets = current_wickets
    completed_overs = len(overs_data)

    if completed_overs < int(min_snapshot_over) + 1:
        # Match collapsed too fast, skip snapshot
        return None

    # Pick a random snapshot over where match is still ongoing (wickets < 10)
    possible_snapshot_overs = [
        o for o in overs_data
        if min_snapshot_over <= o["over"] <= min(max_snapshot_over, completed_overs - 1)
        and o["wkts_cum"] < 10
    ]

    if not possible_snapshot_overs:
        return None

    snapshot = np.random.choice(possible_snapshot_overs)
    snap_over_int = snapshot["over"]
    # Add fractional ball in over (0 to 5 balls)
    snap_ball = np.random.choice([0, 1, 2, 3, 4, 5])
    snap_over = round(snap_over_int + snap_ball * 0.1, 1)

    # Calculate runs and wickets in last 5 overs
    start_over_idx = max(0, snap_over_int - 5)
    runs_in_window = sum(o["over_runs"] for o in overs_data[start_over_idx:snap_over_int])
    # Approximate extra balls in current over
    runs_last_5 = runs_in_window + int(snap_ball * 1.3)
    snap_runs = snapshot["runs_cum"] + int(snap_ball * 1.3)
    snap_wkts = snapshot["wkts_cum"]

    # Calculate wickets in last 5 overs
    wkts_start = overs_data[start_over_idx]["wkts_cum"] if start_over_idx > 0 else 0
    wkts_last_5 = max(0, snap_wkts - wkts_start)

    # Decimal overs for rate calculations
    overs_decimal = snap_over_int + (snap_ball / 6.0)
    current_run_rate = round(snap_runs / overs_decimal, 2)

    return {
        "format": format_type,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "current_score": snap_runs,
        "wickets": snap_wkts,
        "overs": snap_over,
        "runs_last_5": runs_last_5,
        "wickets_last_5": wkts_last_5,
        "current_run_rate": current_run_rate,
        "final_score": final_score
    }


def generate_dataset(n_samples_t20=15000, n_samples_odi=12000):
    """Generates T20 and ODI datasets and saves them as CSV files."""
    print(f"Generating {n_samples_t20} T20 match states...")
    t20_records = []
    while len(t20_records) < n_samples_t20:
        row = simulate_innings("T20")
        if row:
            t20_records.append(row)

    df_t20 = pd.DataFrame(t20_records)
    t20_file = os.path.join(DATA_DIR, "t20_matches.csv")
    df_t20.to_csv(t20_file, index=False)
    print(f"Saved T20 dataset to {t20_file} with shape {df_t20.shape}")
    print(f"T20 Final Score Summary:\n{df_t20['final_score'].describe()}")

    print(f"\nGenerating {n_samples_odi} ODI match states...")
    odi_records = []
    while len(odi_records) < n_samples_odi:
        row = simulate_innings("ODI")
        if row:
            odi_records.append(row)

    df_odi = pd.DataFrame(odi_records)
    odi_file = os.path.join(DATA_DIR, "odi_matches.csv")
    df_odi.to_csv(odi_file, index=False)
    print(f"Saved ODI dataset to {odi_file} with shape {df_odi.shape}")
    print(f"ODI Final Score Summary:\n{df_odi['final_score'].describe()}")


if __name__ == "__main__":
    generate_dataset()

