from pathlib import Path


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/recommendations")

LATEST_IPL_SEASON = 2026
SMAT_RECENT_SEASON_WINDOW = 2
SMAT_MIN_RECENT_BATTING_MATCHES = 3
SMAT_MIN_RECENT_BOWLING_MATCHES = 3

BATTING_ROLE_TARGETS = {
    "aggressive_opener": 2,
    "anchor_opener": 1,
    "top_order_anchor": 1,
    "top_order_aggressor": 1,
    "middle_order": 2,
    "finisher": 2,
    "lower_order_hitter": 1,
}

BOWLING_ROLE_TARGETS = {
    "powerplay_bowler": 2,
    "middle_overs_bowler": 2,
    "death_bowler": 2,
    "all_phases_bowler": 1,
}

ALLROUNDER_ROLE_TARGETS = {
    "batting_allrounder": 2,
    "bowling_allrounder": 2,
    "utility_player": 2,
}

BATTING_FEATURES = [
    "avg_runs",
    "strike_rate",
    "boundary_rate",
    "six_rate",
    "dot_ball_rate",
    "innings_above_30_pct",
    "pp_strike_rate",
    "mid_strike_rate",
    "death_strike_rate",
    "pp_runs_pct",
    "mid_runs_pct",
    "death_runs_pct",
    "sample_reliability_score",
]

BOWLING_FEATURES = [
    "economy_rate",
    "bowling_average",
    "bowling_strike_rate",
    "avg_wickets_per_match",
    "bowling_dot_rate",
    "two_plus_haul_rate",
    "pp_economy",
    "mid_economy",
    "death_economy",
    "pp_workload_pct",
    "mid_workload_pct",
    "death_workload_pct",
    "sample_reliability_score",
]

ALLROUNDER_FEATURES = [
    "batting_strength_score",
    "bowling_strength_score",
    "allrounder_index",
    "total_runs",
    "strike_rate",
    "total_wickets",
    "avg_wickets_per_match",
    "economy_rate",
]
