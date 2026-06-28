import pandas as pd


def build_role_benchmarks(ipl_batting, ipl_bowling, ipl_allrounder):
    rows = []

    for role, group in ipl_batting[ipl_batting["eligible_for_model"]].groupby("batting_role"):
        rows.append(
            {
                "benchmark_type": "batting",
                "role": role,
                "players": len(group),
                "avg_runs_median": group["avg_runs"].median(),
                "strike_rate_median": group["strike_rate"].median(),
                "boundary_rate_median": group["boundary_rate"].median(),
                "sample_reliability_median": group["sample_reliability_score"].median(),
            }
        )

    for role, group in ipl_bowling[ipl_bowling["eligible_for_model"]].groupby("bowling_role"):
        rows.append(
            {
                "benchmark_type": "bowling",
                "role": role,
                "players": len(group),
                "economy_rate_median": group["economy_rate"].median(),
                "avg_wickets_per_match_median": group["avg_wickets_per_match"].median(),
                "bowling_dot_rate_median": group["bowling_dot_rate"].median(),
                "sample_reliability_median": group["sample_reliability_score"].median(),
            }
        )

    for role, group in ipl_allrounder.groupby("allrounder_role"):
        rows.append(
            {
                "benchmark_type": "allrounder",
                "role": role,
                "players": len(group),
                "allrounder_index_median": group["allrounder_index"].median(),
                "batting_strength_median": group["batting_strength_score"].median(),
                "bowling_strength_median": group["bowling_strength_score"].median(),
            }
        )

    return pd.DataFrame(rows)
