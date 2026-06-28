import pandas as pd

from recommendation_engine.config import (
    ALLROUNDER_ROLE_TARGETS,
    BATTING_ROLE_TARGETS,
    BOWLING_ROLE_TARGETS,
    LATEST_IPL_SEASON,
)


def build_latest_ipl_rosters(ipl_current_squad=None, ipl_batting_raw=None, ipl_bowling_raw=None):
    if ipl_current_squad is not None and not ipl_current_squad.empty:
        roster = ipl_current_squad.copy()
        player_col = "player_name" if "player_name" in roster.columns else "player"
        team_col = "team" if "team" in roster.columns else "team_name"
        return (
            roster[[player_col, team_col]]
            .rename(columns={player_col: "player", team_col: "team"})
            .dropna(subset=["player", "team"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

    if ipl_batting_raw is None or ipl_bowling_raw is None:
        return pd.DataFrame(columns=["player", "team"])

    latest_bat = ipl_batting_raw[ipl_batting_raw["season_start_year"] == LATEST_IPL_SEASON]
    latest_bowl = ipl_bowling_raw[ipl_bowling_raw["season_start_year"] == LATEST_IPL_SEASON]
    return pd.concat(
        [
            latest_bat[["player", "team"]],
            latest_bowl[["player", "team"]],
        ],
        ignore_index=True,
    ).drop_duplicates()


def build_squad_gaps(roster, ipl_batting, ipl_bowling, ipl_allrounder):
    rows = []

    def add_gap_rows(team, gap_type, role_counts, targets):
        for role, target in targets.items():
            current = int(role_counts.get(role, 0))
            deficit = max(target - current, 0)
            rows.append(
                {
                    "team": team,
                    "gap_type": gap_type,
                    "role": role,
                    "current_count": current,
                    "target_count": target,
                    "deficit": deficit,
                    "gap_severity": round(deficit / target, 3) if target else 0,
                }
            )

    for team, team_roster in roster.groupby("team"):
        players = set(team_roster["player"])
        batting_counts = (
            ipl_batting[ipl_batting["player"].isin(players)]["batting_role"].value_counts().to_dict()
        )
        bowling_counts = (
            ipl_bowling[ipl_bowling["player"].isin(players)]["bowling_role"].value_counts().to_dict()
        )
        allrounder_counts = (
            ipl_allrounder[ipl_allrounder["player"].isin(players)]["allrounder_role"].value_counts().to_dict()
        )

        add_gap_rows(team, "batting", batting_counts, BATTING_ROLE_TARGETS)
        add_gap_rows(team, "bowling", bowling_counts, BOWLING_ROLE_TARGETS)
        add_gap_rows(team, "allrounder", allrounder_counts, ALLROUNDER_ROLE_TARGETS)

    return pd.DataFrame(rows).sort_values(["team", "gap_type", "deficit"], ascending=[True, True, False])
