import pandas as pd

from recommendation_engine.config import (
    ALLROUNDER_ROLE_TARGETS,
    BATTING_ROLE_TARGETS,
    BOWLING_ROLE_TARGETS,
    LATEST_IPL_SEASON,
)


def _clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_number(value, default=0.0):
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_role_lookup(df, role_column):
    if df is None or df.empty or role_column not in df.columns:
        return {}, {}

    working = df[["player", "team", "season_start_year", role_column]].copy()
    working = working.dropna(subset=["player", role_column])
    if working.empty:
        return {}, {}

    working["player"] = working["player"].astype(str).str.strip()
    working["team"] = working["team"].astype(str).str.strip()
    working["season_start_year"] = pd.to_numeric(working["season_start_year"], errors="coerce").fillna(0)
    working = working.sort_values(["player", "team", "season_start_year"])

    exact = (
        working.drop_duplicates(subset=["player", "team"], keep="last")
        .set_index(["player", "team"])[role_column]
        .to_dict()
    )
    player_only = (
        working.drop_duplicates(subset=["player"], keep="last")
        .set_index("player")[role_column]
        .to_dict()
    )
    return exact, player_only


def _lookup_role(player, team, exact_lookup, player_lookup):
    player = _clean_text(player)
    team = _clean_text(team)
    if not player:
        return None
    if (player, team) in exact_lookup:
        return exact_lookup[(player, team)]
    if player in player_lookup:
        return player_lookup[player]
    return None


def _infer_batting_role(row):
    primary_role = _clean_text(row.get("primary_role")).lower()
    batting_innings = _safe_number(row.get("batting_innings"))
    runs = _safe_number(row.get("runs"))
    strike_rate = _safe_number(row.get("batting_strike_rate"))
    average = _safe_number(row.get("batting_average"))
    fours = _safe_number(row.get("fours"))
    sixes = _safe_number(row.get("sixes"))

    if primary_role == "bowler" and batting_innings <= 1 and runs < 30:
        return None
    if batting_innings <= 0 and "batter" not in primary_role and "keeper" not in primary_role and "rounder" not in primary_role:
        return None

    boundary_heavy = sixes >= max(fours * 0.6, 3)
    if strike_rate >= 155 and (runs >= 220 or average >= 28):
        return "aggressive_opener"
    if average >= 32 and strike_rate <= 140:
        return "anchor_opener"
    if average >= 28 and strike_rate <= 145:
        return "top_order_anchor"
    if average >= 24 and strike_rate >= 140:
        return "top_order_aggressor"
    if strike_rate >= 145 and (boundary_heavy or primary_role == "all rounder"):
        return "finisher"
    if strike_rate >= 135 or runs >= 120:
        return "middle_order"
    if batting_innings > 0:
        return "lower_order_hitter"
    return None


def _infer_bowling_role(row):
    primary_role = _clean_text(row.get("primary_role")).lower()
    bowling_innings = _safe_number(row.get("bowling_innings"))
    wickets = _safe_number(row.get("wickets"))
    economy = _safe_number(row.get("bowling_economy"), default=99)
    bowling_sr = _safe_number(row.get("bowling_strike_rate"), default=99)
    matches = max(_safe_number(row.get("matches_played")), 1)
    wickets_per_match = wickets / matches

    if bowling_innings <= 0 and "bowler" not in primary_role and "rounder" not in primary_role:
        return None

    if economy <= 8.2 and wickets_per_match >= 1.1 and bowling_sr <= 18:
        return "all_phases_bowler"
    if economy <= 8.4 and bowling_sr <= 20:
        return "powerplay_bowler"
    if economy <= 8.8 and wickets_per_match >= 0.8:
        return "middle_overs_bowler"
    if wickets_per_match >= 0.9 or economy >= 9.0:
        return "death_bowler"
    if bowling_innings > 0:
        return "middle_overs_bowler"
    return None


def _infer_allrounder_role(row):
    primary_role = _clean_text(row.get("primary_role")).lower()
    if "rounder" not in primary_role:
        return None

    runs = _safe_number(row.get("runs"))
    strike_rate = _safe_number(row.get("batting_strike_rate"))
    wickets = _safe_number(row.get("wickets"))
    economy = _safe_number(row.get("bowling_economy"), default=12)
    batting_score = runs * 0.18 + strike_rate * 0.35
    bowling_score = wickets * 6.0 + max(0, 10 - economy) * 8.0

    if batting_score >= bowling_score + 10:
        return "batting_allrounder"
    if bowling_score >= batting_score + 10:
        return "bowling_allrounder"
    return "utility_player"


def build_latest_ipl_rosters(ipl_current_squad=None, ipl_batting_raw=None, ipl_bowling_raw=None):
    roster_frames = []

    if ipl_current_squad is not None and not ipl_current_squad.empty:
        roster = ipl_current_squad.copy()
        player_col = "player_name" if "player_name" in roster.columns else "player"
        team_col = "team" if "team" in roster.columns else "team_name"
        keep_columns = [player_col, team_col]
        optional_columns = [
            "primary_role",
            "batting_role",
            "bowling_role",
            "matches_played",
            "batting_innings",
            "runs",
            "balls_faced",
            "fours",
            "sixes",
            "batting_strike_rate",
            "batting_average",
            "bowling_innings",
            "balls_bowled",
            "wickets",
            "bowling_economy",
            "bowling_strike_rate",
        ]
        keep_columns.extend([col for col in optional_columns if col in roster.columns])
        roster_frames.append(
            roster[keep_columns].rename(columns={player_col: "player", team_col: "team"})
        )

    if ipl_batting_raw is None or ipl_bowling_raw is None:
        if roster_frames:
            return (
                pd.concat(roster_frames, ignore_index=True)
                .dropna(subset=["player", "team"])
                .drop_duplicates()
                .reset_index(drop=True)
            )
        return pd.DataFrame(columns=["player", "team"])

    latest_bat = ipl_batting_raw[ipl_batting_raw["season_start_year"] == LATEST_IPL_SEASON]
    latest_bowl = ipl_bowling_raw[ipl_bowling_raw["season_start_year"] == LATEST_IPL_SEASON]
    roster_frames.extend(
        [
            latest_bat[["player", "team"]],
            latest_bowl[["player", "team"]],
        ]
    )
    return (
        pd.concat(roster_frames, ignore_index=True)
        .dropna(subset=["player", "team"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_squad_gaps(roster, ipl_batting, ipl_bowling, ipl_allrounder):
    rows = []
    batting_exact, batting_player = _latest_role_lookup(ipl_batting, "batting_role")
    bowling_exact, bowling_player = _latest_role_lookup(ipl_bowling, "bowling_role")
    allrounder_exact, allrounder_player = _latest_role_lookup(ipl_allrounder, "allrounder_role")

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
        batting_roles = []
        bowling_roles = []
        allrounder_roles = []

        for _, player_row in team_roster.iterrows():
            player = player_row.get("player")
            player_team = player_row.get("team", team)

            batting_role = _lookup_role(player, player_team, batting_exact, batting_player)
            if not batting_role:
                batting_role = _clean_text(player_row.get("batting_role")) or _infer_batting_role(player_row)
            if batting_role:
                batting_roles.append(batting_role)

            bowling_role = _lookup_role(player, player_team, bowling_exact, bowling_player)
            if not bowling_role:
                bowling_role = _clean_text(player_row.get("bowling_role")) or _infer_bowling_role(player_row)
            if bowling_role:
                bowling_roles.append(bowling_role)

            allrounder_role = _lookup_role(player, player_team, allrounder_exact, allrounder_player)
            if not allrounder_role:
                allrounder_role = _infer_allrounder_role(player_row)
            if allrounder_role in {"pure_batter", "pure_bowler"}:
                allrounder_role = None
            if allrounder_role:
                allrounder_roles.append(allrounder_role)

        batting_counts = pd.Series(batting_roles).value_counts().to_dict() if batting_roles else {}
        bowling_counts = pd.Series(bowling_roles).value_counts().to_dict() if bowling_roles else {}
        allrounder_counts = pd.Series(allrounder_roles).value_counts().to_dict() if allrounder_roles else {}

        add_gap_rows(team, "batting", batting_counts, BATTING_ROLE_TARGETS)
        add_gap_rows(team, "bowling", bowling_counts, BOWLING_ROLE_TARGETS)
        add_gap_rows(team, "allrounder", allrounder_counts, ALLROUNDER_ROLE_TARGETS)

    return pd.DataFrame(rows).sort_values(["team", "gap_type", "deficit"], ascending=[True, True, False])
