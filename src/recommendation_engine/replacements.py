import numpy as np
import pandas as pd

RECENT_TREND_SEASON_WINDOW = 3
CURRENT_SEASON_WEIGHT = 0.45
TREND_SCORE_WEIGHT = 0.35
RELIABILITY_WEIGHT = 0.20


def _normalize_name(value):
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum() or ch.isspace()).strip()


def _name_aliases(value):
    normalized = _normalize_name(value)
    parts = normalized.split()
    aliases = {normalized.replace(" ", "")}
    if parts:
        aliases.add("".join(parts))
    if len(parts) >= 2:
        aliases.add(f"{parts[0][0]}{parts[-1]}")
        aliases.add(f"{parts[0][0]}{''.join(parts[1:])}")
        aliases.add("".join(parts[1:]))
    if len(parts) >= 3 and len(parts[0]) == 1:
        aliases.add("".join(parts[1:]))
        aliases.add(f"{parts[1][0]}{parts[-1]}")
    return {alias for alias in aliases if alias}


def _safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_ratio_gap(current, benchmark, reverse=False):
    if pd.isna(current) or pd.isna(benchmark) or benchmark == 0:
        return np.nan
    if reverse:
        return max((current - benchmark) / benchmark, 0)
    return max((benchmark - current) / benchmark, 0)


def _severity(score):
    if score >= 35:
        return "high"
    if score >= 20:
        return "medium"
    return "low"


def _action_level(score):
    if score >= 35:
        return "replace_now"
    if score >= 20:
        return "watchlist"
    if score >= 15:
        return "competition_needed"
    return "stable"


def _mean_score(values):
    cleaned = [value for value in values if not pd.isna(value)]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))


def _reliability_score(matches):
    if pd.isna(matches):
        return 0.0
    return float(min(matches / 10, 1.0) * 100)


def _blended_underperformance_score(current_score, trend_score, reliability_score):
    base_score = (
        current_score * (CURRENT_SEASON_WEIGHT / (CURRENT_SEASON_WEIGHT + TREND_SCORE_WEIGHT))
        + trend_score * (TREND_SCORE_WEIGHT / (CURRENT_SEASON_WEIGHT + TREND_SCORE_WEIGHT))
    )
    evidence_multiplier = 0.55 + 0.45 * (reliability_score / 100)
    return round(base_score * evidence_multiplier, 2)


def _resolve_player_alias_value(player_name, candidates, value_column, preferred_team=None, team_column="team"):
    if candidates is None or candidates.empty or value_column not in candidates.columns:
        return np.nan
    preferred = candidates
    if preferred_team and team_column in preferred.columns:
        team_matches = preferred[preferred[team_column].astype(str).str.lower() == str(preferred_team).lower()].copy()
        if not team_matches.empty:
            preferred = team_matches
    exact = preferred[preferred["player"].astype(str).str.lower() == str(player_name).lower()]
    if not exact.empty:
        return exact.iloc[0].get(value_column, np.nan)
    aliases = _name_aliases(player_name)
    alias_matches = preferred[
        preferred["player"].astype(str).map(lambda value: bool(_name_aliases(value) & aliases))
    ]
    if alias_matches.empty:
        return np.nan
    return alias_matches.iloc[0].get(value_column, np.nan)


def _resolve_historical_player_name(player_name, candidates, preferred_team=None, team_column="team"):
    if candidates is None or candidates.empty or "player" not in candidates.columns:
        return None
    preferred = candidates
    if preferred_team and team_column in preferred.columns:
        team_matches = preferred[preferred[team_column].astype(str).str.lower() == str(preferred_team).lower()].copy()
        if not team_matches.empty:
            preferred = team_matches
    exact = preferred[preferred["player"].astype(str).str.lower() == str(player_name).lower()]
    if not exact.empty:
        return str(exact.iloc[0]["player"])

    aliases = _name_aliases(player_name)
    parts = _normalize_name(player_name).split()
    surname = parts[-1] if parts else ""
    first_initial = parts[0][0] if parts else ""

    scored = []
    for candidate in preferred["player"].dropna().astype(str).unique():
        candidate_parts = _normalize_name(candidate).split()
        candidate_surname = candidate_parts[-1] if candidate_parts else ""
        candidate_initial = candidate_parts[0][0] if candidate_parts else ""
        overlap = len(_name_aliases(candidate) & aliases)
        if overlap <= 0:
            continue
        score = overlap
        if surname and candidate_surname == surname:
            score += 5
        if first_initial and candidate_initial == first_initial:
            score += 3
        if candidate.lower() == player_name.lower():
            score += 10
        scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _resolve_trend_score(player_name, trend_lookup, candidates=None, preferred_team=None):
    exact = trend_lookup.get(player_name)
    if exact is not None:
        return float(exact)
    historical_name = _resolve_historical_player_name(player_name, candidates, preferred_team=preferred_team)
    if historical_name is not None and historical_name in trend_lookup:
        return float(trend_lookup[historical_name])
    return None


def _batting_phase_column(role):
    if role in ["aggressive_opener", "anchor_opener"]:
        return "pp_strike_rate"
    if role in ["top_order_anchor", "top_order_aggressor", "middle_order"]:
        return "mid_strike_rate"
    if role in ["finisher", "lower_order_hitter"]:
        return "death_strike_rate"
    return "strike_rate"


def _bowling_phase_column(role):
    if role == "powerplay_bowler":
        return "pp_economy"
    if role == "middle_overs_bowler":
        return "mid_economy"
    if role == "death_bowler":
        return "death_economy"
    return "economy"


def _infer_batting_role(primary_role, row):
    pp_runs = row.get("pp_runs", 0) or 0
    mid_runs = row.get("mid_runs", 0) or 0
    death_runs = row.get("death_runs", 0) or 0
    total_runs = max(pp_runs + mid_runs + death_runs, 1)
    pp_share = pp_runs / total_runs
    death_share = death_runs / total_runs
    strike_rate = row.get("strike_rate", 0) or 0
    average = row.get("average", 0) or 0
    pp_sr = row.get("pp_strike_rate", np.nan)
    mid_sr = row.get("mid_strike_rate", np.nan)
    death_sr = row.get("death_strike_rate", np.nan)
    role_name = str(primary_role or "").strip().lower()

    if death_share >= 0.35:
        return "lower_order_hitter" if strike_rate >= 155 else "finisher"
    if pp_share >= 0.45 or role_name == "wicket keeper":
        if (pd.notna(pp_sr) and pp_sr >= 145) or strike_rate >= 145:
            return "aggressive_opener"
        return "anchor_opener"
    if average >= 30 and strike_rate < 140:
        return "top_order_anchor"
    if average >= 24 and strike_rate >= 140:
        return "top_order_aggressor"
    if pd.notna(mid_sr) and mid_sr >= 135:
        return "middle_order"
    return "middle_order"


def _infer_bowling_role(row):
    pp_pct = row.get("pp_workload_pct", 0) or 0
    mid_pct = row.get("mid_workload_pct", 0) or 0
    death_pct = row.get("death_workload_pct", 0) or 0
    phase = str(row.get("bowling_phase", "") or "")
    if death_pct >= max(pp_pct, mid_pct, death_pct):
        return "death_bowler"
    if pp_pct >= max(pp_pct, mid_pct, death_pct):
        return "powerplay_bowler"
    if mid_pct >= max(pp_pct, mid_pct, death_pct):
        return "middle_overs_bowler"
    if "death" in phase:
        return "death_bowler"
    if "pp" in phase:
        return "powerplay_bowler"
    return "middle_overs_bowler"


def _infer_allrounder_role(row):
    batting_strength = (
        row.get("average", 0) or 0
    ) * 0.45 + (row.get("strike_rate", 0) or 0) * 0.25 + min((row.get("runs", 0) or 0), 250) * 0.12
    bowling_strength = (
        min((row.get("wickets", 0) or 0), 20) * 4.0
        + max(0, 10 - (row.get("economy", 10) or 10)) * 6.0
    )
    if batting_strength >= bowling_strength + 8:
        return "batting_allrounder"
    if bowling_strength >= batting_strength + 8:
        return "bowling_allrounder"
    return "utility_player"


def _trend_label(current_score, trend_score):
    if trend_score >= 28 and current_score >= 25:
        return "consistent_decline"
    if trend_score >= 22:
        return "multi_season_drop"
    if current_score >= 25:
        return "short_term_dip"
    return "stable_trend"


def _build_batting_trend_lookup(ipl_batting):
    rows = []
    eligible = ipl_batting[ipl_batting["eligible_for_model"]].copy()
    if eligible.empty:
        return {}
    benchmarks = (
        eligible.groupby(["season_start_year", "batting_role"], as_index=False)
        .agg(
            benchmark_avg_runs=("avg_runs", "median"),
            benchmark_strike_rate=("strike_rate", "median"),
            benchmark_pp_strike_rate=("pp_strike_rate", "median"),
            benchmark_mid_strike_rate=("mid_strike_rate", "median"),
            benchmark_death_strike_rate=("death_strike_rate", "median"),
        )
    )
    merged = eligible.merge(benchmarks, on=["season_start_year", "batting_role"], how="left")
    for _, row in merged.iterrows():
        phase_col = _batting_phase_column(row.get("batting_role"))
        score = _mean_score(
            [
                _safe_ratio_gap(row.get("avg_runs"), row.get("benchmark_avg_runs")),
                _safe_ratio_gap(row.get("strike_rate"), row.get("benchmark_strike_rate")),
                _safe_ratio_gap(row.get(phase_col), row.get(f"benchmark_{phase_col}")),
            ]
        )
        rows.append(
            {
                "player": row["player"],
                "season_start_year": row["season_start_year"],
                "trend_component": 0 if pd.isna(score) else float(score * 100),
            }
        )
    trend_df = pd.DataFrame(rows)
    trend_lookup = {}
    for player, group in trend_df.groupby("player"):
        group = group.sort_values("season_start_year", ascending=False).head(RECENT_TREND_SEASON_WINDOW)
        if group.empty:
            continue
        weights = np.linspace(1.0, 0.6, len(group))
        trend_lookup[player] = round(float(np.average(group["trend_component"], weights=weights)), 2)
    return trend_lookup


def _build_bowling_trend_lookup(ipl_bowling):
    rows = []
    eligible = ipl_bowling[ipl_bowling["eligible_for_model"]].copy()
    if eligible.empty:
        return {}
    benchmarks = (
        eligible.groupby(["season_start_year", "bowling_role"], as_index=False)
        .agg(
            benchmark_avg_wickets_per_match=("avg_wickets_per_match", "median"),
            benchmark_economy=("economy_rate", "median"),
            benchmark_pp_economy=("pp_economy", "median"),
            benchmark_mid_economy=("mid_economy", "median"),
            benchmark_death_economy=("death_economy", "median"),
        )
    )
    merged = eligible.merge(benchmarks, on=["season_start_year", "bowling_role"], how="left")
    for _, row in merged.iterrows():
        phase_col = _bowling_phase_column(row.get("bowling_role"))
        score = _mean_score(
            [
                _safe_ratio_gap(
                    row.get("avg_wickets_per_match"),
                    row.get("benchmark_avg_wickets_per_match"),
                ),
                _safe_ratio_gap(row.get("economy_rate"), row.get("benchmark_economy"), reverse=True),
                _safe_ratio_gap(row.get(phase_col), row.get(f"benchmark_{phase_col}"), reverse=True),
            ]
        )
        rows.append(
            {
                "player": row["player"],
                "season_start_year": row["season_start_year"],
                "trend_component": 0 if pd.isna(score) else float(score * 100),
            }
        )
    trend_df = pd.DataFrame(rows)
    trend_lookup = {}
    for player, group in trend_df.groupby("player"):
        group = group.sort_values("season_start_year", ascending=False).head(RECENT_TREND_SEASON_WINDOW)
        if group.empty:
            continue
        weights = np.linspace(1.0, 0.6, len(group))
        trend_lookup[player] = round(float(np.average(group["trend_component"], weights=weights)), 2)
    return trend_lookup


def _build_allrounder_trend_lookup(ipl_allrounder):
    rows = []
    eligible = ipl_allrounder.copy()
    if eligible.empty:
        return {}
    benchmarks = (
        eligible.groupby(["season_start_year", "allrounder_role"], as_index=False)
        .agg(
            benchmark_allrounder_index=("allrounder_index", "median"),
            benchmark_batting_strength=("batting_strength_score", "median"),
            benchmark_bowling_strength=("bowling_strength_score", "median"),
        )
    )
    merged = eligible.merge(benchmarks, on=["season_start_year", "allrounder_role"], how="left")
    for _, row in merged.iterrows():
        score = _mean_score(
            [
                _safe_ratio_gap(row.get("allrounder_index"), row.get("benchmark_allrounder_index")),
                _safe_ratio_gap(row.get("batting_strength_score"), row.get("benchmark_batting_strength")),
                _safe_ratio_gap(row.get("bowling_strength_score"), row.get("benchmark_bowling_strength")),
            ]
        )
        rows.append(
            {
                "player": row["player"],
                "season_start_year": row["season_start_year"],
                "trend_component": 0 if pd.isna(score) else float(score * 100),
            }
        )
    trend_df = pd.DataFrame(rows)
    trend_lookup = {}
    for player, group in trend_df.groupby("player"):
        group = group.sort_values("season_start_year", ascending=False).head(RECENT_TREND_SEASON_WINDOW)
        if group.empty:
            continue
        weights = np.linspace(1.0, 0.6, len(group))
        trend_lookup[player] = round(float(np.average(group["trend_component"], weights=weights)), 2)
    return trend_lookup


def _current_allrounder_frame(ipl_2026_batting, ipl_2026_bowling, ipl_allrounder):
    hist_roles = ipl_allrounder[
        [
            "player",
            "team",
            "allrounder_role",
            "batting_strength_score",
            "bowling_strength_score",
            "allrounder_index",
        ]
    ].drop_duplicates(subset=["player"])
    current = hist_roles.merge(
        ipl_2026_batting[
            [
                "player",
                "matches",
                "runs",
                "average",
                "strike_rate",
                "pp_strike_rate",
                "mid_strike_rate",
                "death_strike_rate",
            ]
        ],
        on="player",
        how="left",
    ).merge(
        ipl_2026_bowling[
            [
                "player",
                "wickets",
                "economy",
                "pp_economy",
                "mid_economy",
                "death_economy",
            ]
        ],
        on="player",
        how="left",
    )
    current["matches"] = current["matches"].fillna(0)
    current["current_batting_strength"] = (
        current["average"].fillna(0) * 0.35
        + current["strike_rate"].fillna(0) * 0.35
        + current["runs"].fillna(0).clip(upper=250) * 0.30 / 2.5
    ).round(2)
    current["current_bowling_strength"] = (
        (100 - current["economy"].fillna(100)).clip(lower=0) * 0.40
        + current["wickets"].fillna(0).clip(upper=20) * 3.5
        + (100 - current["pp_economy"].fillna(current["economy"].fillna(100))).clip(lower=0) * 0.10
    ).round(2)
    current["current_allrounder_index"] = (
        current["current_batting_strength"] * 0.5 + current["current_bowling_strength"] * 0.5
    ).round(2)
    return current


def build_replacement_watchlist(
    ipl_current_squad,
    ipl_2026_batting,
    ipl_2026_bowling,
    ipl_batting,
    ipl_bowling,
    ipl_allrounder,
):
    watch_rows = []
    batting_trend_lookup = _build_batting_trend_lookup(ipl_batting)
    bowling_trend_lookup = _build_bowling_trend_lookup(ipl_bowling)
    allrounder_trend_lookup = _build_allrounder_trend_lookup(ipl_allrounder)
    primary_roles = (
        ipl_current_squad[["team", "player_name", "primary_role"]]
        .drop_duplicates(subset=["team", "player_name"])
        .copy()
    )

    current_batting = ipl_current_squad[["team", "player_name"]].merge(
        ipl_2026_batting,
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_2026"),
    ).merge(
        ipl_batting[
            ["player", "batting_role", "avg_runs", "strike_rate", "pp_strike_rate", "mid_strike_rate", "death_strike_rate"]
        ].drop_duplicates(subset=["player"]),
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_hist"),
    )
    current_batting = current_batting.merge(
        primary_roles,
        on=["team", "player_name"],
        how="left",
    )
    current_batting = current_batting.rename(
        columns={
            "strike_rate": "strike_rate_2026",
            "pp_strike_rate": "pp_strike_rate_2026",
            "mid_strike_rate": "mid_strike_rate_2026",
            "death_strike_rate": "death_strike_rate_2026",
            "team_2026": "current_team_2026",
        }
    )
    current_batting["inferred_batting_role"] = current_batting.apply(
        lambda row: _infer_batting_role(row.get("primary_role"), row), axis=1
    )
    current_batting["batting_role"] = current_batting.apply(
        lambda row: row["batting_role"]
        if pd.notna(row.get("batting_role"))
        else _resolve_player_alias_value(
            row.get("player_name"),
            ipl_batting[["player", "team", "batting_role"]],
            "batting_role",
            preferred_team=row.get("team"),
        ),
        axis=1,
    )
    current_batting["batting_role"] = current_batting["batting_role"].fillna(
        current_batting["inferred_batting_role"]
    )

    batting_benchmarks = (
        ipl_batting[ipl_batting["eligible_for_model"]]
        .groupby("batting_role", as_index=False)
        .agg(
            benchmark_avg_runs=("avg_runs", "median"),
            benchmark_strike_rate=("strike_rate", "median"),
            benchmark_pp_strike_rate=("pp_strike_rate", "median"),
            benchmark_mid_strike_rate=("mid_strike_rate", "median"),
            benchmark_death_strike_rate=("death_strike_rate", "median"),
        )
    )
    current_batting = current_batting.merge(batting_benchmarks, on="batting_role", how="left")

    for _, row in current_batting.iterrows():
        role = row.get("batting_role")
        matches = row.get("matches", 0)
        primary_role = str(row.get("primary_role", "")).strip().lower()
        batting_allowed_roles = {"batter", "wicket keeper", "wicketkeeper", "all rounder"}
        if (
            pd.isna(role)
            or matches < 5
            or role == "tailender"
            or primary_role not in batting_allowed_roles
        ):
            continue
        phase_col = _batting_phase_column(role)
        phase_benchmark_col = f"benchmark_{phase_col}"
        score = _mean_score(
            [
                _safe_ratio_gap(row.get("average"), row.get("benchmark_avg_runs")),
                _safe_ratio_gap(row.get("strike_rate_2026"), row.get("benchmark_strike_rate")),
                _safe_ratio_gap(row.get(f"{phase_col}_2026"), row.get(phase_benchmark_col)),
            ]
        )
        current_score = 0 if pd.isna(score) else round(score * 100, 2)
        trend_score = _resolve_trend_score(
            row["player_name"],
            batting_trend_lookup,
            candidates=ipl_batting[["player", "team"]],
            preferred_team=row.get("team"),
        )
        if trend_score is None:
            trend_score = float(current_score)
        reliability_score = _reliability_score(matches)
        final_score = _blended_underperformance_score(current_score, trend_score, reliability_score)
        if current_score <= 0 and trend_score <= 0:
            continue
        if final_score < 12:
            continue
        watch_rows.append(
            {
                "team": row["team"],
                "player": row["player_name"],
                "replacement_type": "batting",
                "target_role": role,
                "matches_played": int(matches),
                "current_season_score": current_score,
                "multi_season_trend_score": round(trend_score, 2),
                "reliability_score": round(reliability_score, 2),
                "underperformance_score": final_score,
                "trend_label": _trend_label(current_score, trend_score),
                "severity": _severity(final_score),
                "action_level": _action_level(final_score),
                "reason": f"Current batting output below IPL benchmark for {role}",
            }
        )

    current_bowling = ipl_current_squad[["team", "player_name"]].merge(
        ipl_2026_bowling,
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_2026"),
    ).merge(
        ipl_bowling[
            ["player", "bowling_role", "avg_wickets_per_match", "economy_rate", "pp_economy", "mid_economy", "death_economy"]
        ].drop_duplicates(subset=["player"]),
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_hist"),
    )
    current_bowling = current_bowling.rename(
        columns={
            "economy": "economy_2026",
            "pp_economy": "pp_economy_2026",
            "mid_economy": "mid_economy_2026",
            "death_economy": "death_economy_2026",
            "team_2026": "current_team_2026",
        }
    )
    current_bowling["inferred_bowling_role"] = current_bowling.apply(_infer_bowling_role, axis=1)
    current_bowling["bowling_role"] = current_bowling.apply(
        lambda row: row["bowling_role"]
        if pd.notna(row.get("bowling_role"))
        else _resolve_player_alias_value(
            row.get("player_name"),
            ipl_bowling[["player", "team", "bowling_role"]],
            "bowling_role",
            preferred_team=row.get("team"),
        ),
        axis=1,
    )
    current_bowling["bowling_role"] = current_bowling["bowling_role"].fillna(
        current_bowling["inferred_bowling_role"]
    )

    bowling_benchmarks = (
        ipl_bowling[ipl_bowling["eligible_for_model"]]
        .groupby("bowling_role", as_index=False)
        .agg(
            benchmark_avg_wickets_per_match=("avg_wickets_per_match", "median"),
            benchmark_economy=("economy_rate", "median"),
            benchmark_pp_economy=("pp_economy", "median"),
            benchmark_mid_economy=("mid_economy", "median"),
            benchmark_death_economy=("death_economy", "median"),
        )
    )
    current_bowling = current_bowling.merge(bowling_benchmarks, on="bowling_role", how="left")
    current_bowling["current_avg_wickets_per_match"] = (
        current_bowling["wickets"] / current_bowling["matches"].replace(0, np.nan)
    )

    for _, row in current_bowling.iterrows():
        role = row.get("bowling_role")
        matches = row.get("matches", 0)
        if pd.isna(role) or matches < 5 or role == "part_time_bowler":
            continue
        phase_col = _bowling_phase_column(role)
        phase_benchmark_col = f"benchmark_{phase_col}"
        score = _mean_score(
            [
                _safe_ratio_gap(
                    row.get("current_avg_wickets_per_match"),
                    row.get("benchmark_avg_wickets_per_match"),
                ),
                _safe_ratio_gap(row.get("economy_2026"), row.get("benchmark_economy"), reverse=True),
                _safe_ratio_gap(row.get(f"{phase_col}_2026"), row.get(phase_benchmark_col), reverse=True),
            ]
        )
        current_score = 0 if pd.isna(score) else round(score * 100, 2)
        trend_score = _resolve_trend_score(
            row["player_name"],
            bowling_trend_lookup,
            candidates=ipl_bowling[["player", "team"]],
            preferred_team=row.get("team"),
        )
        if trend_score is None:
            trend_score = float(current_score)
        reliability_score = _reliability_score(matches)
        final_score = _blended_underperformance_score(current_score, trend_score, reliability_score)
        if current_score <= 0 and trend_score <= 0:
            continue
        if final_score < 12:
            continue
        watch_rows.append(
            {
                "team": row["team"],
                "player": row["player_name"],
                "replacement_type": "bowling",
                "target_role": role,
                "matches_played": int(matches),
                "current_season_score": current_score,
                "multi_season_trend_score": round(trend_score, 2),
                "reliability_score": round(reliability_score, 2),
                "underperformance_score": final_score,
                "trend_label": _trend_label(current_score, trend_score),
                "severity": _severity(final_score),
                "action_level": _action_level(final_score),
                "reason": f"Current bowling output below IPL benchmark for {role}",
            }
        )

    current_allrounder = _current_allrounder_frame(ipl_2026_batting, ipl_2026_bowling, ipl_allrounder)
    current_allrounder["allrounder_role"] = current_allrounder["allrounder_role"].fillna(
        current_allrounder.apply(_infer_allrounder_role, axis=1)
    )
    allrounder_benchmarks = (
        ipl_allrounder.groupby("allrounder_role", as_index=False)
        .agg(
            benchmark_allrounder_index=("allrounder_index", "median"),
            benchmark_batting_strength=("batting_strength_score", "median"),
            benchmark_bowling_strength=("bowling_strength_score", "median"),
        )
    )
    current_allrounder = current_allrounder.merge(allrounder_benchmarks, on="allrounder_role", how="left")

    for _, row in current_allrounder.iterrows():
        role = row.get("allrounder_role")
        matches = row.get("matches", 0)
        batting_volume = row.get("runs", 0) >= 40 or row.get("average", 0) >= 15
        bowling_volume = row.get("wickets", 0) >= 1 or pd.notna(row.get("economy"))
        if pd.isna(role) or matches < 5 or not batting_volume or not bowling_volume:
            continue
        score = _mean_score(
            [
                _safe_ratio_gap(
                    row.get("current_allrounder_index"),
                    row.get("benchmark_allrounder_index"),
                ),
                _safe_ratio_gap(
                    row.get("current_batting_strength"),
                    row.get("benchmark_batting_strength"),
                ),
                _safe_ratio_gap(
                    row.get("current_bowling_strength"),
                    row.get("benchmark_bowling_strength"),
                ),
            ]
        )
        current_score = 0 if pd.isna(score) else round(score * 100, 2)
        trend_score = _resolve_trend_score(
            row["player"],
            allrounder_trend_lookup,
            candidates=ipl_allrounder[["player", "team"]] if "team" in ipl_allrounder.columns else ipl_allrounder[["player"]],
            preferred_team=row.get("team"),
        )
        if trend_score is None:
            trend_score = float(current_score)
        reliability_score = _reliability_score(matches)
        final_score = _blended_underperformance_score(current_score, trend_score, reliability_score)
        if current_score <= 0 and trend_score <= 0:
            continue
        if final_score < 12:
            continue
        watch_rows.append(
            {
                "team": row["team"],
                "player": row["player"],
                "replacement_type": "allrounder",
                "target_role": role,
                "matches_played": int(matches),
                "current_season_score": current_score,
                "multi_season_trend_score": round(trend_score, 2),
                "reliability_score": round(reliability_score, 2),
                "underperformance_score": final_score,
                "trend_label": _trend_label(current_score, trend_score),
                "severity": _severity(final_score),
                "action_level": _action_level(final_score),
                "reason": f"Current all-round impact below IPL benchmark for {role}",
            }
        )

    watchlist = pd.DataFrame(watch_rows)
    if watchlist.empty:
        return watchlist, watchlist

    watchlist = watchlist.sort_values(
        ["team", "underperformance_score"], ascending=[True, False]
    ).reset_index(drop=True)
    watchlist["replacement_priority_rank"] = watchlist.groupby("team").cumcount() + 1
    return watchlist


def build_replacement_recommendations(watchlist, recommendations, top_n=5):
    if watchlist.empty or recommendations.empty:
        return pd.DataFrame()

    watchlist = watchlist[watchlist["action_level"].isin(["competition_needed", "watchlist", "replace_now"])].copy()
    if watchlist.empty:
        return pd.DataFrame()

    rows = []
    for _, under in watchlist.iterrows():
        exact_matches = recommendations[
            (recommendations["team"] == under["team"])
            & (recommendations["recommendation_type"] == under["replacement_type"])
            & (recommendations["target_role"] == under["target_role"])
        ].copy()
        matches = exact_matches[
            (exact_matches["final_recommendation_score"] >= 58)
            & (exact_matches["role_fit_score"] >= 50)
            & (exact_matches["quality_score"] >= 45)
        ].sort_values("final_recommendation_score", ascending=False).head(top_n)
        role_match_type = "exact_role"
        if matches.empty:
            fallback_matches = recommendations[
                (recommendations["team"] == under["team"])
                & (recommendations["recommendation_type"] == under["replacement_type"])
            ].copy()
            matches = fallback_matches[
                (fallback_matches["final_recommendation_score"] >= 58)
                & (fallback_matches["quality_score"] >= 45)
            ].sort_values("final_recommendation_score", ascending=False).head(top_n)
            role_match_type = "same_domain_fallback"

        for rank, (_, rec) in enumerate(matches.iterrows(), start=1):
            rows.append(
                {
                    "team": under["team"],
                    "underperforming_player": under["player"],
                    "replacement_type": under["replacement_type"],
                    "target_role": under["target_role"],
                    "underperformance_score": under["underperformance_score"],
                    "severity": under["severity"],
                    "action_level": under["action_level"],
                    "replacement_rank": rank,
                    "recommended_player": rec["domestic_player"],
                    "recommended_team": rec["domestic_team"],
                    "replacement_score": rec["final_recommendation_score"],
                    "recent_form_score": rec.get("recent_form_score"),
                    "role_fit_score": rec.get("role_fit_score"),
                    "role_match_type": role_match_type,
                    "closest_ipl_benchmark": rec.get("closest_ipl_benchmark"),
                    "reason": under["reason"],
                }
            )
    return pd.DataFrame(rows)
