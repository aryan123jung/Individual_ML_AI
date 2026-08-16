from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from app.core import loaders


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum() or ch.isspace()).strip()


def _name_aliases(value: str) -> set[str]:
    normalized = _normalize_name(value)
    parts = normalized.split()
    aliases = {normalized.replace(" ", "")}
    if parts:
        aliases.add("".join(parts))
    if len(parts) >= 2:
        aliases.add(f"{parts[0][0]}{parts[-1]}")
        aliases.add(f"{parts[0][0]} {' '.join(parts[1:])}".replace(" ", ""))
    return {alias for alias in aliases if alias}


def _json_safe_record(record):
    safe = {}
    for key, value in record.items():
        if pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value.item() if hasattr(value, "item") else value
    return safe


def _records_by_player(df: pd.DataFrame, player_name: str, column: str = "player"):
    if df.empty or column not in df.columns:
        return []
    exact_matches = df[df[column].astype(str).str.lower() == player_name.lower()]
    if not exact_matches.empty:
        matches = exact_matches
    else:
        player_aliases = _name_aliases(player_name)
        aliases = df[column].astype(str).map(_name_aliases)
        matches = df[aliases.map(lambda candidate_aliases: bool(player_aliases & candidate_aliases))]
    return [_json_safe_record(row) for row in matches.to_dict(orient="records")]


def _resolve_player_matches(df: pd.DataFrame, player_name: str, column: str = "player", preferred_team: str | None = None):
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    exact_matches = df[df[column].astype(str).str.lower() == player_name.lower()].copy()
    if not exact_matches.empty:
        return exact_matches

    player_aliases = _name_aliases(player_name)
    alias_matches = df[df[column].astype(str).map(lambda value: bool(player_aliases & _name_aliases(value)))].copy()
    if alias_matches.empty:
        return alias_matches
    if preferred_team and "team" in alias_matches.columns:
        team_matches = alias_matches[alias_matches["team"].astype(str).str.lower() == preferred_team.lower()].copy()
        if not team_matches.empty:
            return team_matches
    return alias_matches


def _first_record_by_player(df: pd.DataFrame, player_name: str, column: str):
    rows = _records_by_player(df, player_name, column=column)
    return rows[0] if rows else None


def _exact_record_by_player(df: pd.DataFrame, player_name: str, column: str = "player"):
    if df.empty or column not in df.columns:
        return None
    matches = df[df[column].astype(str).str.lower() == player_name.lower()]
    if matches.empty:
        return None
    return _json_safe_record(matches.iloc[0].to_dict())


def _exact_records_by_player(df: pd.DataFrame, player_name: str, column: str = "player"):
    if df.empty or column not in df.columns:
        return []
    matches = df[df[column].astype(str).str.lower() == player_name.lower()].copy()
    if matches.empty:
        return []
    sort_col = "season_start_year" if "season_start_year" in matches.columns else None
    if sort_col:
        matches = matches.sort_values(sort_col)
    return [_json_safe_record(row) for row in matches.to_dict(orient="records")]


def _resolved_records_by_player(df: pd.DataFrame, player_name: str, column: str = "player", preferred_team: str | None = None):
    matches = _resolve_player_matches(df, player_name, column=column, preferred_team=preferred_team)
    if matches.empty:
        return []
    sort_col = "season_start_year" if "season_start_year" in matches.columns else None
    if sort_col:
        matches = matches.sort_values(sort_col)
    return [_json_safe_record(row) for row in matches.to_dict(orient="records")]


def _resolved_record_by_player(df: pd.DataFrame, player_name: str, column: str = "player", preferred_team: str | None = None):
    rows = _resolved_records_by_player(df, player_name, column=column, preferred_team=preferred_team)
    return rows[-1] if rows else None


def _aggregate_batting_history_from_raw(player_name: str, preferred_team: str | None = None):
    raw = _resolve_player_matches(loaders.load_ipl_batting_raw(), player_name, column="player", preferred_team=preferred_team)
    if raw.empty:
        return []
    grouped = (
        raw.groupby(["player", "team", "season_start_year"], as_index=False)
        .agg(
            total_runs=("runs", "sum"),
            total_matches=("match_file", "nunique"),
            total_balls_faced=("balls_faced", "sum"),
            total_fours=("fours", "sum"),
            total_sixes=("sixes", "sum"),
            total_dismissed=("dismissed", "sum"),
            pp_runs=("pp_runs", "sum"),
            pp_balls=("pp_balls", "sum"),
            mid_runs=("mid_runs", "sum"),
            mid_balls=("mid_balls", "sum"),
            death_runs=("death_runs", "sum"),
            death_balls=("death_balls", "sum"),
            dot_balls=("dot_balls", "sum"),
        )
        .sort_values("season_start_year")
    )
    grouped["avg_runs"] = (grouped["total_runs"] / grouped["total_matches"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["strike_rate"] = (grouped["total_runs"] * 100 / grouped["total_balls_faced"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["pp_strike_rate"] = (grouped["pp_runs"] * 100 / grouped["pp_balls"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["mid_strike_rate"] = (grouped["mid_runs"] * 100 / grouped["mid_balls"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["death_strike_rate"] = (grouped["death_runs"] * 100 / grouped["death_balls"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["sample_reliability_score"] = (grouped["total_matches"].clip(upper=12) / 12).round(3)
    return [_json_safe_record(row) for row in grouped.to_dict(orient="records")]


def _aggregate_bowling_history_from_raw(player_name: str, preferred_team: str | None = None):
    raw = _resolve_player_matches(loaders.load_ipl_bowling_raw(), player_name, column="player", preferred_team=preferred_team)
    if raw.empty:
        return []
    grouped = (
        raw.groupby(["player", "team", "season_start_year"], as_index=False)
        .agg(
            total_matches_bowled=("match_file", "nunique"),
            total_wickets=("wickets", "sum"),
            total_balls_bowled=("balls_bowled", "sum"),
            total_runs_conceded=("runs_conceded", "sum"),
            dot_balls=("dot_balls", "sum"),
            pp_runs=("pp_runs", "sum"),
            pp_balls=("pp_balls", "sum"),
            mid_runs=("mid_runs", "sum"),
            mid_balls=("mid_balls", "sum"),
            death_runs=("death_runs", "sum"),
            death_balls=("death_balls", "sum"),
        )
        .sort_values("season_start_year")
    )
    grouped["avg_wickets_per_match"] = (grouped["total_wickets"] / grouped["total_matches_bowled"].replace(0, pd.NA)).fillna(0).round(3)
    grouped["economy_rate"] = (grouped["total_runs_conceded"] * 6 / grouped["total_balls_bowled"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["pp_economy"] = (grouped["pp_runs"] * 6 / grouped["pp_balls"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["mid_economy"] = (grouped["mid_runs"] * 6 / grouped["mid_balls"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["death_economy"] = (grouped["death_runs"] * 6 / grouped["death_balls"].replace(0, pd.NA)).fillna(0).round(2)
    grouped["bowling_dot_rate"] = (grouped["dot_balls"] / grouped["total_balls_bowled"].replace(0, pd.NA)).fillna(0).round(3)
    grouped["sample_reliability_score"] = (grouped["total_matches_bowled"].clip(upper=12) / 12).round(3)
    return [_json_safe_record(row) for row in grouped.to_dict(orient="records")]


def _pick_primary_profile(player_payload: dict):
    domestic = player_payload.get("domestic_profiles", {}) or {}
    ipl = player_payload.get("ipl_historical_profiles", {}) or {}
    squad_rows = player_payload.get("current_squad_profile", []) or []

    if squad_rows:
        return "squad", squad_rows[0], "current_squad"
    for domain in ["batting", "bowling", "allrounder"]:
        if ipl.get(domain):
            return domain, ipl[domain], "ipl_historical"
    for domain in ["batting", "bowling", "allrounder"]:
        if domestic.get(domain):
            return domain, domestic[domain], "domestic"
    return "unknown", {}, "unknown"


def _first_non_null(record: dict, keys: list[str], default=0):
    for key in keys:
        value = record.get(key)
        if value is not None and not pd.isna(value):
            return value
    return default


def _coalesce_records(*records: dict):
    merged = {}
    for record in records:
        if not record:
            continue
        for key, value in record.items():
            if key not in merged or merged[key] is None or (isinstance(merged[key], float) and pd.isna(merged[key])):
                if value is not None and not pd.isna(value):
                    merged[key] = value
    return merged


def _proxy_quality_score(role: str, record: dict):
    role_text = str(role or "").lower()
    if "bowler" in role_text:
        wickets = float(_first_non_null(record, ["avg_wickets_per_match"], 0) or 0)
        economy = float(_first_non_null(record, ["economy_rate", "economy", "bowling_economy"], 12) or 12)
        return max(0.0, min(100.0, wickets * 28 + max(0, 11 - economy) * 8))
    strike_rate = float(_first_non_null(record, ["strike_rate", "batting_strike_rate"], 0) or 0)
    avg_runs = float(_first_non_null(record, ["avg_runs", "average"], 0) or 0)
    return max(0.0, min(100.0, avg_runs * 2.1 + strike_rate * 0.28))


def _proxy_recent_form_score(record: dict):
    matches = float(_first_non_null(record, ["matches", "total_matches", "total_matches_bowled", "matches_played"], 0) or 0)
    reliability = float(_first_non_null(record, ["sample_reliability_score"], 0) or 0) * 100
    return max(0.0, min(100.0, reliability * 0.65 + min(matches, 14) * 2.5))


def _proxy_ml_fit(role: str, quality_score: float, reliability_score: float):
    role_text = str(role or "").lower()
    role_bonus = 8 if "allround" in role_text else 4 if "bowler" in role_text or "opener" in role_text else 0
    return max(0.0, min(100.0, quality_score * 0.72 + reliability_score * 0.2 + role_bonus))


def _proxy_average_runs(record: dict):
    runs = float(_first_non_null(record, ["total_runs", "runs"], 0) or 0)
    matches = float(_first_non_null(record, ["total_matches", "matches", "matches_played"], 0) or 0)
    if matches <= 0:
        return 0.0
    return round(runs / matches, 2)


def _proxy_reliability_from_matches(record: dict):
    matches = float(_first_non_null(record, ["total_matches", "total_matches_bowled", "matches", "matches_played"], 0) or 0)
    return round(min(matches / 12, 1.0), 3)


def _comparison_model_domain(role: str, batting_record: dict, bowling_record: dict, allrounder_record: dict):
    role_text = str(role or "").lower()
    if "allround" in role_text or "utility" in role_text:
        return "allrounder"
    if "bowler" in role_text:
        return "bowling"
    if batting_record:
        return "batting"
    if bowling_record:
        return "bowling"
    if allrounder_record:
        return "allrounder"
    return "batting"


def _model_feature_columns(domain: str) -> list[str]:
    model = loaders.load_suitability_model(domain)
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None or not getattr(preprocessor, "transformers_", None):
        return []
    for _, _, columns in preprocessor.transformers_:
        if columns is not None:
            return list(columns)
    return []


def _predict_model_score(domain: str, record: dict):
    if not record:
        return None
    feature_columns = _model_feature_columns(domain)
    if not feature_columns:
        return None
    row = {column: record.get(column) for column in feature_columns}
    row_df = pd.DataFrame([row])
    model = loaders.load_suitability_model(domain)
    probability = float(model.predict_proba(row_df)[0][1])
    return round(probability * 100, 2)


def _comparison_summary(player_payload: dict):
    domain, record, source = _pick_primary_profile(player_payload)
    squad_row = (player_payload.get("current_squad_profile") or [None])[0] or {}
    recommendation_row = (player_payload.get("recommendations") or [None])[0] or {}
    similarity_row = (player_payload.get("similarity_matches") or [None])[0] or {}
    player_name = player_payload.get("player")
    preferred_team = squad_row.get("team") if squad_row else None
    exact_ipl_batting = _resolved_record_by_player(loaders.load_ipl_batting_features(), player_name, column="player", preferred_team=preferred_team)
    exact_ipl_bowling = _resolved_record_by_player(loaders.load_ipl_bowling_features(), player_name, column="player", preferred_team=preferred_team)
    exact_ipl_allrounder = _resolved_record_by_player(loaders.load_ipl_allrounder_features(), player_name, column="player", preferred_team=preferred_team)
    exact_smat_batting = _resolved_record_by_player(loaders.load_smat_batting_features(), player_name, column="player")
    exact_smat_bowling = _resolved_record_by_player(loaders.load_smat_bowling_features(), player_name, column="player")
    exact_smat_allrounder = _resolved_record_by_player(loaders.load_smat_allrounder_features(), player_name, column="player")
    domestic_profiles = player_payload.get("domestic_profiles", {}) or {}
    ipl_profiles = player_payload.get("ipl_historical_profiles", {}) or {}
    if squad_row:
        batting_record = _coalesce_records(
            squad_row,
            exact_ipl_batting or {},
            exact_ipl_allrounder or {},
        )
        bowling_record = _coalesce_records(
            squad_row,
            exact_ipl_bowling or {},
            exact_ipl_allrounder or {},
        )
        allrounder_record = _coalesce_records(
            squad_row,
            exact_ipl_allrounder or {},
            exact_ipl_batting or {},
            exact_ipl_bowling or {},
        )
        merged_record = _coalesce_records(
            squad_row,
            exact_ipl_batting or {},
            exact_ipl_bowling or {},
            exact_ipl_allrounder or {},
            record,
        )
    else:
        batting_record = _coalesce_records(
            exact_ipl_batting or {},
            exact_smat_batting or {},
            ipl_profiles.get("batting") or {},
            domestic_profiles.get("batting") or {},
        )
        bowling_record = _coalesce_records(
            exact_ipl_bowling or {},
            exact_smat_bowling or {},
            ipl_profiles.get("bowling") or {},
            domestic_profiles.get("bowling") or {},
        )
        allrounder_record = _coalesce_records(
            exact_ipl_allrounder or {},
            exact_smat_allrounder or {},
            ipl_profiles.get("allrounder") or {},
            domestic_profiles.get("allrounder") or {},
        )
        merged_record = _coalesce_records(
            exact_ipl_batting or {},
            exact_ipl_bowling or {},
            exact_ipl_allrounder or {},
            exact_smat_batting or {},
            exact_smat_bowling or {},
            exact_smat_allrounder or {},
            record,
        )

    inferred_role = _first_non_null(
        merged_record,
        ["batting_role", "bowling_role", "allrounder_role"],
        _first_non_null(recommendation_row, ["target_role"], squad_row.get("primary_role")),
    )
    sample_reliability_score = float(
        _first_non_null(merged_record, ["sample_reliability_score"], _proxy_reliability_from_matches(merged_record)) or 0
    )
    reliability_score = sample_reliability_score * 100
    model_domain = _comparison_model_domain(inferred_role, batting_record, bowling_record, allrounder_record)
    model_meta = loaders.get_production_model_metadata(model_domain)
    model_record = {
        "batting": _coalesce_records(batting_record, merged_record),
        "bowling": _coalesce_records(bowling_record, merged_record),
        "allrounder": _coalesce_records(allrounder_record, merged_record),
    }.get(model_domain, merged_record)
    direct_model_score = _predict_model_score(model_domain, model_record)

    quality_score = float(_first_non_null(recommendation_row, ["quality_score"], None) or 0)
    if quality_score == 0:
        quality_score = float(direct_model_score or 0)
    if quality_score == 0:
        quality_score = _proxy_quality_score(inferred_role, merged_record)

    recent_form_score = float(_first_non_null(recommendation_row, ["recent_form_score"], None) or 0)
    if recent_form_score == 0:
        recent_form_score = _proxy_recent_form_score(squad_row or merged_record)

    ml_fit = float(_first_non_null(recommendation_row, ["ml_suitability_score"], None) or 0)
    if ml_fit == 0 and direct_model_score is not None:
        ml_fit = float(direct_model_score)
    if ml_fit == 0:
        ml_fit = _proxy_ml_fit(inferred_role, quality_score, reliability_score)

    summary = {
        "player": player_payload.get("player"),
        "profile_source": source,
        "model_domain": model_domain,
        "comparison_domain": domain,
        "team": _first_non_null(merged_record, ["team"], squad_row.get("team")),
        "role": inferred_role,
        "matches": _first_non_null(merged_record, ["total_matches", "total_matches_bowled", "matches"], squad_row.get("matches_played", 0)),
        "runs": _first_non_null(merged_record, ["total_runs", "runs"], squad_row.get("runs", 0)),
        "strike_rate": _first_non_null(merged_record, ["strike_rate", "batting_strike_rate"], squad_row.get("batting_strike_rate", 0)),
        "wickets": _first_non_null(merged_record, ["total_wickets", "wickets"], squad_row.get("wickets", 0)),
        "economy": _first_non_null(merged_record, ["economy_rate", "economy", "bowling_economy"], squad_row.get("bowling_economy", None)),
        "avg_runs": _first_non_null(batting_record, ["avg_runs", "average"], _proxy_average_runs(batting_record)),
        "avg_wickets_per_match": _first_non_null(bowling_record, ["avg_wickets_per_match"], 0),
        "sample_reliability_score": sample_reliability_score,
        "ml_suitability_score": round(ml_fit, 2),
        "recent_form_score": round(recent_form_score, 2),
        "quality_score": round(quality_score, 2),
        "similarity_score": _first_non_null(recommendation_row, ["similarity_score"], _first_non_null(similarity_row, ["similarity_score"], 0)),
        "closest_ipl_benchmark": _first_non_null(recommendation_row, ["closest_ipl_benchmark"], _first_non_null(similarity_row, ["ipl_player"], None)),
        "model_family": model_meta["model_family"],
        "model_name": model_meta["model_name"],
        "model_artifact": model_meta["artifact"],
    }

    radar_metrics = {
        "batting_impact": float(max(summary["quality_score"], float(summary["avg_runs"] or 0) * 2.2)),
        "scoring_speed": float(summary["strike_rate"] or 0),
        "bowling_impact": float((summary["avg_wickets_per_match"] or 0) * 25),
        "control": float(max(0, 12 - float(summary["economy"] or 12)) * 10),
        "reliability": float(reliability_score),
        "ml_fit": float(summary["ml_suitability_score"] or 0),
    }
    summary["radar_metrics"] = radar_metrics
    return _json_safe_record(summary)


def _player_role_benchmarks(player_payload: dict):
    benchmarks = loaders.load_role_benchmarks()
    if benchmarks.empty:
        return {}
    result = {}
    domestic = player_payload.get("domestic_profiles", {}) or {}
    ipl = player_payload.get("ipl_historical_profiles", {}) or {}
    merged = _coalesce_records(
        ipl.get("batting") or {},
        ipl.get("bowling") or {},
        ipl.get("allrounder") or {},
        domestic.get("batting") or {},
        domestic.get("bowling") or {},
        domestic.get("allrounder") or {},
    )

    role_map = [
        ("batting", merged.get("batting_role")),
        ("bowling", merged.get("bowling_role")),
        ("allrounder", merged.get("allrounder_role")),
    ]
    for benchmark_type, role in role_map:
        if not role:
            continue
        match = benchmarks[
            (benchmarks["benchmark_type"].astype(str).str.lower() == benchmark_type)
            & (benchmarks["role"].astype(str).str.lower() == str(role).lower())
        ]
        if not match.empty:
            result[benchmark_type] = _json_safe_record(match.iloc[0].to_dict())
    return result


def get_player(player_name: str):
    loaders.clear_caches()
    recommendations = loaders.load_recommendations()
    current_squad = loaders.load_current_squad()
    similarity = loaders.load_similarity()
    player_master = loaders.load_player_master()

    rec_rows = _records_by_player(recommendations, player_name, column="domestic_player")
    squad_player_column = "player_name" if "player_name" in current_squad.columns else "player"
    squad_rows = _records_by_player(current_squad, player_name, column=squad_player_column)
    preferred_team = squad_rows[0].get("team") if squad_rows else None
    similarity_player_column = "domestic_player" if "domestic_player" in similarity.columns else "smat_player"
    similarity_rows = _records_by_player(similarity, player_name, column=similarity_player_column)
    player_master_row = _first_record_by_player(player_master, player_name, column="player_name")

    domestic_profiles = {
        "batting": _resolved_record_by_player(loaders.load_smat_batting_features(), player_name, column="player"),
        "bowling": _resolved_record_by_player(loaders.load_smat_bowling_features(), player_name, column="player"),
        "allrounder": _resolved_record_by_player(loaders.load_smat_allrounder_features(), player_name, column="player"),
    }
    domestic_history = {
        "batting": _resolved_records_by_player(loaders.load_smat_batting_features(), player_name, column="player"),
        "bowling": _resolved_records_by_player(loaders.load_smat_bowling_features(), player_name, column="player"),
        "allrounder": _resolved_records_by_player(loaders.load_smat_allrounder_features(), player_name, column="player"),
    }
    ipl_historical_profiles = {
        "batting": _resolved_record_by_player(loaders.load_ipl_batting_features(), player_name, column="player", preferred_team=preferred_team),
        "bowling": _resolved_record_by_player(loaders.load_ipl_bowling_features(), player_name, column="player", preferred_team=preferred_team),
        "allrounder": _resolved_record_by_player(loaders.load_ipl_allrounder_features(), player_name, column="player", preferred_team=preferred_team),
    }
    ipl_historical_history = {
        "batting": _resolved_records_by_player(loaders.load_ipl_batting_features(), player_name, column="player", preferred_team=preferred_team),
        "bowling": _resolved_records_by_player(loaders.load_ipl_bowling_features(), player_name, column="player", preferred_team=preferred_team),
        "allrounder": _resolved_records_by_player(loaders.load_ipl_allrounder_features(), player_name, column="player", preferred_team=preferred_team),
    }
    if len(ipl_historical_history["batting"]) <= 1:
        raw_batting_history = _aggregate_batting_history_from_raw(player_name, preferred_team=preferred_team)
        if raw_batting_history:
            ipl_historical_history["batting"] = raw_batting_history
            ipl_historical_profiles["batting"] = raw_batting_history[-1]
    if len(ipl_historical_history["bowling"]) <= 1:
        raw_bowling_history = _aggregate_bowling_history_from_raw(player_name, preferred_team=preferred_team)
        if raw_bowling_history:
            ipl_historical_history["bowling"] = raw_bowling_history
            ipl_historical_profiles["bowling"] = raw_bowling_history[-1]

    has_any_data = any(
        [
            rec_rows,
            squad_rows,
            similarity_rows,
            player_master_row,
            domestic_profiles["batting"],
            domestic_profiles["bowling"],
            domestic_profiles["allrounder"],
            domestic_history["batting"],
            domestic_history["bowling"],
            domestic_history["allrounder"],
            ipl_historical_profiles["batting"],
            ipl_historical_profiles["bowling"],
            ipl_historical_profiles["allrounder"],
            ipl_historical_history["batting"],
            ipl_historical_history["bowling"],
            ipl_historical_history["allrounder"],
        ]
    )
    if not has_any_data:
        raise HTTPException(status_code=404, detail=f"Player not found: {player_name}")

    payload = {
        "player": player_name,
        "player_master": player_master_row,
        "recommendations": rec_rows,
        "current_squad_profile": squad_rows,
        "similarity_matches": similarity_rows,
        "domestic_profiles": domestic_profiles,
        "domestic_history": domestic_history,
        "ipl_historical_profiles": ipl_historical_profiles,
        "ipl_historical_history": ipl_historical_history,
    }
    payload["role_benchmarks"] = _player_role_benchmarks(payload)
    return payload


def compare_players(player_a: str, player_b: str):
    payload_a = get_player(player_a)
    payload_b = get_player(player_b)
    return {
        "player_a": payload_a,
        "player_b": payload_b,
        "summary": {
            "player_a": _comparison_summary(payload_a),
            "player_b": _comparison_summary(payload_b),
        },
    }
