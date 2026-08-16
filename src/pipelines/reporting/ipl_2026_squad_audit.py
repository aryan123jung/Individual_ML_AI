from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipelines.engineering import ipl_current_2026 as mod

REPORT_DIR = BASE_DIR / "reports" / "data_quality"
OUTPUT_PATH = REPORT_DIR / "ipl_2026_squad_join_audit.csv"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    team_map = mod.load_team_map()
    player_master = mod.build_player_master()
    squads = mod.build_squads(team_map, player_master)
    deliveries = mod.prepare_deliveries(team_map)
    batting_match, bowling_match = mod.build_match_innings_tables(deliveries)
    batting = mod.aggregate_player_batting(batting_match)
    bowling = mod.aggregate_player_bowling(bowling_match)
    current = mod.build_current_squad_table(squads, batting, bowling, player_master)

    alias_map = mod._build_player_alias_map(player_master)
    master_name_map = mod._build_player_master_name_map(player_master)
    team_player_lookup = mod._build_team_player_lookup_map(
        squads.copy(),
        batting.copy(),
        bowling.copy(),
        alias_map,
        master_name_map,
    )

    current["lookup"] = current.apply(
        lambda row: team_player_lookup.get(
            (mod.normalize_key(row.get("team", "")), mod.clean_text(row.get("player_name", ""))),
            mod._resolve_player_lookup(row.get("player_name", ""), alias_map, master_name_map),
        ),
        axis=1,
    )
    batting["lookup"] = batting["player"].map(lambda x: mod._resolve_player_lookup(x, alias_map, master_name_map))
    bowling["lookup"] = bowling["player"].map(lambda x: mod._resolve_player_lookup(x, alias_map, master_name_map))

    batting_lookup = set(batting["lookup"].dropna())
    bowling_lookup = set(bowling["lookup"].dropna())

    current["has_batting_source"] = current["lookup"].isin(batting_lookup)
    current["has_bowling_source"] = current["lookup"].isin(bowling_lookup)
    current["batting_join_ok"] = (~current["has_batting_source"]) | (
        (current["runs"].fillna(0) > 0) | (current["balls_faced"].fillna(0) > 0)
    )
    current["bowling_join_ok"] = (~current["has_bowling_source"]) | (
        (current["balls_bowled"].fillna(0) > 0)
        | (current["wickets"].fillna(0) > 0)
        | (current["runs_conceded"].fillna(0) > 0)
    )

    current["audit_status"] = "ok"
    current.loc[~current["batting_join_ok"], "audit_status"] = "batting_join_issue"
    current.loc[~current["bowling_join_ok"], "audit_status"] = "bowling_join_issue"
    current.loc[
        (~current["batting_join_ok"]) & (~current["bowling_join_ok"]),
        "audit_status",
    ] = "batting_and_bowling_join_issue"

    current.to_csv(OUTPUT_PATH, index=False)
    print(f"IPL 2026 squad audit written: {OUTPUT_PATH}")
    print(current["audit_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
