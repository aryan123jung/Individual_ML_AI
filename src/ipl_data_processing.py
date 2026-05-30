import json
import os
import pandas as pd
from collections import defaultdict

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_FOLDER = "data/raw/ipl_raw"
OUTPUT_FOLDER = "data/processed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─── HELPER: Determine match phase from over number ──────────────────────────
def get_phase(over_num):
    if over_num <= 5:
        return "powerplay"
    elif over_num <= 14:
        return "middle"
    else:
        return "death"


# ─── MAIN PARSER ─────────────────────────────────────────────────────────────
def parse_all_matches(data_folder):
    batting_records = []
    bowling_records = []

    all_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    print(f"Processing {len(all_files)} IPL match files...")

    for idx, filename in enumerate(all_files):
        filepath = os.path.join(data_folder, filename)

        with open(filepath, "r") as f:
            match = json.load(f)

        info = match["info"]
        match_date = info["dates"][0]
        season = str(info.get("season", "Unknown"))  # IPL season is integer
        venue = info.get("venue", "Unknown")
        city = info.get("city", "Unknown")           # IPL has city field
        teams = info["teams"]
        outcome = info.get("outcome", {})
        winner = outcome.get("winner", "No Result")
        event = info.get("event", {})
        match_number = event.get("match_number", "Unknown")

        # ── Per-match batting accumulator ─────────────────────────────────
        bat_stats = defaultdict(lambda: {
            "runs": 0, "balls_faced": 0,
            "fours": 0, "sixes": 0, "dismissed": 0,
            "pp_runs": 0, "pp_balls": 0,
            "mid_runs": 0, "mid_balls": 0,
            "death_runs": 0, "death_balls": 0,
            "dot_balls": 0
        })

        bowl_stats = defaultdict(lambda: {
            "runs_conceded": 0, "balls_bowled": 0,
            "wickets": 0, "wides": 0, "noballs": 0,
            "dot_balls": 0,
            "pp_runs": 0, "pp_balls": 0,
            "mid_runs": 0, "mid_balls": 0,
            "death_runs": 0, "death_balls": 0
        })
        bat_meta = {}
        bowl_meta = {}

        for inning in match["innings"]:
            batting_team = inning["team"]
            bowling_team = [t for t in teams if t != batting_team][0]

            for over_data in inning["overs"]:
                over_num = over_data["over"]
                phase = get_phase(over_num)

                for delivery in over_data["deliveries"]:
                    batter = delivery["batter"]
                    bowler = delivery["bowler"]
                    batter_runs = delivery["runs"]["batter"]
                    total_runs = delivery["runs"]["total"]
                    extras = delivery.get("extras", {})
                    is_wide = "wides" in extras
                    is_noball = "noballs" in extras

                    bat_meta.setdefault(batter, {
                        "team": batting_team,
                        "opponent": bowling_team,
                    })
                    bowl_meta.setdefault(bowler, {
                        "team": bowling_team,
                        "opponent": batting_team,
                    })

                    # ── Batting stats ──────────────────────────────────────
                    bat = bat_stats[batter]
                    bat["runs"] += batter_runs
                    if not is_wide:
                        bat["balls_faced"] += 1
                    if batter_runs == 4:
                        bat["fours"] += 1
                    if batter_runs == 6:
                        bat["sixes"] += 1
                    if batter_runs == 0 and not is_wide:
                        bat["dot_balls"] += 1

                    # Phase-wise batting
                    if phase == "powerplay":
                        bat["pp_runs"] += batter_runs
                        if not is_wide:
                            bat["pp_balls"] += 1
                    elif phase == "middle":
                        bat["mid_runs"] += batter_runs
                        if not is_wide:
                            bat["mid_balls"] += 1
                    else:
                        bat["death_runs"] += batter_runs
                        if not is_wide:
                            bat["death_balls"] += 1

                    # ── Wickets ────────────────────────────────────────────
                    if "wickets" in delivery:
                        for wicket in delivery["wickets"]:
                            if wicket["kind"] not in ["run out", "retired hurt", "obstructing the field"]:
                                bowl_stats[bowler]["wickets"] += 1
                            player_out = wicket["player_out"]
                            bat_meta.setdefault(player_out, {
                                "team": batting_team,
                                "opponent": bowling_team,
                            })
                            bat_stats[player_out]["dismissed"] += 1

                    # ── Bowling stats ──────────────────────────────────────
                    bowl = bowl_stats[bowler]
                    bowl["runs_conceded"] += total_runs
                    if not is_wide and not is_noball:
                        bowl["balls_bowled"] += 1
                    if is_wide:
                        bowl["wides"] += 1
                    if is_noball:
                        bowl["noballs"] += 1
                    if total_runs == 0:
                        bowl["dot_balls"] += 1

                    # Phase-wise bowling
                    if phase == "powerplay":
                        bowl["pp_runs"] += total_runs
                        if not is_wide and not is_noball:
                            bowl["pp_balls"] += 1
                    elif phase == "middle":
                        bowl["mid_runs"] += total_runs
                        if not is_wide and not is_noball:
                            bowl["mid_balls"] += 1
                    else:
                        bowl["death_runs"] += total_runs
                        if not is_wide and not is_noball:
                            bowl["death_balls"] += 1

        # ── Save batting records for this match ───────────────────────────
        for player, stats in bat_stats.items():
            meta = bat_meta.get(player, {"team": "Unknown", "opponent": "Unknown"})
            batting_records.append({
                "player": player,
                "match_file": filename,
                "tournament": "IPL",
                "match_number": match_number,
                "date": match_date,
                "season": season,
                "venue": venue,
                "city": city,
                "team": meta["team"],
                "opponent": meta["opponent"],
                "winner": winner,
                **stats
            })

        # ── Save bowling records for this match ───────────────────────────
        for player, stats in bowl_stats.items():
            meta = bowl_meta.get(player, {"team": "Unknown", "opponent": "Unknown"})
            bowling_records.append({
                "player": player,
                "match_file": filename,
                "tournament": "IPL",
                "match_number": match_number,
                "date": match_date,
                "season": season,
                "venue": venue,
                "city": city,
                "team": meta["team"],
                "opponent": meta["opponent"],
                "winner": winner,
                **stats
            })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(all_files)} files...")

    print("Done parsing all IPL matches!")
    return pd.DataFrame(batting_records), pd.DataFrame(bowling_records)


# ─── RUN & SAVE ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batting_df, bowling_df = parse_all_matches(DATA_FOLDER)

    # Save
    batting_df.to_csv(f"{OUTPUT_FOLDER}/ipl_batting_raw.csv", index=False)
    bowling_df.to_csv(f"{OUTPUT_FOLDER}/ipl_bowling_raw.csv", index=False)

    print(f"\nBatting records : {len(batting_df)} rows")
    print(f"Bowling records : {len(bowling_df)} rows")
    print(f"\nBatting columns : {list(batting_df.columns)}")
    print(f"Bowling columns : {list(bowling_df.columns)}")

    print("\n=== BATTING SAMPLE ===")
    print(batting_df.head(3).to_string())
    print("\n=== BOWLING SAMPLE ===")
    print(bowling_df.head(3).to_string())
