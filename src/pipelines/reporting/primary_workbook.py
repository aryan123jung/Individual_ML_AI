from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "primary_data_workbook"
OUTPUT_FILE = OUTPUT_DIR / "current_season_primary_data.xlsx"

IPL_BATTING_RAW = DATA_DIR / "ipl_batting_raw.csv"
IPL_BOWLING_RAW = DATA_DIR / "ipl_bowling_raw.csv"
IPL_CURRENT_SQUAD = DATA_DIR / "ipl_2026_current_squad.csv"
SMAT_BATTING_RAW = DATA_DIR / "smat_batting_raw.csv"
SMAT_BOWLING_RAW = DATA_DIR / "smat_bowling_raw.csv"

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

ET.register_namespace("", NS_MAIN)
ET.register_namespace("r", NS_REL)


def excel_col_letter(n: int) -> str:
    result = ""
    while n:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def normalize_season_value(value: str) -> int:
    text = str(value).strip()
    if "/" in text:
        left = text.split("/", 1)[0]
        if left.isdigit():
            return int(left)
    if text.isdigit():
        return int(text)
    return -1


def latest_season(df: pd.DataFrame) -> str:
    seasons = df["season"].dropna().astype(str).unique().tolist()
    if not seasons:
        raise ValueError("No seasons found in the dataframe.")
    return max(seasons, key=normalize_season_value)


def build_roster(competition: str, batting_path: Path, bowling_path: Path) -> tuple[pd.DataFrame, str]:
    if competition == "IPL" and IPL_CURRENT_SQUAD.exists():
        roster = pd.read_csv(IPL_CURRENT_SQUAD)
        season = latest_season(roster)
        return roster, season

    batting = pd.read_csv(batting_path)
    bowling = pd.read_csv(bowling_path)

    season = latest_season(batting if competition == "IPL" else bowling)
    batting = batting[batting["season"].astype(str) == season].copy()
    bowling = bowling[bowling["season"].astype(str) == season].copy()

    batting["runs"] = pd.to_numeric(batting["runs"], errors="coerce").fillna(0)
    batting["balls_faced"] = pd.to_numeric(batting["balls_faced"], errors="coerce").fillna(0)
    batting["fours"] = pd.to_numeric(batting["fours"], errors="coerce").fillna(0)
    batting["sixes"] = pd.to_numeric(batting["sixes"], errors="coerce").fillna(0)
    batting["dismissed"] = pd.to_numeric(batting["dismissed"], errors="coerce").fillna(0)
    batting["dot_balls"] = pd.to_numeric(batting["dot_balls"], errors="coerce").fillna(0)

    bowling["runs_conceded"] = pd.to_numeric(bowling["runs_conceded"], errors="coerce").fillna(0)
    bowling["balls_bowled"] = pd.to_numeric(bowling["balls_bowled"], errors="coerce").fillna(0)
    bowling["wickets"] = pd.to_numeric(bowling["wickets"], errors="coerce").fillna(0)
    bowling["dot_balls"] = pd.to_numeric(bowling["dot_balls"], errors="coerce").fillna(0)

    batting_agg = (
        batting.groupby(["team", "player"], as_index=False)
        .agg(
            batting_innings=("match_file", "count"),
            runs=("runs", "sum"),
            balls_faced=("balls_faced", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
            dismissals=("dismissed", "sum"),
            batting_dot_balls=("dot_balls", "sum"),
        )
        .rename(columns={"player": "player_name"})
    )

    bowling_agg = (
        bowling.groupby(["team", "player"], as_index=False)
        .agg(
            bowling_innings=("match_file", "count"),
            balls_bowled=("balls_bowled", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            wickets=("wickets", "sum"),
            bowling_dot_balls=("dot_balls", "sum"),
        )
        .rename(columns={"player": "player_name"})
    )

    roster = pd.concat(
        [
            batting[["team", "player"]].assign(competition=competition, season=season),
            bowling[["team", "player"]].assign(competition=competition, season=season),
        ],
        ignore_index=True,
    ).dropna(subset=["team", "player"]).drop_duplicates()
    roster = roster.rename(columns={"player": "player_name"})

    match_counts = (
        pd.concat(
            [
                batting[["team", "player", "match_file"]],
                bowling[["team", "player", "match_file"]],
            ],
            ignore_index=True,
        )
        .dropna(subset=["team", "player", "match_file"])
        .drop_duplicates()
        .groupby(["team", "player"], as_index=False)["match_file"]
        .count()
        .rename(columns={"player": "player_name", "match_file": "matches_played"})
    )

    roster = roster.merge(match_counts, on=["team", "player_name"], how="left")
    roster = roster.merge(batting_agg, on=["team", "player_name"], how="left")
    roster = roster.merge(bowling_agg, on=["team", "player_name"], how="left")

    roster["status"] = "Verified from scorecard data"
    roster["primary_role"] = ""
    roster["batting_role"] = ""
    roster["bowling_role"] = ""
    roster["notes"] = ""

    int_cols = [
        "matches_played",
        "batting_innings",
        "runs",
        "balls_faced",
        "fours",
        "sixes",
        "dismissals",
        "bowling_innings",
        "balls_bowled",
        "runs_conceded",
        "wickets",
    ]
    for col in int_cols:
        roster[col] = pd.to_numeric(roster[col], errors="coerce").fillna(0).round(0).astype(int)

    roster["dot_balls"] = (
        pd.to_numeric(roster.get("batting_dot_balls"), errors="coerce").fillna(0)
        + pd.to_numeric(roster.get("bowling_dot_balls"), errors="coerce").fillna(0)
    ).round(0).astype(int)

    roster["batting_strike_rate"] = roster.apply(
        lambda r: round((r["runs"] / r["balls_faced"]) * 100, 2) if r["balls_faced"] > 0 else None,
        axis=1,
    )
    roster["batting_average"] = roster.apply(
        lambda r: round(r["runs"] / r["dismissals"], 2) if r["dismissals"] > 0 else None,
        axis=1,
    )
    roster["bowling_economy"] = roster.apply(
        lambda r: round(r["runs_conceded"] / (r["balls_bowled"] / 6), 2) if r["balls_bowled"] > 0 else None,
        axis=1,
    )
    roster["bowling_strike_rate"] = roster.apply(
        lambda r: round(r["balls_bowled"] / r["wickets"], 2) if r["wickets"] > 0 else None,
        axis=1,
    )
    roster["source_note"] = f"Derived from {competition} raw processed season {season} scorecard data"

    roster = roster[
        [
            "competition",
            "season",
            "team",
            "player_name",
            "status",
            "primary_role",
            "batting_role",
            "bowling_role",
            "matches_played",
            "batting_innings",
            "runs",
            "balls_faced",
            "fours",
            "sixes",
            "dismissals",
            "batting_strike_rate",
            "batting_average",
            "bowling_innings",
            "balls_bowled",
            "runs_conceded",
            "wickets",
            "bowling_economy",
            "bowling_strike_rate",
            "dot_balls",
            "notes",
            "source_note",
        ]
    ].sort_values(["team", "player_name"], kind="stable").reset_index(drop=True)
    return roster, season


def build_summary_tables(primary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    competition_summary = (
        primary_df.groupby("competition", as_index=False)
        .agg(rows=("player_name", "count"), unique_players=("player_name", "nunique"), teams=("team", "nunique"))
        .sort_values("competition")
    )

    team_summary = (
        primary_df.groupby(["competition", "team"], as_index=False)
        .agg(players=("player_name", "count"))
        .sort_values(["competition", "players", "team"], ascending=[True, False, True])
    )

    lookups = pd.DataFrame(
        {
            "role_bucket": [
                "aggressive_opener",
                "anchor_opener",
                "top_order_aggressor",
                "top_order_anchor",
                "middle_order",
                "finisher",
                "lower_order_hitter",
                "powerplay_bowler",
                "middle_overs_bowler",
                "death_bowler",
                "allrounder",
                "wicketkeeper_batter",
            ],
            "use_case": [
                "Fast scoring opener",
                "Stability at the top",
                "Attack early in the innings",
                "Anchor the top order",
                "Control the middle overs",
                "Finish innings strongly",
                "Lower-order hitting depth",
                "New-ball specialist",
                "Middle-overs control",
                "Death overs specialist",
                "Contribute with both bat and ball",
                "Keeper plus batting value",
            ],
        }
    )

    return {"competition_summary": competition_summary, "team_summary": team_summary, "role_lookup": lookups}


def row_to_cell_value(value):
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    return str(value)


def build_sheet_xml(
    rows: list[list],
    *,
    styled_rows: set[int] | None = None,
    freeze_row: int | None = None,
    freeze_col: int | None = None,
    auto_filter: str | None = None,
    col_widths: list[float] | None = None,
    merged_ranges: list[str] | None = None,
) -> bytes:
    styled_rows = styled_rows or set()
    merged_ranges = merged_ranges or []

    worksheet = ET.Element(f"{{{NS_MAIN}}}worksheet")
    if rows:
        worksheet.set("dimension", f"A1:{excel_col_letter(max(len(r) for r in rows))}{len(rows)}")

    sheet_views = ET.SubElement(worksheet, f"{{{NS_MAIN}}}sheetViews")
    sheet_view = ET.SubElement(sheet_views, f"{{{NS_MAIN}}}sheetView", {"workbookViewId": "0"})
    if freeze_row or freeze_col:
        pane_attrs = {"state": "frozen"}
        if freeze_col:
            pane_attrs["xSplit"] = str(freeze_col)
        if freeze_row:
            pane_attrs["ySplit"] = str(freeze_row)
        if freeze_row and freeze_col:
            pane_attrs["topLeftCell"] = f"{excel_col_letter(freeze_col + 1)}{freeze_row + 1}"
            pane_attrs["activePane"] = "bottomRight"
        elif freeze_row:
            pane_attrs["topLeftCell"] = f"A{freeze_row + 1}"
            pane_attrs["activePane"] = "bottomLeft"
        elif freeze_col:
            pane_attrs["topLeftCell"] = f"{excel_col_letter(freeze_col + 1)}1"
            pane_attrs["activePane"] = "topRight"
        ET.SubElement(sheet_view, f"{{{NS_MAIN}}}pane", pane_attrs)

    ET.SubElement(worksheet, f"{{{NS_MAIN}}}sheetFormatPr", {"defaultRowHeight": "15"})

    if col_widths:
        cols_el = ET.SubElement(worksheet, f"{{{NS_MAIN}}}cols")
        for idx, width in enumerate(col_widths, start=1):
            ET.SubElement(
                cols_el,
                f"{{{NS_MAIN}}}col",
                {"min": str(idx), "max": str(idx), "width": f"{width:.2f}", "customWidth": "1"},
            )

    sheet_data = ET.SubElement(worksheet, f"{{{NS_MAIN}}}sheetData")
    for row_idx, row_values in enumerate(rows, start=1):
        row_el = ET.SubElement(sheet_data, f"{{{NS_MAIN}}}row", {"r": str(row_idx)})
        row_style = 1 if row_idx in styled_rows else 0
        for col_idx, value in enumerate(row_values, start=1):
            cell_value = row_to_cell_value(value)
            if cell_value is None:
                continue
            cell_ref = f"{excel_col_letter(col_idx)}{row_idx}"
            cell_attrs = {"r": cell_ref}
            if row_style:
                cell_attrs["s"] = "1"
            cell = ET.SubElement(row_el, f"{{{NS_MAIN}}}c", cell_attrs)
            if isinstance(cell_value, (int, float)) and not isinstance(cell_value, bool):
                cell.set("t", "n")
                ET.SubElement(cell, f"{{{NS_MAIN}}}v").text = str(cell_value)
            else:
                cell.set("t", "inlineStr")
                is_el = ET.SubElement(cell, f"{{{NS_MAIN}}}is")
                ET.SubElement(is_el, f"{{{NS_MAIN}}}t").text = str(cell_value)

    if auto_filter and rows:
        ET.SubElement(worksheet, f"{{{NS_MAIN}}}autoFilter", {"ref": auto_filter})

    if merged_ranges:
        merge_el = ET.SubElement(worksheet, f"{{{NS_MAIN}}}mergeCells", {"count": str(len(merged_ranges))})
        for ref in merged_ranges:
            ET.SubElement(merge_el, f"{{{NS_MAIN}}}mergeCell", {"ref": ref})

    ET.SubElement(
        worksheet,
        f"{{{NS_MAIN}}}pageMargins",
        {"left": "0.7", "right": "0.7", "top": "0.75", "bottom": "0.75", "header": "0.3", "footer": "0.3"},
    )
    return ET.tostring(worksheet, encoding="utf-8", xml_declaration=True)


def build_styles_xml() -> bytes:
    root = ET.Element(f"{{{NS_MAIN}}}styleSheet")
    fonts = ET.SubElement(root, f"{{{NS_MAIN}}}fonts", {"count": "2"})
    font1 = ET.SubElement(fonts, f"{{{NS_MAIN}}}font")
    ET.SubElement(font1, f"{{{NS_MAIN}}}sz", {"val": "11"})
    ET.SubElement(font1, f"{{{NS_MAIN}}}color", {"rgb": "FF000000"})
    ET.SubElement(font1, f"{{{NS_MAIN}}}name", {"val": "Calibri"})
    ET.SubElement(font1, f"{{{NS_MAIN}}}family", {"val": "2"})
    font2 = ET.SubElement(fonts, f"{{{NS_MAIN}}}font")
    ET.SubElement(font2, f"{{{NS_MAIN}}}b")
    ET.SubElement(font2, f"{{{NS_MAIN}}}sz", {"val": "11"})
    ET.SubElement(font2, f"{{{NS_MAIN}}}color", {"rgb": "FFFFFFFF"})
    ET.SubElement(font2, f"{{{NS_MAIN}}}name", {"val": "Calibri"})
    ET.SubElement(font2, f"{{{NS_MAIN}}}family", {"val": "2"})

    fills = ET.SubElement(root, f"{{{NS_MAIN}}}fills", {"count": "2"})
    fill1 = ET.SubElement(fills, f"{{{NS_MAIN}}}fill")
    ET.SubElement(fill1, f"{{{NS_MAIN}}}patternFill", {"patternType": "none"})
    fill2 = ET.SubElement(fills, f"{{{NS_MAIN}}}fill")
    pattern = ET.SubElement(fill2, f"{{{NS_MAIN}}}patternFill", {"patternType": "solid"})
    ET.SubElement(pattern, f"{{{NS_MAIN}}}fgColor", {"rgb": "FF1F4E78"})
    ET.SubElement(pattern, f"{{{NS_MAIN}}}bgColor", {"indexed": "64"})

    borders = ET.SubElement(root, f"{{{NS_MAIN}}}borders", {"count": "1"})
    border = ET.SubElement(borders, f"{{{NS_MAIN}}}border")
    for edge in ["left", "right", "top", "bottom", "diagonal"]:
        ET.SubElement(border, f"{{{NS_MAIN}}}{edge}")

    cell_style_xfs = ET.SubElement(root, f"{{{NS_MAIN}}}cellStyleXfs", {"count": "1"})
    ET.SubElement(cell_style_xfs, f"{{{NS_MAIN}}}xf", {"numFmtId": "0", "fontId": "0", "fillId": "0", "borderId": "0"})
    cell_xfs = ET.SubElement(root, f"{{{NS_MAIN}}}cellXfs", {"count": "2"})
    ET.SubElement(cell_xfs, f"{{{NS_MAIN}}}xf", {"numFmtId": "0", "fontId": "0", "fillId": "0", "borderId": "0", "xfId": "0"})
    ET.SubElement(
        cell_xfs,
        f"{{{NS_MAIN}}}xf",
        {"numFmtId": "0", "fontId": "1", "fillId": "1", "borderId": "0", "xfId": "0", "applyFont": "1", "applyFill": "1"},
    )
    cell_styles = ET.SubElement(root, f"{{{NS_MAIN}}}cellStyles", {"count": "1"})
    ET.SubElement(cell_styles, f"{{{NS_MAIN}}}cellStyle", {"name": "Normal", "xfId": "0", "builtinId": "0"})
    ET.SubElement(root, f"{{{NS_MAIN}}}dxfs", {"count": "0"})
    ET.SubElement(root, f"{{{NS_MAIN}}}tableStyles", {"count": "0", "defaultTableStyle": "TableStyleMedium2", "defaultPivotStyle": "PivotStyleLight16"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_workbook_xml(sheet_names: list[str]) -> bytes:
    root = ET.Element(f"{{{NS_MAIN}}}workbook")
    sheets_el = ET.SubElement(root, f"{{{NS_MAIN}}}sheets")
    for idx, name in enumerate(sheet_names, start=1):
        ET.SubElement(sheets_el, f"{{{NS_MAIN}}}sheet", {"name": name, "sheetId": str(idx), f"{{{NS_REL}}}id": f"rId{idx}"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_workbook_rels_xml(sheet_count: int) -> bytes:
    root = ET.Element(f"{{{NS_PKG_REL}}}Relationships")
    for idx in range(1, sheet_count + 1):
        ET.SubElement(
            root,
            f"{{{NS_PKG_REL}}}Relationship",
            {
                "Id": f"rId{idx}",
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
                "Target": f"worksheets/sheet{idx}.xml",
            },
        )
    ET.SubElement(
        root,
        f"{{{NS_PKG_REL}}}Relationship",
        {"Id": f"rId{sheet_count + 1}", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "Target": "styles.xml"},
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_root_rels_xml() -> bytes:
    root = ET.Element(f"{{{NS_PKG_REL}}}Relationships")
    ET.SubElement(
        root,
        f"{{{NS_PKG_REL}}}Relationship",
        {"Id": "rId1", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "Target": "xl/workbook.xml"},
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_content_types_xml(sheet_count: int) -> bytes:
    root = ET.Element("Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types")
    ET.SubElement(root, "Default", {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(root, "Default", {"Extension": "xml", "ContentType": "application/xml"})
    ET.SubElement(
        root,
        "Override",
        {"PartName": "/xl/workbook.xml", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"},
    )
    for idx in range(1, sheet_count + 1):
        ET.SubElement(
            root,
            "Override",
            {"PartName": f"/xl/worksheets/sheet{idx}.xml", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"},
        )
    ET.SubElement(root, "Override", {"PartName": "/xl/styles.xml", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def widths_for_columns(widths: list[int], minimums: dict[int, int] | None = None) -> list[float]:
    minimums = minimums or {}
    result = []
    for idx, width in enumerate(widths, start=1):
        result.append(min(float(max(width, minimums.get(idx, 10))) + 2, 40.0))
    return result


def text_width(values: Iterable) -> int:
    max_len = 0
    for value in values:
        if is_missing(value):
            continue
        max_len = max(max_len, len(str(value)))
    return max_len


def build_sheet_rows_from_df(df: pd.DataFrame) -> list[list]:
    rows = [df.columns.tolist()]
    for _, record in df.iterrows():
        rows.append([record[col] for col in df.columns])
    return rows


def main() -> None:
    ipl_df, ipl_season = build_roster("IPL", IPL_BATTING_RAW, IPL_BOWLING_RAW)
    smat_df, smat_season = build_roster("SMAT", SMAT_BATTING_RAW, SMAT_BOWLING_RAW)
    primary_df = pd.concat([ipl_df, smat_df], ignore_index=True)
    tables = build_summary_tables(primary_df)

    summary_rows = [
        ["Primary data workbook for IPL and SMAT season player entry"],
        [
            f"IPL current season: {ipl_season} | SMAT latest season: {smat_season} | "
            "Player names and season statistics are generated from the cleaned processed datasets for review.",
        ],
        [],
        ["Competition Summary"],
        ["competition", "rows", "unique_players", "teams"],
    ]
    for _, row in tables["competition_summary"].iterrows():
        summary_rows.append([row["competition"], int(row["rows"]), int(row["unique_players"]), int(row["teams"])])

    summary_rows.extend([[], ["Top Team Counts"], ["competition", "team", "players"]])
    for _, row in tables["team_summary"].head(20).iterrows():
        summary_rows.append([row["competition"], row["team"], int(row["players"])])

    lookups = tables["role_lookup"]
    ipl_teams = sorted(ipl_df["team"].dropna().astype(str).unique().tolist())
    smat_teams = sorted(smat_df["team"].dropna().astype(str).unique().tolist())
    max_len = max(len(ipl_teams), len(smat_teams), len(lookups))
    lookup_rows = [["IPL Teams", "SMAT Teams", "Role Buckets", "Role Use Case"]]
    for i in range(max_len):
        lookup_rows.append(
            [
                ipl_teams[i] if i < len(ipl_teams) else "",
                smat_teams[i] if i < len(smat_teams) else "",
                lookups.iloc[i]["role_bucket"] if i < len(lookups) else "",
                lookups.iloc[i]["use_case"] if i < len(lookups) else "",
            ]
        )

    sheets = {
        "Summary": {
            "rows": summary_rows,
            "styled_rows": {1, 4, 5, 9, 10},
            "freeze_row": 4,
            "freeze_col": None,
            "auto_filter": None,
            "merged_ranges": ["A1:F1"],
            "col_widths": [18, 16, 16, 12, 14, 14],
        },
        "Primary_Data": {
            "rows": build_sheet_rows_from_df(primary_df),
            "styled_rows": {1},
            "freeze_row": 1,
            "freeze_col": 1,
            "auto_filter": f"A1:{excel_col_letter(len(primary_df.columns))}{len(primary_df) + 1}",
            "merged_ranges": [],
            "col_widths": widths_for_columns(
                [
                    text_width(primary_df["competition"]),
                    text_width(primary_df["season"]),
                    text_width(primary_df["team"]),
                    text_width(primary_df["player_name"]),
                    10,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    14,
                    14,
                    12,
                    12,
                    12,
                    12,
                    14,
                    14,
                    12,
                    16,
                    32,
                ],
                {4: 24},
            ),
        },
        "IPL_2026": {
            "rows": build_sheet_rows_from_df(ipl_df),
            "styled_rows": {1},
            "freeze_row": 1,
            "freeze_col": 1,
            "auto_filter": f"A1:{excel_col_letter(len(ipl_df.columns))}{len(ipl_df) + 1}",
            "merged_ranges": [],
            "col_widths": widths_for_columns(
                [
                    text_width(ipl_df["competition"]),
                    text_width(ipl_df["season"]),
                    text_width(ipl_df["team"]),
                    text_width(ipl_df["player_name"]),
                    10,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    14,
                    14,
                    12,
                    12,
                    12,
                    12,
                    14,
                    14,
                    12,
                    16,
                    32,
                ],
                {4: 24},
            ),
        },
        "SMAT_2024_25": {
            "rows": build_sheet_rows_from_df(smat_df),
            "styled_rows": {1},
            "freeze_row": 1,
            "freeze_col": 1,
            "auto_filter": f"A1:{excel_col_letter(len(smat_df.columns))}{len(smat_df) + 1}",
            "merged_ranges": [],
            "col_widths": widths_for_columns(
                [
                    text_width(smat_df["competition"]),
                    text_width(smat_df["season"]),
                    text_width(smat_df["team"]),
                    text_width(smat_df["player_name"]),
                    10,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    14,
                    14,
                    12,
                    12,
                    12,
                    12,
                    14,
                    14,
                    12,
                    16,
                    32,
                ],
                {4: 24},
            ),
        },
        "Lookups": {
            "rows": lookup_rows,
            "styled_rows": {1},
            "freeze_row": 1,
            "freeze_col": None,
            "auto_filter": f"A1:D{len(lookup_rows)}",
            "merged_ranges": [],
            "col_widths": [20, 20, 24, 36],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sheet_names = list(sheets.keys())
    with ZipFile(OUTPUT_FILE, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", build_content_types_xml(len(sheet_names)))
        zf.writestr("_rels/.rels", build_root_rels_xml())
        zf.writestr("xl/workbook.xml", build_workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", build_workbook_rels_xml(len(sheet_names)))
        zf.writestr("xl/styles.xml", build_styles_xml())

        for idx, (sheet_name, spec) in enumerate(sheets.items(), start=1):
            zf.writestr(
                f"xl/worksheets/sheet{idx}.xml",
                build_sheet_xml(
                    spec["rows"],
                    styled_rows=spec["styled_rows"],
                    freeze_row=spec["freeze_row"],
                    freeze_col=spec["freeze_col"],
                    auto_filter=spec["auto_filter"],
                    col_widths=spec["col_widths"],
                    merged_ranges=spec["merged_ranges"],
                ),
            )

    print(f"Saved workbook to {OUTPUT_FILE}")
    print(f"Sheets: {', '.join(sheet_names)}")
    print(f"Primary data rows: {len(primary_df)}")


if __name__ == "__main__":
    main()
