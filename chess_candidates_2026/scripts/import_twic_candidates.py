"""
Import classical games involving the 8 Candidates 2026 players from TWIC PGN files.
Uses regex-based header parsing (no python-chess dependency).
"""

import re
import sqlite3
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / "data" / "chess_matches.db")
TWIC_DIR = PROJECT_ROOT / "data" / "raw" / "twic"

# TWIC name → (canonical DB name, DB player id)
# DB player ids match the fide_id column used as primary key
CANDIDATE_TWIC_NAMES = {
    "Nakamura,Hi": ("Hikaru Nakamura", 2004887),
    "Caruana,F": ("Fabiano Caruana", 2020009),
    "Wei Yi": ("Wei Yi", 8603405),
    "Giri,A": ("Anish Giri", 24116068),
    "Sindarov,Javokhir": ("Javokhir Sindarov", 14205481),
    "Praggnanandhaa,R": ("R. Praggnanandhaa", 25059650),
    "Bluebaum,M": ("Matthias Bluebaum", 4661654),
    "Esipenko,Andrey": ("Andrey Esipenko", 24175439),
}

SKIP_KEYWORDS = ["blitz", "rapid", "bullet", "armageddon", "960", "fischer"]
RESULT_MAP = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}
MIN_OPPONENT_ELO = 2400  # ignore weak opponents to keep signal clean

_TAG = re.compile(r'\[(\w+)\s+"([^"]+)"\]')


def parse_headers(game_text: str) -> dict:
    return {m.group(1): m.group(2) for m in _TAG.finditer(game_text)}


def get_or_create_player(cursor, name: str, rating: int | None) -> int:
    row = cursor.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cursor.execute(
        "INSERT INTO players (name, rating_initial) VALUES (?, ?)",
        (name, rating),
    )
    return cursor.lastrowid


def already_exists(
    cursor, white_id: int, black_id: int, played_at: str, tournament: str
) -> bool:
    return (
        cursor.execute(
            "SELECT 1 FROM matches WHERE white_id=? AND black_id=? AND played_at=? AND tournament=?",
            (white_id, black_id, played_at, tournament),
        ).fetchone()
        is not None
    )


def import_file(pgn_path: Path, conn: sqlite3.Connection) -> tuple[int, int]:
    cursor = conn.cursor()
    content = pgn_path.read_text(encoding="utf-8", errors="replace")

    # Split on game boundaries (each game starts with [Event)
    games = re.split(r"\n(?=\[Event )", content)

    imported = skipped = 0
    candidate_names = set(CANDIDATE_TWIC_NAMES.keys())

    for raw in games:
        # Quick pre-filter: must mention at least one candidate name
        if not any(name in raw for name in candidate_names):
            skipped += 1
            continue

        h = parse_headers(raw)
        white_twic = h.get("White", "")
        black_twic = h.get("Black", "")

        w_is_cand = white_twic in CANDIDATE_TWIC_NAMES
        b_is_cand = black_twic in CANDIDATE_TWIC_NAMES
        if not (w_is_cand or b_is_cand):
            skipped += 1
            continue

        result = RESULT_MAP.get(h.get("Result", ""), None)
        if result is None:
            skipped += 1
            continue

        event = h.get("Event", "Unknown")
        if any(kw in event.lower() for kw in SKIP_KEYWORDS):
            skipped += 1
            continue

        # Skip Candidates 2026 — already in DB
        if "Candidates" in event and "2026" in event:
            skipped += 1
            continue

        try:
            white_elo = int(h.get("WhiteElo", 0)) or None
            black_elo = int(h.get("BlackElo", 0)) or None
        except ValueError:
            white_elo = black_elo = None

        # Enforce minimum opponent Elo
        opp_elo = black_elo if w_is_cand else white_elo
        if opp_elo and opp_elo < MIN_OPPONENT_ELO:
            skipped += 1
            continue

        # Normalise date (TWIC: "2024.01.15" → "2024-01-15")
        raw_date = h.get("Date", "").replace(".", "-")
        if "?" in raw_date or not raw_date:
            skipped += 1
            continue

        round_num = h.get("Round", "1")
        try:
            round_num = int(float(round_num))
        except (ValueError, TypeError):
            round_num = 1

        # Resolve player IDs
        if w_is_cand:
            _, w_id = CANDIDATE_TWIC_NAMES[white_twic]
            b_name = black_twic
            b_id = get_or_create_player(cursor, b_name, black_elo)
        else:
            w_name = white_twic
            w_id = get_or_create_player(cursor, w_name, white_elo)
            _, b_id = CANDIDATE_TWIC_NAMES[black_twic]

        if already_exists(cursor, w_id, b_id, raw_date, event):
            skipped += 1
            continue

        cursor.execute(
            "INSERT INTO matches (white_id, black_id, result, played_at, tournament, round) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (w_id, b_id, result, raw_date, event, round_num),
        )
        imported += 1

    conn.commit()
    return imported, skipped


def main():
    pgn_files = sorted(TWIC_DIR.glob("twic*.pgn"))
    if not pgn_files:
        logger.error(f"No TWIC PGN files found in {TWIC_DIR}")
        return

    conn = sqlite3.connect(DB_PATH)
    total_imported = total_skipped = 0

    for pgn in pgn_files:
        imp, skp = import_file(pgn, conn)
        logger.info(f"{pgn.name}: +{imp} imported, {skp} skipped")
        total_imported += imp
        total_skipped += skp

    conn.close()
    logger.success(f"Done — total imported: {total_imported}, skipped: {total_skipped}")

    # Summary by tournament
    conn2 = sqlite3.connect(DB_PATH)
    import pandas as pd

    df = pd.read_sql_query(
        "SELECT tournament, COUNT(*) cnt FROM matches "
        "WHERE result IS NOT NULL GROUP BY tournament ORDER BY cnt DESC LIMIT 15",
        conn2,
    )
    conn2.close()
    logger.info(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
