import sqlite3
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DB_PATH = PROJECT_ROOT / "data" / "chess_matches.db"
CSV_PATH = PROJECT_ROOT / "data" / "elite_classical_history.csv"

# CONSTANTS - OFFICIAL 8 CANDIDATES
NAK = 2004887
CAR = 2020009
WEI = 8603405
GIR = 24116068
SIN = 14205481
PRA = 25059650
BLU = 4661654
ESI = 24175439

PLAYERS = {
    "Hikaru Nakamura": (NAK, "USA", 2810),
    "Fabiano Caruana": (CAR, "USA", 2795),
    "Wei Yi": (WEI, "CHN", 2754),
    "Anish Giri": (GIR, "NED", 2753),
    "Javokhir Sindarov": (SIN, "UZB", 2745),
    "R. Praggnanandhaa": (PRA, "IND", 2741),
    "Matthias Bluebaum": (BLU, "GER", 2698),
    "Andrey Esipenko": (ESI, "FID", 2698),
}

# OTHER PLAYERS in CSV (for history)
OTHERS = {
    "Magnus Carlsen": (1503014, "NOR", 2832),
    "Alireza Firouzja": (12573981, "FRA", 2760),
    "Vidit Gujrathi": (5029465, "IND", 2720),
    "Gukesh D": (46616543, "IND", 2765),
    "Vincent Keymer": (46616540, "GER", 2730),
}

# ALL MAPPINGS
ALL_PLAYERS = {**PLAYERS, **OTHERS}


def reset_db():
    print("Repairing DB with stable FIDE IDs...")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS matches")
    cursor.execute("DROP TABLE IF EXISTS players")

    cursor.execute(
        "CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT, country TEXT, rating_initial REAL)"
    )
    cursor.execute(
        "CREATE TABLE matches (id INTEGER PRIMARY KEY AUTOINCREMENT, white_id INTEGER, black_id INTEGER, result REAL, played_at TEXT, tournament TEXT, round INTEGER)"
    )

    # 1. Insert Players
    for name, (fid, cry, rat) in ALL_PLAYERS.items():
        cursor.execute(
            "INSERT INTO players (id, name, country, rating_initial) VALUES (?, ?, ?, ?)",
            (fid, name, cry, rat),
        )

    # 2. Insert CSV Matches (if any)
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            w_id = ALL_PLAYERS.get(row["white"], (None,))[0]
            b_id = ALL_PLAYERS.get(row["black"], (None,))[0]
            if w_id and b_id:
                dt = str(row["date"]).replace(".", "-")
                # Important: candidates matches in CSV should use standard tournament name
                cursor.execute(
                    "INSERT INTO matches (white_id, black_id, result, played_at, tournament, round) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        w_id,
                        b_id,
                        float(row["result"]),
                        dt,
                        row["tournament"],
                        int(row["round"]),
                    ),
                )

    # 3. Ensure ALL 56 matches for Candidates 2026 are present
    # We already have R1-R4 (16 matches) from CSV.
    # We add R5-R14 (40 matches) as NULL results.
    # Official pairings for R5-R14 (based on FIDE double round robin)

    UPCOMING = [
        # Ronde 5
        (PRA, ESI, 5),
        (CAR, BLU, 5),
        (NAK, SIN, 5),
        (GIR, WEI, 5),
        # Ronde 6
        (CAR, ESI, 6),
        (NAK, PRA, 6),
        (GIR, BLU, 6),
        (WEI, SIN, 6),
        # Ronde 7
        (ESI, WEI, 7),
        (SIN, GIR, 7),
        (BLU, NAK, 7),
        (PRA, CAR, 7),
        # Ronde 8
        (ESI, SIN, 8),
        (WEI, BLU, 8),
        (GIR, PRA, 8),
        (NAK, CAR, 8),
        # Ronde 9
        (NAK, ESI, 9),
        (CAR, GIR, 9),
        (PRA, WEI, 9),
        (BLU, SIN, 9),
        # Ronde 10
        (ESI, BLU, 10),
        (SIN, PRA, 10),
        (WEI, CAR, 10),
        (GIR, NAK, 10),
        # Ronde 11
        (GIR, ESI, 11),
        (NAK, WEI, 11),
        (CAR, SIN, 11),
        (PRA, BLU, 11),
        # Ronde 12
        (ESI, PRA, 12),
        (BLU, CAR, 12),
        (SIN, NAK, 12),
        (WEI, GIR, 12),
        # Ronde 13
        (ESI, CAR, 13),
        (PRA, NAK, 13),
        (BLU, GIR, 13),
        (SIN, WEI, 13),
        # Ronde 14
        (WEI, ESI, 14),
        (GIR, SIN, 14),
        (NAK, BLU, 14),
        (CAR, PRA, 14),
    ]

    for w, b, rd in UPCOMING:
        dt = f"2026-04-{10 + rd}"
        cursor.execute(
            "INSERT INTO matches (white_id, black_id, result, played_at, tournament, round) VALUES (?, ?, ?, ?, ?, ?)",
            (w, b, None, dt, "Candidates 2026", rd),
        )

    conn.commit()
    conn.close()
    print("Database REPAIRED and SYNCED.")


if __name__ == "__main__":
    reset_db()
