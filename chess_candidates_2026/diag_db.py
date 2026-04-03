import sqlite3
import pandas as pd
import yaml
import os

DEFAULT_DB = "c:/Users/valen/Documents/Projets/Projets_Informatiques/table_tennis_pred/chess_candidates_2026/data/chess_matches.db"


def diag():
    print("--- Diagnostic Report ---")
    if not os.path.exists(DEFAULT_DB):
        print(f"DB Not Found at {DEFAULT_DB}!")
        return

    conn = sqlite3.connect(DEFAULT_DB)

    print("\n[TABLE: players]")
    try:
        players_df = pd.read_sql_query("SELECT id, name FROM players LIMIT 5", conn)
        print(players_df)
    except:
        print("Table players or column id/name not found")

    print("\n[TABLE: matches] (Candidates 2026)")
    try:
        matches_df = pd.read_sql_query(
            "SELECT id, white_id, black_id, round FROM matches WHERE tournament = 'Candidates 2026' LIMIT 5",
            conn,
        )
        print(matches_df)
    except Exception as e:
        print(f"Error checking matches: {e}")

    with open(
        "c:/Users/valen/Documents/Projets/Projets_Informatiques/table_tennis_pred/config/settings.yaml"
    ) as f:
        config = yaml.safe_load(f)
    print("\n[CONFIG: FIDE IDs]")
    print([p["fide_id"] for p in config["players"]])

    conn.close()


if __name__ == "__main__":
    diag()
