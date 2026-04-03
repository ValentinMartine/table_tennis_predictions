import pandas as pd
import sqlite3
import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from chess_src.database import DEFAULT_DB


class ResultsImporter:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path

    def import_csv(self, csv_path: str):
        """Imports results from a CSV file into the database."""
        if not os.path.exists(csv_path):
            logger.error(f"CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        logger.info(f"Importing {len(df)} results from {csv_path}...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        for _, row in df.iterrows():
            w_name = str(row["white"]).strip()
            b_name = str(row["black"]).strip()
            res = float(row["result"])
            date = str(row["date"])
            tournament = str(row["tournament"])
            round_num = str(row["round"])

            w_id = self._get_or_create_player(cursor, w_name)
            b_id = self._get_or_create_player(cursor, b_name)

            # Check for existing match
            cursor.execute(
                """
            SELECT id FROM matches 
            WHERE white_id = ? AND black_id = ? AND played_at = ? AND round = ?
            """,
                (w_id, b_id, date, round_num),
            )

            if not cursor.fetchone():
                cursor.execute(
                    """
                INSERT INTO matches (white_id, black_id, result, played_at, tournament, round)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (w_id, b_id, res, date, tournament, round_num),
                )
                count += 1

        conn.commit()
        conn.close()
        logger.success(f"Successfully imported {count} NEW matches.")

    def _get_or_create_player(self, cursor, name):
        cursor.execute("SELECT id FROM players WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            # Check for name variants (e.g. "R. Praggnanandhaa" vs "Praggnanandhaa")
            cursor.execute("INSERT INTO players (name) VALUES (?)", (name,))
            return cursor.lastrowid


if __name__ == "__main__":
    importer = ResultsImporter()
    # If run directly as a script, import the history file
    history_file = PROJECT_ROOT / "data" / "elite_classical_history.csv"
    if history_file.exists():
        importer.import_csv(str(history_file))
    else:
        logger.warning("Usage: python import_results.py [/path/to/your/results.csv]")
