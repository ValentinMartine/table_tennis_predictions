import chess.pgn
import sqlite3
from pathlib import Path
from loguru import logger


class PGNImporter:
    def __init__(self, db_path: str = "data/chess_matches.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        # Already handled by database.py but just in case
        pass

    def import_file(
        self,
        pgn_path: str,
        player_whitelist: list[str] = None,
        tournament_name: str = None,
    ):
        """
        Imports games from a PGN file.
        If player_whitelist is provided, only imports games where at least one player is in the list.
        """
        if not Path(pgn_path).exists():
            logger.error(f"File not found: {pgn_path}")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        logger.info(f"Scanning PGN for candidates: {pgn_path}")
        # 1. Pre-filter: Only parse games that have a candidate name in the text
        # This is MUCH faster than individual read_game calls.
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split into games (PGN games start with [Event ...)
        import re

        games_raw = re.split(r"\n(?=\[Event )", content)
        logger.info(f"Found {len(games_raw)} raw games in file.")

        count = 0
        skipped = 0

        for raw_game in games_raw:
            # Quick check if any candidate name is in the headers part
            # We look for [White "Name"] or [Black "Name"]
            interesting = False
            if player_whitelist:
                for candidate in player_whitelist:
                    # We check for the name in quotes to avoid partial matches
                    if f'"{candidate}"' in raw_game:
                        interesting = True
                        break
            else:
                interesting = True

            if not interesting:
                skipped += 1
                continue

            # 2. Parse the interesting game
            from io import StringIO

            game = chess.pgn.read_game(StringIO(raw_game))
            if not game:
                continue

            headers = game.headers
            white = headers.get("White", "Unknown")
            black = headers.get("Black", "Unknown")
            result_str = headers.get("Result", "*")
            date = headers.get("Date", "???")
            round_num = headers.get("Round", "1")
            event = tournament_name or headers.get("Event", "Unknown")

            # Filter Blitz/Rapid
            if any(
                x in event.lower() for x in ["blitz", "rapid", "bullet", "armageddon"]
            ):
                skipped += 1
                continue

            # Map result
            result_map = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}
            res_val = result_map.get(result_str, None)
            if res_val is None:
                skipped += 1
                continue

            # Duplicate Check
            cursor.execute(
                """
                SELECT id FROM matches 
                WHERE white_id = (SELECT id FROM players WHERE name = ?) 
                  AND black_id = (SELECT id FROM players WHERE name = ?)
                  AND played_at = ?
                  AND tournament = ?
            """,
                (white, black, date, event),
            )
            if cursor.fetchone():
                skipped += 1
                continue

            # Get or create player IDs
            w_id = self._get_or_create_player(cursor, white)
            b_id = self._get_or_create_player(cursor, black)

            # Insert match
            cursor.execute(
                """
            INSERT INTO matches (white_id, black_id, result, played_at, tournament, round)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (w_id, b_id, res_val, date, event, round_num),
            )

            count += 1
            if count % 100 == 0:
                logger.info(f"Imported {count} games...")

                # Get or create player IDs
                w_id = self._get_or_create_player(cursor, white)
                b_id = self._get_or_create_player(cursor, black)

                # Insert match
                cursor.execute(
                    """
                INSERT INTO matches (white_id, black_id, result, played_at, tournament, round)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (w_id, b_id, res_val, date, event, round_num),
                )

                count += 1
                if count % 100 == 0:
                    logger.info(f"Imported {count} games (Skipped {skipped})...")

        conn.commit()
        conn.close()
        logger.success(f"Final: Imported {count}, Skipped {skipped} from {pgn_path}")

    def _get_or_create_player(self, cursor, name):
        cursor.execute("SELECT id FROM players WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            cursor.execute("INSERT INTO players (name) VALUES (?)", (name,))
            return cursor.lastrowid


if __name__ == "__main__":
    # Test
    importer = PGNImporter()
    # importer.import_file("data/sample.pgn")
