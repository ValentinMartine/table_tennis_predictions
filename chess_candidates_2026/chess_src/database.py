import sqlite3
from pathlib import Path

# Standardized path
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB = str(PROJECT_ROOT / "data" / "chess_matches.db")


def init_db(db_path: str = DEFAULT_DB):
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Table Players
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        fide_id INTEGER UNIQUE,
        country TEXT,
        rating_initial INTEGER
    )
    """)

    # Table Matches (historical and upcoming)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        white_id INTEGER NOT NULL,
        black_id INTEGER NOT NULL,
        result REAL, -- 1.0 = White Wins, 0.5 = Draw, 0.0 = Black Wins, NULL = Upcoming
        played_at DATE,
        tournament TEXT,
        round INTEGER,
        is_candidates INTEGER DEFAULT 0,
        FOREIGN KEY (white_id) REFERENCES players (id),
        FOREIGN KEY (black_id) REFERENCES players (id)
    )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")


if __name__ == "__main__":
    init_db()
