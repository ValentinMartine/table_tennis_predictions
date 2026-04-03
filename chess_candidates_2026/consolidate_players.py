import sqlite3
import yaml

DEFAULT_DB = "c:/Users/valen/Documents/Projets/Projets_Informatiques/table_tennis_pred/chess_candidates_2026/data/chess_matches.db"
CONFIG_PATH = "c:/Users/valen/Documents/Projets/Projets_Informatiques/table_tennis_pred/chess_candidates_2026/config/settings.yaml"


def consolidate_and_fix():
    print(f"Consolidating players in {DEFAULT_DB}...")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    players_config = config["players"]
    conn = sqlite3.connect(DEFAULT_DB)
    cursor = conn.cursor()

    total_consolidated = 0
    for p in players_config:
        name = p["name"]
        f_id = p["fide_id"]

        # Find all IDs for this player (fuzzy)
        last_name = name.split(" ")[-1]
        cursor.execute("SELECT id FROM players WHERE name LIKE ?", (f"%{last_name}%",))
        db_ids = [row[0] for row in cursor.fetchall()]

        if not db_ids:
            print(f"Warning: No DB records found for {name}")
            continue

        master_id = db_ids[0]
        duplicates = db_ids[1:]

        # Consolidation
        for dup in duplicates:
            # Update matches
            cursor.execute(
                "UPDATE matches SET white_id = ? WHERE white_id = ?", (master_id, dup)
            )
            cursor.execute(
                "UPDATE matches SET black_id = ? WHERE black_id = ?", (master_id, dup)
            )
            # Delete duplicate
            cursor.execute("DELETE FROM players WHERE id = ?", (dup,))
            total_consolidated += 1
            print(f"Consolidated ID {dup} -> {master_id} for {name}")

        # Set Master FIDE ID
        cursor.execute("UPDATE players SET fide_id = ? WHERE id = ?", (f_id, master_id))
        print(f"Master ID {master_id} set to FIDE ID {f_id} for {name}")

    conn.commit()
    conn.close()
    print(
        f"Done. Processed 8 candidates, consolidated {total_consolidated} duplicates."
    )


if __name__ == "__main__":
    consolidate_and_fix()
