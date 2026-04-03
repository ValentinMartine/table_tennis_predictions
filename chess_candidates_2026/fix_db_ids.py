import sqlite3
import yaml

DEFAULT_DB = "c:/Users/valen/Documents/Projets/Projets_Informatiques/table_tennis_pred/chess_candidates_2026/data/chess_matches.db"
CONFIG_PATH = "c:/Users/valen/Documents/Projets/Projets_Informatiques/table_tennis_pred/chess_candidates_2026/config/settings.yaml"


def update_fide_ids():
    print(f"Updating FIDE IDs in {DEFAULT_DB}...")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    players = config["players"]
    conn = sqlite3.connect(DEFAULT_DB)
    cursor = conn.cursor()

    updates = 0
    for p in players:
        name = p["name"]
        fide_id = p["fide_id"]

        # Try exact match first
        cursor.execute("UPDATE players SET fide_id = ? WHERE name = ?", (fide_id, name))
        if cursor.rowcount > 0:
            updates += cursor.rowcount
            print(f"Updated {name} -> {fide_id}")
        else:
            # Try fuzzy match (LIKE %Last%)
            last_name = name.split(" ")[-1]
            cursor.execute(
                "UPDATE players SET fide_id = ? WHERE name LIKE ?",
                (fide_id, f"%{last_name}%"),
            )
            if cursor.rowcount > 0:
                updates += cursor.rowcount
                print(f"Fuzzy Updated {name} -> {fide_id}")
            else:
                print(f"Warning: Could not find player {name} in DB.")

    conn.commit()
    conn.close()
    print(f"Done. {updates} records updated.")


if __name__ == "__main__":
    update_fide_ids()
