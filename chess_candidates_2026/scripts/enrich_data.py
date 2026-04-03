import sys
import yaml
from pathlib import Path
from loguru import logger

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from chess_src.scraping.twic_downloader import TWICDownloader
from chess_src.scraping.pgn_importer import PGNImporter
from chess_src.database import DEFAULT_DB


def get_candidate_names():
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)

    names = [p["name"] for p in config["players"]]
    variations = set(names)

    for name in names:
        parts = name.split(" ")
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            # "Fabiano Caruana" -> "Caruana, Fabiano"
            variations.add(f"{last}, {first}")
            # "Caruana,F"
            variations.add(f"{last},{first[0]}")
            # "Caruana, F"
            variations.add(f"{last}, {first[0]}")

            # Special case for "R. Praggnanandhaa"
            if "Praggnanandhaa" in name:
                variations.add("Praggnanandhaa,R")
                variations.add("Praggnanandhaa, R")
                variations.add("Praggnanandhaa,Rameshbabu")
                variations.add("Rameshbabu,P")

    return list(variations)


def enrich_main(start_issue=1365, end_issue=1645, batch_size=20):
    candidates = get_candidate_names()
    logger.info(f"Starting enrichment for Candidates: {candidates}")

    downloader = TWICDownloader(
        download_dir=str(PROJECT_ROOT / "data" / "raw" / "twic")
    )
    importer = PGNImporter(db_path=DEFAULT_DB)

    current = start_issue
    while current <= end_issue:
        batch_end = min(current + batch_size - 1, end_issue)
        logger.info(f"Processing batch: TWIC {current} to {batch_end}")

        pgn_paths = downloader.download_range(current, batch_end)

        for pgn_path in pgn_paths:
            importer.import_file(pgn_path, player_whitelist=candidates)
            # Optional: remove PGN to save space after import
            # os.remove(pgn_path)

        current += batch_size
        logger.success(f"Completed batch up to {batch_end}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1365)
    parser.add_argument("--end", type=int, default=1645)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    enrich_main(args.start, args.end, args.batch)
