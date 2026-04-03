import pandas as pd
from loguru import logger


class ChessDataFetcher:
    def __init__(self, players_config: list):
        self.players = players_config

    def fetch_live_ratings(self) -> pd.DataFrame:
        """
        Skeleton for fetching live ratings from 2700chess.com.
        In a real scenario, this would use BeautifulSoup to parse the main table.
        """
        logger.info("Fetching live ratings from 2700chess.com...")
        # Since we are in simulation, we return the ratings from config
        data = []
        for p in self.players:
            data.append(
                {
                    "name": p["name"],
                    "fide_id": p["fide_id"],
                    "live_rating": p["rating_april_2006"],
                }
            )
        return pd.DataFrame(data)

    def fetch_candidates_pairings(self) -> pd.DataFrame:
        """Official 14-round pairings for FIDE Candidates 2026 Paphos."""
        logger.info("Fetching official 2026 tournament schedule...")

        # Mapping Names to FIDE IDs for readability
        P = {p["name"]: p["fide_id"] for p in self.players}

        # Official Rounds 1-14
        schedule = [
            # Round 1
            {
                "round": 1,
                "white_id": P["Javokhir Sindarov"],
                "black_id": P["Andrey Esipenko"],
            },
            {"round": 1, "white_id": P["Matthias Bluebaum"], "black_id": P["Wei Yi"]},
            {
                "round": 1,
                "white_id": P["R. Praggnanandhaa"],
                "black_id": P["Anish Giri"],
            },
            {
                "round": 1,
                "white_id": P["Fabiano Caruana"],
                "black_id": P["Hikaru Nakamura"],
            },
            # Round 2
            {
                "round": 2,
                "white_id": P["Andrey Esipenko"],
                "black_id": P["Hikaru Nakamura"],
            },
            {"round": 2, "white_id": P["Anish Giri"], "black_id": P["Fabiano Caruana"]},
            {"round": 2, "white_id": P["Wei Yi"], "black_id": P["R. Praggnanandhaa"]},
            {
                "round": 2,
                "white_id": P["Javokhir Sindarov"],
                "black_id": P["Matthias Bluebaum"],
            },
            # Round 3
            {
                "round": 3,
                "white_id": P["Matthias Bluebaum"],
                "black_id": P["Andrey Esipenko"],
            },
            {
                "round": 3,
                "white_id": P["R. Praggnanandhaa"],
                "black_id": P["Javokhir Sindarov"],
            },
            {"round": 3, "white_id": P["Fabiano Caruana"], "black_id": P["Wei Yi"]},
            {"round": 3, "white_id": P["Hikaru Nakamura"], "black_id": P["Anish Giri"]},
            # Round 4
            {"round": 4, "white_id": P["Andrey Esipenko"], "black_id": P["Anish Giri"]},
            {"round": 4, "white_id": P["Wei Yi"], "black_id": P["Hikaru Nakamura"]},
            {
                "round": 4,
                "white_id": P["Javokhir Sindarov"],
                "black_id": P["Fabiano Caruana"],
            },
            {
                "round": 4,
                "white_id": P["Matthias Bluebaum"],
                "black_id": P["R. Praggnanandhaa"],
            },
            # Round 5
            {
                "round": 5,
                "white_id": P["R. Praggnanandhaa"],
                "black_id": P["Andrey Esipenko"],
            },
            {
                "round": 5,
                "white_id": P["Fabiano Caruana"],
                "black_id": P["Matthias Bluebaum"],
            },
            {
                "round": 5,
                "white_id": P["Hikaru Nakamura"],
                "black_id": P["Javokhir Sindarov"],
            },
            {"round": 5, "white_id": P["Anish Giri"], "black_id": P["Wei Yi"]},
            # Round 6
            {
                "round": 6,
                "white_id": P["Fabiano Caruana"],
                "black_id": P["Andrey Esipenko"],
            },
            {
                "round": 6,
                "white_id": P["Hikaru Nakamura"],
                "black_id": P["R. Praggnanandhaa"],
            },
            {
                "round": 6,
                "white_id": P["Anish Giri"],
                "black_id": P["Matthias Bluebaum"],
            },
            {"round": 6, "white_id": P["Wei Yi"], "black_id": P["Javokhir Sindarov"]},
            # Round 7
            {"round": 7, "white_id": P["Andrey Esipenko"], "black_id": P["Wei Yi"]},
            {
                "round": 7,
                "white_id": P["Javokhir Sindarov"],
                "black_id": P["Anish Giri"],
            },
            {
                "round": 7,
                "white_id": P["Matthias Bluebaum"],
                "black_id": P["Hikaru Nakamura"],
            },
            {
                "round": 7,
                "white_id": P["R. Praggnanandhaa"],
                "black_id": P["Fabiano Caruana"],
            },
            # Round 8
            {
                "round": 8,
                "white_id": P["Andrey Esipenko"],
                "black_id": P["Javokhir Sindarov"],
            },
            {"round": 8, "white_id": P["Wei Yi"], "black_id": P["Matthias Bluebaum"]},
            {
                "round": 8,
                "white_id": P["Anish Giri"],
                "black_id": P["R. Praggnanandhaa"],
            },
            {
                "round": 8,
                "white_id": P["Hikaru Nakamura"],
                "black_id": P["Fabiano Caruana"],
            },
            # Round 9
            {
                "round": 9,
                "white_id": P["Hikaru Nakamura"],
                "black_id": P["Andrey Esipenko"],
            },
            {"round": 9, "white_id": P["Fabiano Caruana"], "black_id": P["Anish Giri"]},
            {"round": 9, "white_id": P["R. Praggnanandhaa"], "black_id": P["Wei Yi"]},
            {
                "round": 9,
                "white_id": P["Matthias Bluebaum"],
                "black_id": P["Javokhir Sindarov"],
            },
            # Round 10
            {
                "round": 10,
                "white_id": P["Andrey Esipenko"],
                "black_id": P["Matthias Bluebaum"],
            },
            {
                "round": 10,
                "white_id": P["Javokhir Sindarov"],
                "black_id": P["R. Praggnanandhaa"],
            },
            {"round": 10, "white_id": P["Wei Yi"], "black_id": P["Fabiano Caruana"]},
            {
                "round": 10,
                "white_id": P["Anish Giri"],
                "black_id": P["Hikaru Nakamura"],
            },
            # Round 11
            {
                "round": 11,
                "white_id": P["Anish Giri"],
                "black_id": P["Andrey Esipenko"],
            },
            {"round": 11, "white_id": P["Hikaru Nakamura"], "black_id": P["Wei Yi"]},
            {
                "round": 11,
                "white_id": P["Fabiano Caruana"],
                "black_id": P["Javokhir Sindarov"],
            },
            {
                "round": 11,
                "white_id": P["R. Praggnanandhaa"],
                "black_id": P["Matthias Bluebaum"],
            },
            # Round 12
            {
                "round": 12,
                "white_id": P["Andrey Esipenko"],
                "black_id": P["R. Praggnanandhaa"],
            },
            {
                "round": 12,
                "white_id": P["Matthias Bluebaum"],
                "black_id": P["Fabiano Caruana"],
            },
            {
                "round": 12,
                "white_id": P["Javokhir Sindarov"],
                "black_id": P["Hikaru Nakamura"],
            },
            {"round": 12, "white_id": P["Wei Yi"], "black_id": P["Anish Giri"]},
            # Round 13
            {
                "round": 13,
                "white_id": P["Andrey Esipenko"],
                "black_id": P["Fabiano Caruana"],
            },
            {
                "round": 13,
                "white_id": P["R. Praggnanandhaa"],
                "black_id": P["Hikaru Nakamura"],
            },
            {
                "round": 13,
                "white_id": P["Matthias Bluebaum"],
                "black_id": P["Anish Giri"],
            },
            {"round": 13, "white_id": P["Javokhir Sindarov"], "black_id": P["Wei Yi"]},
            # Round 14
            {"round": 14, "white_id": P["Wei Yi"], "black_id": P["Andrey Esipenko"]},
            {
                "round": 14,
                "white_id": P["Anish Giri"],
                "black_id": P["Javokhir Sindarov"],
            },
            {
                "round": 14,
                "white_id": P["Hikaru Nakamura"],
                "black_id": P["Matthias Bluebaum"],
            },
            {
                "round": 14,
                "white_id": P["Fabiano Caruana"],
                "black_id": P["R. Praggnanandhaa"],
            },
        ]

        return pd.DataFrame(schedule)


if __name__ == "__main__":
    # Test script
    import yaml

    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    fetcher = ChessDataFetcher(config["players"])
    print(fetcher.fetch_candidates_pairings().head())
