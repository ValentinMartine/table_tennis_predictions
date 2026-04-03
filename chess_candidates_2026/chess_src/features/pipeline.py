from .elo import ChessEloCalculator
from .h2h import ChessH2HCalculator
from .form import ChessFormCalculator
from .context import ChessContextCalculator
import pandas as pd
import numpy as np


class ChessFeaturePipeline:
    def __init__(self, config=None, db_path=None):
        self.config = config or {}
        # Initial ratings from config (8 candidates)
        players_config = self.config.get("players", [])
        initial_ratings = {
            p["fide_id"]: p["rating_april_2006"]
            for p in players_config
            if "fide_id" in p
        }

        # Also load ratings for all other players from DB (TWIC opponents etc.)
        if db_path:
            import sqlite3

            conn = sqlite3.connect(db_path)
            for pid, rating in conn.execute(
                "SELECT id, rating_initial FROM players WHERE rating_initial IS NOT NULL"
            ):
                initial_ratings.setdefault(int(pid), int(rating))
            conn.close()

        self.elo_calc = ChessEloCalculator(
            initial_rating=self.config.get("features", {})
            .get("elo", {})
            .get("initial_rating", 1500),
            k=self.config.get("features", {}).get("elo", {}).get("k_factor", 20),
            initial_ratings_dict=initial_ratings,
        )
        self.h2h_calc = ChessH2HCalculator(
            min_matches=self.config.get("features", {})
            .get("h2h", {})
            .get("min_matches", 0),
            decay_rate=self.config.get("features", {})
            .get("h2h", {})
            .get("decay_rate", 0.0005),
        )
        self.form_calc = ChessFormCalculator(window=10)
        self.context_calc = ChessContextCalculator(total_rounds=14)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full feature engineering pipeline.
        Dataframe must have: [white_id, black_id, played_at, round, tournament]
        'result' is optional — if absent or NaN, rows are kept for prediction.
        """
        df = df.copy()

        # Add result column if missing (prediction mode)
        if "result" not in df.columns:
            df["result"] = np.nan

        df["result"] = pd.to_numeric(df["result"], errors="coerce")
        df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce")
        df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(1)

        # Only drop rows missing positional data — keep NaN results (prediction rows)
        df = df.dropna(subset=["played_at", "white_id", "black_id"])

        # Ensure IDs are ints
        df["white_id"] = df["white_id"].astype(int)
        df["black_id"] = df["black_id"].astype(int)

        # 1. Elo (Order matters: Elo first)
        df = self.elo_calc.compute(df)

        # 1b. Intra-tournament Elo delta (change in Elo since first game of tournament)
        df = df.sort_values(["tournament", "played_at"]).reset_index(drop=True)
        white_app = df[["tournament", "played_at", "white_id", "white_elo"]].rename(
            columns={"white_id": "player_id", "white_elo": "elo"}
        )
        black_app = df[["tournament", "played_at", "black_id", "black_elo"]].rename(
            columns={"black_id": "player_id", "black_elo": "elo"}
        )
        first_elos = (
            pd.concat([white_app, black_app])
            .sort_values(["tournament", "player_id", "played_at"])
            .groupby(["tournament", "player_id"])["elo"]
            .first()
            .reset_index()
            .rename(columns={"elo": "elo_start"})
        )
        df = df.merge(
            first_elos.rename(
                columns={"player_id": "white_id", "elo_start": "_w_elo_start"}
            ),
            on=["tournament", "white_id"],
            how="left",
        )
        df = df.merge(
            first_elos.rename(
                columns={"player_id": "black_id", "elo_start": "_b_elo_start"}
            ),
            on=["tournament", "black_id"],
            how="left",
        )
        df["white_elo_intra_delta"] = df["white_elo"] - df["_w_elo_start"]
        df["black_elo_intra_delta"] = df["black_elo"] - df["_b_elo_start"]
        df = df.drop(columns=["_w_elo_start", "_b_elo_start"])

        # 2. H2H
        df = self.h2h_calc.compute(df)

        # 3. Form (Depends on Elo)
        df = self.form_calc.compute(df)

        # 4. Context (Tournament standings + momentum + color balance)
        df = self.context_calc.compute(df)

        # Final cleanup: fill feature NaNs with 0, but preserve result column (NaN = upcoming match)
        result_col = df["result"].copy() if "result" in df.columns else None
        df = df.fillna(0)
        if result_col is not None:
            df["result"] = result_col
        return df
