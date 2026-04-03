import pandas as pd
import yaml
import sys
import os

# Append the current directory for imports to work
sys.path.append(os.path.abspath("."))

from chess_src.features.pipeline import ChessFeaturePipeline
from chess_src.database import init_db


def test_pipeline():
    # Load config
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize DB (test)
    init_db("data/test_chess.db")

    # Mock data
    players = config["players"]
    p1, p2 = players[0]["fide_id"], players[1]["fide_id"]

    data = [
        {
            "white_id": p1,
            "black_id": p2,
            "result": 1.0,
            "round": 1,
            "tournament": "Candidates 2026",
            "played_at": "2026-04-01",
        },
        {
            "white_id": p2,
            "black_id": p1,
            "result": 0.5,
            "round": 2,
            "tournament": "Candidates 2026",
            "played_at": "2026-04-02",
        },
    ]
    df = pd.DataFrame(data)

    # Run pipeline
    pipeline = ChessFeaturePipeline(config)
    df_processed = pipeline.process(df)

    print("Processed DataFrame columns:")
    print(df_processed.columns.tolist())
    print("\nSample values:")
    print(
        df_processed[["white_elo", "black_elo", "h2h_points_white", "form_diff"]].head()
    )


if __name__ == "__main__":
    test_pipeline()
