from src.scraping.oddspapi import extract_home_away_odds, is_usable_secret


def test_is_usable_secret_rejects_placeholders() -> None:
    assert not is_usable_secret("")
    assert not is_usable_secret("your_oddspapi_key_here")
    assert not is_usable_secret("placeholder")
    assert is_usable_secret("realistic-token")


def test_extract_home_away_odds_keeps_best_prices() -> None:
    payload = {
        "bookmakerOdds": {
            "book_a": {
                "markets": {
                    "match": {
                        "outcomes": {
                            "home": {"players": {"p1": [{"bookmakerOutcomeId": "home", "price": "1.80"}]}},
                            "away": {"players": {"p2": [{"bookmakerOutcomeId": "away", "price": "2.10"}]}},
                        }
                    }
                }
            },
            "book_b": {
                "markets": {
                    "match": {
                        "outcomes": {
                            "home": {"players": {"p1": [{"bookmakerOutcomeId": "home", "price": "1.95"}]}},
                            "away": {"players": {"p2": [{"bookmakerOutcomeId": "away", "price": "2.00"}]}},
                        }
                    }
                }
            },
        }
    }

    home, away, source = extract_home_away_odds(payload)

    assert home == 1.95
    assert away == 2.10
    assert source == "oddspapi_best"
