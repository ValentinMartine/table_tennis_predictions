import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from ..database.db import engine
from .elo import expected_score

def build_single_match_features(p1_id: int, p2_id: int, 
                                p1_ittf: int = 9999, p2_ittf: int = 9999,
                                p1_wtt: int = 9999, p2_wtt: int = 9999,
                                odds_p1: float = None, odds_p2: float = None) -> dict:
    """
    Construit TOUTES les features pour un match entre P1 et P2 à l'instant T.
    """
    # 1. Elo & Expérience
    def _get_elo_state(pid):
        q = text("SELECT rating, matches_played FROM elo_ratings WHERE player_id = :pid ORDER BY computed_at DESC LIMIT 1")
        with engine.connect() as conn:
            r = conn.execute(q, {"pid": pid}).fetchone()
        return (float(r[0]), int(r[1])) if r else (1500.0, 0)

    elo1, n1 = _get_elo_state(p1_id)
    elo2, n2 = _get_elo_state(p2_id)
    elo_diff = elo1 - elo2
    elo_prob = expected_score(elo1, elo2)

    # 1b. Style/Hand (Nouveau)
    def _get_player_style(pid):
        q = text("SELECT hand, style, grip FROM players WHERE id = :pid")
        with engine.connect() as conn:
            r = conn.execute(q, {"pid": pid}).fetchone()
        return (r[0], r[1], r[2]) if r else (None, None, None)

    h1, s1_type, g1 = _get_player_style(p1_id)
    h2, s2_type, g2 = _get_player_style(p2_id)

    # 2. H2H
    def _get_h2h(pid1, pid2):
        q = text("""
            SELECT winner, player1_id FROM matches 
            WHERE ((player1_id = :p1 AND player2_id = :p2) OR (player1_id = :p2 AND player2_id = :p1))
            AND is_walkover = 0 ORDER BY played_at DESC LIMIT 20
        """)
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"p1": pid1, "p2": pid2})
        if df.empty:
            return 0, 0.5
        p1_wins = sum(1 for _, r in df.iterrows() if (r["player1_id"] == pid1 and r["winner"] == 1) or (r["player1_id"] == pid2 and r["winner"] == 2))
        return len(df), p1_wins / len(df)

    h2h_n, h2h_wr = _get_h2h(p1_id, p2_id)

    # 3. Forme (basé sur 10 derniers matchs)
    def _get_form(pid):
        q = text("""
            SELECT winner, player1_id FROM matches 
            WHERE (player1_id = :pid OR player2_id = :pid) AND is_walkover = 0 
            ORDER BY played_at DESC LIMIT 10
        """)
        with engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"pid": pid})
        if df.empty: return 0.5
        # Pondération exponentielle simple (0.85^i)
        wins = [1 if (r["player1_id"] == pid and r["winner"] == 1) or (r["player1_id"] != pid and r["winner"] == 2) else 0 for _, r in df.iterrows()]
        weights = [0.85**i for i in range(len(wins))]
        return sum(w*x for w,x in zip(weights, wins)) / sum(weights)

    f1 = _get_form(p1_id)
    f2 = _get_form(p2_id)

    # 4. Trajectoire de classement (Velocity/Stability)
    def _get_trajectory(pid):
        q = text("SELECT rank FROM ittf_rankings WHERE player_id = :pid ORDER BY snapshot_date DESC LIMIT 10")
        with engine.connect() as conn:
            res = conn.execute(q, {"pid": pid}).fetchall()
        if len(res) < 2: return 0.0, 0.0
        ranks = [r[0] for r in res]
        velocity = float(np.clip(ranks[-1] - ranks[0], -200, 200)) # Progression = rang de départ - rang actuel
        stability = float(np.std(ranks))
        return velocity, stability

    v1, s1 = _get_trajectory(p1_id)
    v2, s2 = _get_trajectory(p2_id)

    # 5. Age
    def _get_age(pid):
        q = text("SELECT date_of_birth FROM players WHERE id = :pid")
        with engine.connect() as conn:
            r = conn.execute(q, {"pid": pid}).fetchone()
        if r and r[0]:
            dob = pd.to_datetime(r[0])
            return (datetime.now() - dob).days / 365.25
        return 25.0

    a1 = _get_age(p1_id)
    a2 = _get_age(p2_id)

    # 6. Market (Odds)
    has_odds = 1 if (odds_p1 and odds_p2 and odds_p1 > 1 and odds_p2 > 1) else 0
    if has_odds:
        # Probabilité implicite normalisée (sans la marge du bookmaker)
        raw_p1 = 1 / odds_p1
        raw_p2 = 1 / odds_p2
        implied_p1 = raw_p1 / (raw_p1 + raw_p2)
    else:
        implied_p1 = elo_prob

    # Feature Map
    return {
        "elo_diff": elo_diff,
        "elo_win_prob_p1": elo_prob,
        "elo_intl_diff": elo_diff, 
        "elo_intl_win_prob_p1": elo_prob,
        "h2h_matches": h2h_n,
        "h2h_winrate_p1": h2h_wr,
        "h2h_recent_winrate_p1": h2h_wr,
        "form_p1": f1,
        "form_p2": f2,
        "form_diff": f1 - f2,
        "avg_sets_p1": 2.0,
        "avg_sets_p2": 2.0,
        "rest_hours_p1": 48.0,
        "rest_hours_p2": 48.0,
        "ittf_rank_p1": p1_ittf,
        "ittf_rank_p2": p2_ittf,
        "rank_diff": p1_ittf - p2_ittf,
        
        # Log-rank pour écraser l'échelle au sommet (1 vs 10 >> 500 vs 510)
        "log_ittf_rank_p1": np.log1p(p1_ittf),
        "log_ittf_rank_p2": np.log1p(p2_ittf),
        "log_rank_diff": np.log1p(p1_ittf) - np.log1p(p2_ittf),
        
        "wtt_rank_p1": p1_wtt,
        "wtt_rank_p2": p2_wtt,
        "wtt_rank_diff": p1_wtt - p2_wtt,
        "log_wtt_rank_p1": np.log1p(p1_wtt),
        "log_wtt_rank_p2": np.log1p(p2_wtt),
        
        "rank_velocity_p1": v1,
        "rank_velocity_p2": v2,
        "rank_stability_p1": s1,
        "rank_stability_p2": s2,
        "age_p1": a1,
        "age_p2": a2,
        "age_diff": a1 - a2,
        
        # Style features
        "is_p1_lefty": 1 if h1 == "L" else 0,
        "is_p2_lefty": 1 if h2 == "L" else 0,
        "is_opposite_hand": 1 if (h1 and h2 and h1 != h2) else 0,
        
        # Nouvelles features cruciales pour éviter le biais des "inconnus"
        "is_p1_unknown": 1 if (p1_ittf >= 9999 and n1 < 10) else 0,
        "is_p2_unknown": 1 if (p2_ittf >= 9999 and n2 < 10) else 0,
        "matches_played_p1": n1,
        "matches_played_p2": n2,
        
        "has_odds": has_odds,
        "implied_prob_p1": implied_p1,
        
        # Metadata
        "_elo_p1": elo1,
        "_elo_p2": elo2,
        "_form_5_p1": f1,
        "_form_5_p2": f2,
    }
