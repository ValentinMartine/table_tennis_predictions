import pandas as pd
from loguru import logger
from datetime import datetime, timedelta

class TournamentSimulator:
    def __init__(self, model, player_map):
        self.model = model
        self.player_map = player_map

    def _get_player_info(self, name):
        """Helper to get player info from name."""
        from scripts.predict_upcoming import _match_player
        pid = _match_player(name, self.player_map)
        if pid is None:
            return None
        return self.player_map[self.player_map["id"] == pid].iloc[0]

    def _predict_match(self, p1_name, p2_name):
        """Predict win probability for p1 vs p2."""
        from scripts.predict_upcoming import build_features_for_match
        p1 = self._get_player_info(p1_name)
        p2 = self._get_player_info(p2_name)
        
        if p1 is None or p2 is None:
            return 0.5 # Unknown
            
        features = build_features_for_match(
            int(p1["id"]), int(p2["id"]),
            int(p1["ittf_rank"]), int(p2["ittf_rank"]),
            int(p1["wtt_rank"]), int(p2["wtt_rank"])
        )
        prob_p1 = float(self.model.predict_proba(features)[0])
        return prob_p1

    def simulate_world_cup_groups(self, all_matches):
        """
        Simulate ITTF World Cup groups and identify the 1st of each group.
        Handles multiple competitions (Men, Women).
        """
        # Group matches by tourney
        competitions = {}
        for m in all_matches:
            if "World Cup" in m["tournament"] and m["group_name"]:
                t_base = m["tournament"].split(", Group")[0]
                if t_base not in competitions:
                    competitions[t_base] = {}
                
                g_key = m["group_name"]
                if g_key not in competitions[t_base]:
                    competitions[t_base][g_key] = []
                competitions[t_base][g_key].append(m)

        all_leaders = {}
        for t_name, groups in competitions.items():
            leaders = {}
            for g_name, matches in groups.items():
                standings = {}
                for m in matches:
                    p1, p2 = m["p1_name"], m["p2_name"]
                    if p1 not in standings: standings[p1] = 0
                    if p2 not in standings: standings[p2] = 0
                    prob1 = self._predict_match(p1, p2)
                    if prob1 >= 0.5:
                        standings[p1] += 2; standings[p2] += 1
                    else:
                        standings[p2] += 2; standings[p1] += 1
                
                sorted_standings = sorted(standings.items(), key=lambda x: x[1], reverse=True)
                if sorted_standings:
                    leaders[g_name] = sorted_standings[0][0]
            
            all_leaders[t_name] = leaders
        
        return all_leaders

    def project_knockout_stage(self, all_leaders):
        """
        Produce projections for all discovered tournaments.
        """
        all_projections = []
        for t_name, leaders in all_leaders.items():
            bracket_order = [1, 16, 8, 9, 4, 13, 5, 12, 2, 15, 7, 10, 3, 14, 6, 11]
            r16_players = []
            for g_num in bracket_order:
                g_key = f"Group {g_num}"
                r16_players.append(leaders.get(g_key, f"Vainqueur {g_key}"))

            # Simulation levels: R16 -> QF -> SF -> F
            levels = [
                ("8ème de finale", r16_players, 16),
                ("Quart de finale", None, 8),
                ("Demi-finale", None, 4),
                ("Finale", None, 2)
            ]
            
            current_players = r16_players
            for round_label, players, count in levels:
                if players is None: # Infer from winners of previous level
                    # This logic is handled iteratively below
                    pass
                
                next_round_players = []
                for i in range(0, len(current_players), 2):
                    p1, p2 = current_players[i], current_players[i+1]
                    p1_str, p2_str = str(p1), str(p2)
                    prob1 = self._predict_match(p1_str, p2_str) if "Vainqueur" not in p1_str and "Vainqueur" not in p2_str else 0.5
                    winner = p1 if prob1 >= 0.5 else p2
                    next_round_players.append(winner)
                    all_projections.append({
                        "tournoi": t_name,
                        "round": round_label,
                        "p1": p1, "p2": p2, "prob_p1": prob1, "projected_winner": winner
                    })
                current_players = next_round_players
        
        return all_projections

    def simulate_generic_bracket(self, all_matches):
        """
        Simulate a standard KO bracket where 'Winner of Match X' appears.
        (Work in progress, needs match-to-id mapping)
        """
        # Placeholder for now, as it requires mapping Match IDs from Sofascore Draws
        return []
