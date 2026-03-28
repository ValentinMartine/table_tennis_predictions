"""
Matching entre joueurs scrapés et joueurs ITTF.

Les joueurs scrapés (Sofascore etc.) ont : name="Hugo Calderano", country="BR"
Les joueurs ITTF ont                     : name="CALDERANO Hugo", country="BRA", gender, ittf_id, date_of_birth

Ce script :
  1. Normalise les noms des deux côtés (lowercase, sans accents, tokens inversés pour ITTF)
  2. Matche par nom exact puis fuzzy (score ≥ 85) + pays compatible
  3. Fusionne : copie ittf_id / gender / date_of_birth dans le joueur scrapé,
               re-linke les ittf_rankings, supprime le doublon ITTF

Usage :
    python scripts/match_players.py [--dry-run] [--min-score 85]
"""
import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from loguru import logger
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text as sa_text

from src.database.db import engine, get_session
from src.database.models import Player

# Mapping ISO2 → ISO3 (principales associations TT)
ISO2_TO_ISO3 = {
    "AF": "AFG", "AL": "ALB", "DZ": "ALG", "AD": "AND", "AO": "ANG",
    "AG": "ANT", "AR": "ARG", "AM": "ARM", "AU": "AUS", "AT": "AUT",
    "AZ": "AZE", "BS": "BAH", "BH": "BRN", "BD": "BAN", "BB": "BAR",
    "BY": "BLR", "BE": "BEL", "BZ": "BIZ", "BJ": "BEN", "BT": "BHU",
    "BO": "BOL", "BA": "BIH", "BW": "BOT", "BR": "BRA", "BN": "BRU",
    "BG": "BUL", "BF": "BUR", "BI": "BDI", "CV": "CPV", "KH": "CAM",
    "CM": "CMR", "CA": "CAN", "CF": "CAF", "TD": "CHA", "CL": "CHI",
    "CN": "CHN", "CO": "COL", "KM": "COM", "CG": "CGO", "CD": "COD",
    "CR": "CRC", "HR": "CRO", "CU": "CUB", "CY": "CYP", "CZ": "CZE",
    "DK": "DEN", "DJ": "DJI", "DM": "DMA", "DO": "DOM", "EC": "ECU",
    "EG": "EGY", "SV": "ESA", "GQ": "GEQ", "ER": "ERI", "EE": "EST",
    "ET": "ETH", "FJ": "FIJ", "FI": "FIN", "FR": "FRA", "GA": "GAB",
    "GM": "GAM", "GE": "GEO", "DE": "GER", "GH": "GHA", "GR": "GRE",
    "GD": "GRN", "GT": "GUA", "GN": "GUI", "GW": "GBS", "GY": "GUY",
    "HT": "HAI", "HN": "HON", "HK": "HKG", "HU": "HUN", "IS": "ISL",
    "IN": "IND", "ID": "INA", "IR": "IRI", "IQ": "IRQ", "IE": "IRL",
    "IL": "ISR", "IT": "ITA", "JM": "JAM", "JP": "JPN", "JO": "JOR",
    "KZ": "KAZ", "KE": "KEN", "KI": "KIR", "KP": "PRK", "KR": "KOR",
    "KW": "KUW", "KG": "KGZ", "LA": "LAO", "LV": "LAT", "LB": "LIB",
    "LS": "LES", "LR": "LBR", "LY": "LBA", "LI": "LIE", "LT": "LTU",
    "LU": "LUX", "MK": "MKD", "MG": "MAD", "MW": "MAW", "MY": "MAS",
    "MV": "MDV", "ML": "MLI", "MT": "MLT", "MH": "MHL", "MR": "MTN",
    "MU": "MRI", "MX": "MEX", "FM": "FSM", "MD": "MDA", "MC": "MON",
    "MN": "MGL", "ME": "MNE", "MA": "MAR", "MZ": "MOZ", "MM": "MYA",
    "NA": "NAM", "NR": "NRU", "NP": "NEP", "NL": "NED", "NZ": "NZL",
    "NI": "NCA", "NE": "NIG", "NG": "NGR", "NO": "NOR", "OM": "OMA",
    "PK": "PAK", "PW": "PLW", "PA": "PAN", "PG": "PNG", "PY": "PAR",
    "PE": "PER", "PH": "PHI", "PL": "POL", "PT": "POR", "QA": "QAT",
    "RO": "ROU", "RU": "RUS", "RW": "RWA", "KN": "SKN", "LC": "LCA",
    "VC": "VIN", "WS": "SAM", "SM": "SMR", "ST": "STP", "SA": "KSA",
    "SN": "SEN", "RS": "SRB", "SC": "SEY", "SL": "SLE", "SG": "SGP",
    "SK": "SVK", "SI": "SLO", "SB": "SOL", "SO": "SOM", "ZA": "RSA",
    "SS": "SSD", "ES": "ESP", "LK": "SRI", "SD": "SUD", "SR": "SUR",
    "SZ": "SWZ", "SE": "SWE", "CH": "SUI", "SY": "SYR", "TW": "TPE",
    "TJ": "TJK", "TZ": "TAN", "TH": "THA", "TL": "TLS", "TG": "TOG",
    "TO": "TGA", "TT": "TTO", "TN": "TUN", "TR": "TUR", "TM": "TKM",
    "TV": "TUV", "UG": "UGA", "UA": "UKR", "AE": "UAE", "GB": "GBR",
    "US": "USA", "UY": "URU", "UZ": "UZB", "VU": "VAN", "VE": "VEN",
    "VN": "VIE", "YE": "YEM", "ZM": "ZAM", "ZW": "ZIM",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_name(name: str, is_ittf: bool = False) -> str:
    """Normalise un nom pour comparaison."""
    name = _strip_accents(name).lower().strip()
    # Enlève la ponctuation sauf espaces
    name = "".join(c if c.isalnum() or c == " " else " " for c in name)
    tokens = name.split()
    if is_ittf and tokens:
        # Format ITTF : LASTNAME Firstname → inverser
        tokens = tokens[1:] + [tokens[0]]
    return " ".join(tokens)


def _country_compatible(c1: str | None, c2: str | None) -> bool:
    """Vérifie si deux codes pays sont compatibles (ISO2 vs ISO3)."""
    if not c1 or not c2:
        return True  # données manquantes → on ne rejette pas
    c1, c2 = c1.upper().strip(), c2.upper().strip()
    if c1 == c2:
        return True
    # ISO2 → ISO3
    if len(c1) == 2:
        c1_iso3 = ISO2_TO_ISO3.get(c1, c1)
        if c1_iso3 == c2:
            return True
    if len(c2) == 2:
        c2_iso3 = ISO2_TO_ISO3.get(c2, c2)
        if c2_iso3 == c1:
            return True
    return False


def _is_doubles(name: str) -> bool:
    return "/" in name or " & " in name


def load_candidates(conn) -> pd.DataFrame:
    """Charge tous les joueurs ITTF (avec ittf_id) avec leurs infos."""
    df = pd.read_sql(sa_text("""
        SELECT id, name, country, gender, ittf_id, date_of_birth
        FROM players
        WHERE ittf_id IS NOT NULL
    """), conn)
    df["name_norm"] = df["name"].apply(lambda n: _normalize_name(n, is_ittf=True))
    return df


def load_scraped(conn) -> pd.DataFrame:
    """Charge les joueurs scrapés (sans ittf_id, ayant des matchs)."""
    df = pd.read_sql(sa_text("""
        SELECT DISTINCT p.id, p.name, p.country
        FROM players p
        JOIN matches m ON p.id = m.player1_id OR p.id = m.player2_id
        WHERE p.ittf_id IS NULL
    """), conn)
    df = df[~df["name"].apply(_is_doubles)]
    df["name_norm"] = df["name"].apply(lambda n: _normalize_name(n, is_ittf=False))
    return df


def find_best_match(
    scraped_row: pd.Series,
    ittf_df: pd.DataFrame,
    min_score: int,
) -> tuple[int | None, int]:
    """
    Cherche le meilleur candidat ITTF pour un joueur scrapé.
    Retourne (ittf_player_id, score) ou (None, 0).
    """
    name_norm = scraped_row["name_norm"]
    country = scraped_row["country"]

    # Filtre par pays d'abord pour réduire le nb de comparaisons
    candidates = ittf_df[
        ittf_df["country"].apply(lambda c: _country_compatible(country, c))
    ]

    if candidates.empty:
        candidates = ittf_df  # fallback sans filtre pays

    best_id, best_score = None, 0
    for _, cand in candidates.iterrows():
        score = fuzz.token_sort_ratio(name_norm, cand["name_norm"])
        if score > best_score:
            best_score = score
            best_id = int(cand["id"])

    if best_score >= min_score:
        return best_id, best_score
    return None, best_score


def merge_players(scraped_id: int, ittf_id: int, dry_run: bool) -> None:
    """
    Fusionne ittf_player dans scraped_player :
      - copie ittf_id, gender, date_of_birth
      - re-linke les ittf_rankings
      - supprime l'enregistrement ITTF-only
    """
    if dry_run:
        return

    with engine.connect() as conn:
        # Récupère les infos ITTF
        row = conn.execute(sa_text(
            "SELECT ittf_id, gender, date_of_birth FROM players WHERE id = :id"
        ), {"id": ittf_id}).fetchone()

        if not row:
            return

        # 1. Re-linke les ittf_rankings vers le joueur scrapé
        conn.execute(sa_text("""
            UPDATE ittf_rankings
            SET player_id = :scraped_id
            WHERE player_id = :ittf_id
        """), {"scraped_id": scraped_id, "ittf_id": ittf_id})

        # 2. Vide l'ittf_id du joueur ITTF pour libérer la contrainte UNIQUE
        conn.execute(sa_text(
            "UPDATE players SET ittf_id = NULL WHERE id = :id"
        ), {"id": ittf_id})

        # 3. Copie ittf_id / gender / dob dans le joueur scrapé
        conn.execute(sa_text("""
            UPDATE players
            SET ittf_id       = :ittf_id,
                gender        = COALESCE(gender, :gender),
                date_of_birth = COALESCE(date_of_birth, :dob)
            WHERE id = :scraped_id
        """), {
            "ittf_id": row[0],
            "gender": row[1],
            "dob": row[2],
            "scraped_id": scraped_id,
        })

        # 4. Supprime le doublon ITTF-only
        conn.execute(sa_text("DELETE FROM players WHERE id = :id"), {"id": ittf_id})
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Match scraped players with ITTF data")
    parser.add_argument("--dry-run", action="store_true", help="Simulation sans modifier la DB")
    parser.add_argument("--min-score", type=int, default=85,
                        help="Score fuzzy minimum (0-100, défaut: 85)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if args.dry_run:
        logger.info("MODE DRY-RUN — aucune modification en DB")

    with engine.connect() as conn:
        logger.info("Chargement des joueurs...")
        ittf_df = load_candidates(conn)
        scraped_df = load_scraped(conn)

    logger.info(f"  Joueurs ITTF    : {len(ittf_df)}")
    logger.info(f"  Joueurs scrapés : {len(scraped_df)}")

    matched, unmatched = 0, 0
    results = []

    for _, row in scraped_df.iterrows():
        best_id, score = find_best_match(row, ittf_df, args.min_score)

        if best_id is not None:
            ittf_row = ittf_df[ittf_df["id"] == best_id].iloc[0]
            results.append({
                "scraped_name": row["name"],
                "ittf_name": ittf_row["name"],
                "score": score,
                "gender": ittf_row["gender"],
                "ittf_id": ittf_row["ittf_id"],
            })
            merge_players(int(row["id"]), best_id, args.dry_run)
            # Retire le candidat pour éviter les doubles attributions
            ittf_df = ittf_df[ittf_df["id"] != best_id]
            matched += 1
        else:
            unmatched += 1

    logger.info(f"\n=== Résultats ===")
    logger.info(f"  Matchés    : {matched}")
    logger.info(f"  Non matchés: {unmatched}")

    if results:
        df_results = pd.DataFrame(results).sort_values("score")
        # Affiche les 20 cas les moins sûrs pour vérification
        logger.info("\nCas à vérifier (score le plus bas) :")
        for _, r in df_results.head(20).iterrows():
            logger.info(
                f"  [{int(r['score']):3d}] {r['scraped_name']!r:30s} -> {r['ittf_name']!r:30s} "
                f"({r['gender']}, id={r['ittf_id']})"
            )

    if not args.dry_run:
        logger.info("\nDB mise à jour. Redémarre le dashboard pour voir les filtres Sexe/Âge.")


if __name__ == "__main__":
    main()
