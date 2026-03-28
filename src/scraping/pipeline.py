"""
Pipeline de scraping : orchestre tous les scrapers et insère en DB.
"""
from datetime import datetime

import yaml
from loguru import logger
from sqlalchemy.orm import Session

from ..database.db import get_session
from ..database.models import Competition, Match, Player
from .base import RawMatch
from .betsapi import BetsAPIScraper
from .flashscore import FlashscoreScraper
from .oddsmatrix import OddsMatrixScraper
from .skillgamesboard import SkillGamesBoardScraper
from .sofascore import SofascoreScraper
from .tabletennis_guide import TableTennisGuideScraper


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_or_create_player(session: Session, name: str, country: str | None) -> Player:
    player = session.query(Player).filter_by(name=name).first()
    if not player:
        player = Player(name=name, country=country)
        session.add(player)
        session.flush()
    return player


def _get_or_create_competition(session: Session, comp_id: str, config: dict) -> Competition:
    comp = session.query(Competition).filter_by(comp_id=comp_id).first()
    if not comp:
        # Cherche les métadonnées dans le config
        meta = _find_comp_meta(comp_id, config)
        comp = Competition(
            comp_id=comp_id,
            name=meta.get("name", comp_id),
            country=meta.get("country"),
            comp_type=meta.get("type", "league"),
            priority=meta.get("priority", 2),
        )
        session.add(comp)
        session.flush()
    return comp


def _find_comp_meta(comp_id: str, config: dict) -> dict:
    for section in ("leagues", "international"):
        for c in config.get("competitions", {}).get(section, []):
            if c["id"] == comp_id:
                c["type"] = section
                return c
    return {}


def insert_matches(raw_matches: list[RawMatch], config: dict, batch_size: int = 500) -> int:
    """Insère les matchs en DB, retourne le nombre de nouveaux matchs."""
    if not raw_matches:
        return 0

    # Déduplication intra-batch (Sofascore peut retourner le même match deux fois)
    seen_ids: set[str] = set()
    unique_matches = []
    for raw in raw_matches:
        if raw.external_id not in seen_ids:
            seen_ids.add(raw.external_id)
            unique_matches.append(raw)

    inserted = 0
    for i in range(0, len(unique_matches), batch_size):
        batch = unique_matches[i : i + batch_size]
        with get_session() as session:
            # Charge les external_ids déjà en DB pour ce batch
            batch_ids = [r.external_id for r in batch]
            existing_ids = {
                row[0]
                for row in session.query(Match.external_id)
                .filter(Match.external_id.in_(batch_ids))
                .all()
            }
            for raw in batch:
                if raw.external_id in existing_ids:
                    continue
                comp = _get_or_create_competition(session, raw.competition_id, config)
                p1 = _get_or_create_player(session, raw.player1_name, raw.player1_country)
                p2 = _get_or_create_player(session, raw.player2_name, raw.player2_country)
                match = Match(
                    external_id=raw.external_id,
                    competition_id=comp.id,
                    player1_id=p1.id,
                    player2_id=p2.id,
                    played_at=raw.played_at,
                    winner=raw.winner,
                    score_p1=raw.score_p1,
                    score_p2=raw.score_p2,
                    sets_detail=raw.sets_detail,
                    round_name=raw.round_name,
                    stage=raw.stage,
                    is_walkover=raw.is_walkover,
                    odds_p1=raw.odds_p1,
                    odds_p2=raw.odds_p2,
                    odds_source=raw.odds_source,
                )
                session.add(match)
                inserted += 1

    return inserted


def run_scraping(
    start_date: datetime,
    end_date: datetime,
    config_path: str = "config/settings.yaml",
    priority_max: int = 2,
) -> dict[str, int]:
    """
    Lance le scraping de toutes les compétitions jusqu'à priority_max.

    Returns:
        dict source → nombre de matchs insérés
    """
    config = load_config(config_path)
    scraping_cfg = config.get("scraping", {})

    oddsmatrix_cfg = {**scraping_cfg, **config.get("oddsmatrix", {})}
    scrapers = {
        "skillgamesboard": SkillGamesBoardScraper(scraping_cfg),
        "flashscore": FlashscoreScraper(scraping_cfg),
        "sofascore": SofascoreScraper(scraping_cfg),
        "tabletennis_guide": TableTennisGuideScraper(scraping_cfg),
        "betsapi": BetsAPIScraper(scraping_cfg),
        "oddsmatrix": OddsMatrixScraper(oddsmatrix_cfg),
    }

    # Regroupe les compétitions par source
    by_source: dict[str, list[str]] = {}
    for section in ("leagues", "international"):
        for comp in config.get("competitions", {}).get(section, []):
            if comp.get("priority", 99) > priority_max:
                continue
            src = comp.get("source", "flashscore")
            by_source.setdefault(src, []).append(comp["id"])

    stats: dict[str, int] = {}
    for source, comp_ids in by_source.items():
        scraper = scrapers.get(source)
        if not scraper:
            logger.warning(f"Scraper inconnu : {source}")
            continue
        raw_matches = scraper.scrape_all_competitions(comp_ids, start_date, end_date)
        n = insert_matches(raw_matches, config)
        stats[source] = n
        logger.info(f"{source} : {n} nouveaux matchs insérés")

    return stats
