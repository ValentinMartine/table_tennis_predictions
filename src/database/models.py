"""
SQLAlchemy ORM models for the table tennis prediction database.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    country = Column(String(3))           # ISO 3166-1 alpha-3
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(1))            # M / F
    ittf_id = Column(String, unique=True, nullable=True)
    flashscore_id = Column(String, unique=True, nullable=True)
    
    # Nouvelles caractéristiques TT
    hand = Column(String(1))              # L / R
    style = Column(String)                # Attack / Defense / All-round
    grip = Column(String)                 # Shakehand / Penhold
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    elo_ratings = relationship("EloRating", back_populates="player")
    home_matches = relationship("Match", foreign_keys="Match.player1_id", back_populates="player1")
    away_matches = relationship("Match", foreign_keys="Match.player2_id", back_populates="player2")

    def __repr__(self) -> str:
        return f"<Player {self.name} ({self.country})>"


class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True)
    comp_id = Column(String, unique=True, nullable=False)   # clé depuis settings.yaml
    name = Column(String, nullable=False)
    country = Column(String(3))
    comp_type = Column(String)            # league / international
    priority = Column(Integer, default=2)

    matches = relationship("Match", back_populates="competition")

    def __repr__(self) -> str:
        return f"<Competition {self.name}>"


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    external_id = Column(String, nullable=True)             # ID source (flashscore, betsapi…)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)
    player1_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    player2_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    # Résultat
    played_at = Column(DateTime, nullable=False)
    winner = Column(Integer)              # 1 ou 2
    score_p1 = Column(Integer)           # sets gagnés joueur 1
    score_p2 = Column(Integer)           # sets gagnés joueur 2
    sets_detail = Column(String)         # ex: "11-8,9-11,11-7" par set

    # Contexte du match
    round_name = Column(String)          # "QF", "SF", "F", "Group Stage"…
    stage = Column(String)               # "knockout" / "group" / "regular"
    is_walkover = Column(Boolean, default=False)

    # Odds bookmaker au moment du match
    odds_p1 = Column(Float, nullable=True)
    odds_p2 = Column(Float, nullable=True)
    odds_source = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("external_id", "competition_id", name="uq_match_external"),
    )

    competition = relationship("Competition", back_populates="matches")
    player1 = relationship("Player", foreign_keys=[player1_id], back_populates="home_matches")
    player2 = relationship("Player", foreign_keys=[player2_id], back_populates="away_matches")

    def __repr__(self) -> str:
        return (
            f"<Match {self.player1_id} vs {self.player2_id} "
            f"({self.score_p1}-{self.score_p2}) @ {self.played_at}>"
        )


class EloRating(Base):
    """Snapshot Elo par joueur après chaque match (pour reconstruction temporelle)."""

    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    rating = Column(Float, nullable=False, default=1500.0)
    matches_played = Column(Integer, default=0)
    computed_at = Column(DateTime, default=datetime.utcnow)   # date du dernier match inclus

    player = relationship("Player", back_populates="elo_ratings")

    __table_args__ = (
        UniqueConstraint("player_id", "computed_at", name="uq_elo_snapshot"),
    )


class IttfRanking(Base):
    """Classement officiel ITTF (scrape hebdomadaire)."""

    __tablename__ = "ittf_rankings"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    points = Column(Float)
    snapshot_date = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("player_id", "snapshot_date", name="uq_ittf_snapshot"),
    )


class WttRanking(Base):
    """Classement officiel WTT (snapshot hebdomadaire)."""

    __tablename__ = "wtt_rankings"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    points_ytd = Column(Float, nullable=True)
    ranking_year = Column(Integer, nullable=False)
    ranking_week = Column(Integer, nullable=False)
    snapshot_date = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("player_id", "snapshot_date", name="uq_wtt_snapshot"),
    )


class BettingRecord(Base):
    """Historique des paris (paper trade + réels)."""

    __tablename__ = "betting_records"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bet_player = Column(Integer, nullable=False)   # 1 ou 2
    stake = Column(Float, nullable=False)
    odds = Column(Float, nullable=False)
    predicted_prob = Column(Float, nullable=False)
    model_edge = Column(Float, nullable=False)     # prob - 1/odds
    result = Column(String)                        # "win" / "loss" / "pending"
    profit_loss = Column(Float, nullable=True)
    is_paper = Column(Boolean, default=True)
    placed_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match")
