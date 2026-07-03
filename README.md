# Prédictions au tennis de table

Modèle de prédiction des matchs de tennis de table (WTT et grandes compétitions ITTF),
combinant un **ensemble LightGBM + XGBoost** et une baseline **Elo**, avec **calibration
isotonique** des probabilités et évaluation de la **valeur de pari** (critère de Kelly)
face aux cotes de marché.

## Objectif

À partir de l'historique des matchs et des classements ITTF/WTT, le projet estime la
probabilité de victoire de chaque joueur sur les matchs à venir, mesure l'**écart (edge)**
entre le modèle et les cotes des bookmakers, et évalue une stratégie de mise par backtest.

## Méthode

1. **Données** : matchs récupérés via l'API World Table Tennis et Sofascore ; classements
   ITTF/WTT et infos joueurs (snapshots historiques) ; stockage SQLite.
2. **Variables** : Elo pondéré par l'importance de la compétition, forme récente (fenêtre
   décroissante), confrontations directes, classements ITTF/WTT, repos et fatigue.
3. **Modèles** : ensemble **LightGBM** et **XGBoost**, comparés à une baseline **Elo** ;
   **calibration isotonique** ; interprétation par **SHAP**.
4. **Value betting** : calcul de l'edge face aux cotes, mise fractionnée de **Kelly**, et
   backtest de la bankroll avec garde-fous (stop-loss, edge et cote minimaux).

## Structure

```text
src/                  Cœur du projet (package Python)
  features/           Construction des variables (Elo, forme, h2h, classements)
  models/             Modèles LightGBM / XGBoost et baseline Elo
  database/           Schéma SQLAlchemy et accès aux données
  scraping/           Collecte des matchs, cotes et classements
  backtesting/        Backtest de mise (Kelly) et utilitaires de cotes
config/               Compétitions, hyperparamètres, paramètres de mise
scripts/              Bootstrap des données, scraping, entraînement, prédiction, backtest
data/                 Base SQLite, modèles entraînés, métriques, prédictions
```

## Installation et usage

```bash
pip install -r requirements.txt
cp .env.example .env        # renseigner les tokens d'API (facultatif)

python scripts/bootstrap_data.py     # import initial des données publiques
python scripts/run_scraper.py        # collecte des matchs récents
python scripts/train_model.py        # entraînement des modèles
python scripts/predict_upcoming.py   # prédictions des matchs à venir
python scripts/backtest.py           # backtest de la stratégie de mise
```

## Limites

Les cotes et matchs à venir dépendent de sources tierces dont la disponibilité varie. Les
performances de mise sont issues d'un backtest et ne constituent pas un conseil financier ;
le projet a une visée méthodologique.

## Licence

MIT.
