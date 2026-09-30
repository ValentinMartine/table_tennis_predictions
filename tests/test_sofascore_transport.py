"""Sofascore renvoie 403 « challenge » à curl-cffi depuis septembre 2026 : bascule sur navigateur.

L'ancien point d'accès par date (`sport/table-tennis/scheduled-events`) répond 404 ;
les compétitions ITTF et WTT se lisent dans la catégorie « International » (id 88).
"""

from datetime import datetime

import pytest

from src.scraping import sofascore
from src.scraping.sofascore import SofascoreScraper


class _Reponse:
    def __init__(self, status, contenu=None):
        self.status_code = status
        self._contenu = contenu or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._contenu


class _SessionBloquee:
    def __init__(self):
        self.appels = 0

    def get(self, url, timeout=None):
        self.appels += 1
        return _Reponse(403, {"error": {"code": 403, "reason": "challenge"}})


class _NavigateurFactice:
    def __init__(self):
        self.urls = []

    def get_json(self, url):
        self.urls.append(url)
        return {"events": []}


@pytest.fixture
def scraper(monkeypatch):
    s = SofascoreScraper({"request_delay_seconds": 0})
    s._request_delay = 0
    s._cffi_session = _SessionBloquee()
    navigateur = _NavigateurFactice()
    monkeypatch.setattr(s, "_ouvrir_navigateur", lambda: navigateur)
    return s, navigateur


def test_un_challenge_403_bascule_sur_le_navigateur(scraper):
    s, navigateur = scraper
    assert s._get_json(f"{sofascore.API_BASE}/unique-tournament/15005/seasons") == {"events": []}
    assert navigateur.urls == ["https://www.sofascore.com/api/v1/unique-tournament/15005/seasons"]


def test_apres_bascule_curl_cffi_n_est_plus_sollicite(scraper):
    s, navigateur = scraper
    s._get_json(f"{sofascore.API_BASE}/a")
    s._get_json(f"{sofascore.API_BASE}/b")
    assert s._cffi_session.appels == 1
    assert len(navigateur.urls) == 2


def test_la_collecte_par_date_lit_la_categorie_internationale(scraper):
    s, navigateur = scraper
    jour = datetime(2026, 9, 27)
    s._scrape_by_date_filtered("wtt_feeder", ["WTT Feeder"], jour, jour)
    assert navigateur.urls == ["https://www.sofascore.com/api/v1/category/88/scheduled-events/2026-09-27"]


def test_une_meme_journee_n_est_telechargee_qu_une_fois(scraper):
    s, navigateur = scraper
    jour = datetime(2026, 9, 27)
    s._scrape_by_date_filtered("wtt_feeder", ["WTT Feeder"], jour, jour)
    s._scrape_by_date_filtered("wtt_contenders", ["WTT Contender"], jour, jour)
    assert len(navigateur.urls) == 1


class _SessionNavigateurFactice:
    instances = 0

    def __init__(self):
        _SessionNavigateurFactice.instances += 1

    def get(self, url, timeout=None):
        return _Reponse(200, {"events": []})


def test_les_matchs_a_venir_basculent_sur_le_navigateur(monkeypatch):
    from scripts import predict_upcoming as pu

    class _Cffi:
        def Session(self, impersonate=None):
            s = _SessionBloquee()
            s.headers = {}
            return s

    monkeypatch.setattr(pu, "cffi_requests", _Cffi())
    monkeypatch.setattr(pu, "SessionNavigateur", _SessionNavigateurFactice)
    monkeypatch.setattr(pu, "_session_sofascore", None)
    monkeypatch.setattr(pu, "_sofascore_blocked_until", 0.0)
    _SessionNavigateurFactice.instances = 0
    premiere = pu._try_sofascore_session()
    assert isinstance(premiere, _SessionNavigateurFactice)
    assert pu._try_sofascore_session() is premiere
    assert _SessionNavigateurFactice.instances == 1


def test_la_session_navigateur_reecrit_l_hote_de_l_api():
    class _Nav:
        def get_texte(self, url):
            self.url = url
            return 200, '{"events": []}'

    nav = _Nav()
    r = sofascore.SessionNavigateur(nav).get(f"{sofascore.API_BASE}/event/1/odds/1/all")
    assert nav.url == "https://www.sofascore.com/api/v1/event/1/odds/1/all"
    assert (r.status_code, r.json()) == (200, {"events": []})


def test_les_identifiants_fixes_sont_completes_par_la_recherche_par_nom(scraper, monkeypatch):
    s, _ = scraper
    appels = []
    monkeypatch.setattr(s, "_scrape_by_tournament_id", lambda *a: appels.append("ids") or [])
    monkeypatch.setattr(s, "_scrape_by_date_filtered", lambda *a: appels.append("noms") or [])
    jour = datetime(2026, 9, 27)
    s.scrape_competition("wtt_star_contenders", jour, jour)
    assert appels == ["ids", "noms"]
