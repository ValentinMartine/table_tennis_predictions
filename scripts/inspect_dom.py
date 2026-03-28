"""
Script de diagnostic DOM — ouvre les sites et sauvegarde le HTML rendu.
Permet d'identifier les vraies classes CSS sans ouvrir le navigateur manuellement.

Usage :
    python scripts/inspect_dom.py --site flashscore
    python scripts/inspect_dom.py --site skillgamesboard
    python scripts/inspect_dom.py --site all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from playwright.sync_api import sync_playwright


SITES = {
    "flashscore_ttnet": {
        "url": "https://www.flashscore.com/table-tennis/turkey/ttnet-league/results/",
        "wait_selector": None,
        "wait_ms": 5000,
    },
    "flashscore_tt": {
        "url": "https://www.flashscore.com/table-tennis/",
        "wait_selector": None,
        "wait_ms": 4000,
    },
    "skillgamesboard": {
        "url": "https://tt.skillgamesboard.com",
        "wait_selector": None,
        "wait_ms": 8000,
    },
    "skillgamesboard_setka": {
        "url": "https://tt.skillgamesboard.com/tournament/setka-cup/results",
        "wait_selector": None,
        "wait_ms": 8000,
    },
}


def inspect(site_key: str, headless: bool = True):
    config = SITES.get(site_key)
    if not config:
        logger.error(f"Site inconnu : {site_key}")
        return

    output_dir = Path("data/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ouverture : {config['url']}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="fr-FR",
        )
        page = context.new_page()

        try:
            page.goto(config["url"], wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(config["wait_ms"])

            # Accepte les cookies si un bouton apparaît
            for selector in [
                "button#onetrust-accept-btn-handler",
                "button.fc-cta-consent",
                "button[id*='accept']",
                "a[id*='accept']",
                "button:has-text('Accept')",
                "button:has-text('Accepter')",
                "button:has-text('Agree')",
            ]:
                try:
                    btn = page.query_selector(selector)
                    if btn and btn.is_visible():
                        btn.click()
                        page.wait_for_timeout(1500)
                        logger.info(f"  Cookie banner cliqué : {selector}")
                        break
                except Exception:
                    pass

            # Attend encore un peu après le cookie
            page.wait_for_timeout(2000)

            # Sauvegarde HTML
            html = page.content()
            html_path = output_dir / f"{site_key}.html"
            html_path.write_text(html, encoding="utf-8")
            logger.info(f"HTML sauvegardé ({len(html):,} chars) → {html_path}")

            # Screenshot
            screenshot_path = output_dir / f"{site_key}.png"
            page.screenshot(path=str(screenshot_path), full_page=False)
            logger.info(f"Screenshot → {screenshot_path}")

            # Analyse rapide des classes CSS présentes
            classes_info = page.evaluate("""
                () => {
                    const allClasses = new Set();
                    document.querySelectorAll('[class]').forEach(el => {
                        el.className.toString().split(/\\s+/).forEach(c => {
                            if (c && c.length > 2) allClasses.add(c);
                        });
                    });

                    // Cherche spécifiquement les patterns de matchs/résultats
                    const matchPatterns = ['match', 'event', 'result', 'score',
                                          'player', 'team', 'game', 'row'];
                    const relevant = [...allClasses].filter(c =>
                        matchPatterns.some(p => c.toLowerCase().includes(p))
                    );

                    return {
                        total_classes: allClasses.size,
                        relevant: relevant.sort(),
                        title: document.title,
                        url: window.location.href,
                    };
                }
            """)

            logger.info(f"  Titre : {classes_info['title']}")
            logger.info(f"  URL finale : {classes_info['url']}")
            logger.info(f"  Total classes CSS : {classes_info['total_classes']}")
            logger.info(f"  Classes liées aux matchs/résultats :")
            for cls in classes_info["relevant"]:
                logger.info(f"    .{cls}")

        except Exception as e:
            logger.error(f"Erreur : {e}")

        browser.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--site",
        choices=list(SITES.keys()) + ["all"],
        default="flashscore_ttnet",
    )
    p.add_argument("--visible", action="store_true",
                   help="Ouvre le navigateur en mode visible (non headless)")
    return p.parse_args()


def main():
    args = parse_args()
    headless = not args.visible

    if args.site == "all":
        for key in SITES:
            inspect(key, headless=headless)
    else:
        inspect(args.site, headless=headless)


if __name__ == "__main__":
    main()
