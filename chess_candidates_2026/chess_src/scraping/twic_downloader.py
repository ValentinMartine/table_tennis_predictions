import requests
import zipfile
import io
from pathlib import Path
from loguru import logger


class TWICDownloader:
    BASE_URL = "https://theweekinchess.com/zips/"

    def __init__(self, download_dir: str = "data/raw/twic"):
        self.download_dir = download_dir
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)

    def download_issue(self, issue_number: int):
        """Downloads a specific TWIC issue zip and extracts the PGN."""
        url = f"{self.BASE_URL}twic{issue_number}g.zip"
        logger.info(f"Downloading TWIC issue {issue_number} from {url}...")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Normally TWIC zip contains one .pgn file named twicXXXX.pgn
                for filename in z.namelist():
                    if filename.endswith(".pgn"):
                        target_path = Path(self.download_dir) / filename
                        with open(target_path, "wb") as f:
                            f.write(z.read(filename))
                        logger.success(f"Saved {filename} to {target_path}")
                        return target_path

        except Exception as e:
            logger.error(f"Failed to download issue {issue_number}: {e}")
            return None

    def download_range(self, start_issue: int, end_issue: int):
        """Downloads a range of TWIC issues."""
        downloaded = []
        for i in range(start_issue, end_issue + 1):
            path = self.download_issue(i)
            if path:
                downloaded.append(path)
        return downloaded

    def download_latest(self, count: int = 5):
        """Downloads the last N issues of TWIC."""
        # For April 2026, the latest issue is around 1640.
        # We can probe or use a fixed end point.
        END_ISSUE = 1640
        return self.download_range(END_ISSUE - count + 1, END_ISSUE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, help="Start issue")
    parser.add_argument("--end", type=int, help="End issue")
    args = parser.parse_args()

    downloader = TWICDownloader()
    if args.start and args.end:
        downloader.download_range(args.start, args.end)
