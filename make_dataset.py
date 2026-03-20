import os
import json
import time
import pystac
import stac_asset.blocking
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Union

COLLECTION_URL = (
    "https://capella-open-data.s3.us-west-2.amazonaws.com"
    "/stac/capella-open-data-ieee-data-contest/collection.json"
)
ASSETS_TO_DOWNLOAD = ["preview", "thumbnail"]
RETRY_DELAY = 5

LOG_SUCCESS = "log_success.txt"
LOG_FAILED  = "log_failed.txt"


def log(filepath, message):
    with open(filepath, "a") as f:
        f.write(f"{datetime.now().isoformat()} | {message}\n")


def already_downloaded(root: Path, item_id: str) -> bool:
    """Check if this item's preview file already exists under root."""
    return len(list(root.rglob(f"{item_id}_preview.tif"))) > 0


def download_item(link, index: int, total: int, root: Path, assets: list) -> bool:
    """Download a single STAC item. Returns True on success."""
    item_url = link.absolute_href
    item_id  = item_url.split("/")[-1].replace(".json", "")

    if already_downloaded(root, item_id):
        tqdm.write(f"[{index}/{total}] SKIP (exists): {item_id}")
        log(root / LOG_SUCCESS, f"SKIPPED | {item_id}")
        return True

    tqdm.write(f"[{index}/{total}] Downloading: {item_id}")

    include_args = sum([["-i", asset] for asset in assets], [])
    cmd = (
        ["stac-asset", "download", item_url, str(root)]
        + include_args
    )

    exit_code = os.system(" ".join(cmd))

    if exit_code == 0:
        tqdm.write(f"[{index}/{total}] ✓ Done: {item_id}")
        log(root / LOG_SUCCESS, f"OK | {item_id}")
        return True
    else:
        tqdm.write(f"[{index}/{total}] ✗ FAILED (exit {exit_code}): {item_id}")
        log(root / LOG_FAILED, f"FAILED | exit={exit_code} | {item_url}")
        return False


def load_ieee_data_contest_2026_dataset(
    root: Union[str, Path],
    download: bool = True,
    max_items: int = None,
    assets: list = ASSETS_TO_DOWNLOAD,
):
    """Load the IEEE Data Contest 2026 dataset.

    Args:
        root: Directory where the dataset should be stored.
        download: If True, download the dataset to the specified root directory.
                  If the dataset already exists, it will not be downloaded again.
        max_items: Maximum number of items to download. If None, all items will be downloaded.
        assets: List of asset types to download (e.g., ["preview", "thumbnail"]).
    """
    # Verify if dataset already exists at the specified root directory
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        print(f"Dataset already exists at: {root}")
        return

    if not download:
        print(f"Dataset not found at: {root} and download is set to False.")
        return

    # Create output directory
    root.mkdir(parents=True, exist_ok=True)

    # Load collection and get item links
    print(f"[→] Loading collection from:\n    {COLLECTION_URL}\n")
    collection = pystac.Collection.from_file(COLLECTION_URL)
    item_links = collection.get_item_links()

    total = len(item_links)
    print(f"[✓] Found {total} items in collection.")

    if max_items is not None:
        item_links = item_links[:max_items]
        print(f"[→] Limiting to {max_items} item(s).\n")
    else:
        print(f"[→] Downloading all {total} item(s). (~1.2 TB estimated)\n")

    failed_links = []

    # First pass — download all items
    with tqdm(total=len(item_links), desc="Overall progress", unit="item") as pbar:
        for i, link in enumerate(item_links, start=1):
            success = download_item(link, i, len(item_links), root, assets)
            if not success:
                failed_links.append(link)
            pbar.update(1)

    # Retry pass — attempt failed items once more
    if failed_links:
        print(f"\n[!] Retrying {len(failed_links)} failed item(s)...\n")
        still_failed = []
        for i, link in enumerate(failed_links, start=1):
            time.sleep(RETRY_DELAY)
            success = download_item(link, i, len(failed_links), root, assets)
            if not success:
                still_failed.append(link.absolute_href)

        if still_failed:
            print(f"\n[✗] {len(still_failed)} item(s) permanently failed.")
            print(f"    See: {(root / LOG_FAILED).resolve()}")
        else:
            print(f"\n[✓] All retries succeeded!")
    else:
        print(f"\n[✓] All items downloaded successfully!")

    print(f"\n[→] Files saved  : {root.resolve()}")
    print(f"[→] Success log  : {(root / LOG_SUCCESS).resolve()}")
    print(f"[→] Failure log  : {(root / LOG_FAILED).resolve()}\n")


if __name__ == "__main__":
    load_ieee_data_contest_2026_dataset(
        root="./data/capella_ieee_data",
        download=True,
        max_items=None,        
        assets=["preview", "thumbnail"],
    )
