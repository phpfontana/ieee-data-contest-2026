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

def create_directory(path: Union[str, Path]):
    """Create a directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def already_downloaded(root: Path, item_id: str) -> bool:
    """Check if this item's preview file already exists under root."""
    return len(list(root.rglob(f"{item_id}.tif"))) > 0

def main():
    # Create output directory
    root = Path("./data/capella_ieee_data")
    create_directory(root)

    # Load collection and get item links
    print(f"Loading collection from:\n {COLLECTION_URL}\n")
    collection = pystac.Collection.from_file(COLLECTION_URL)
    item_links = collection.get_item_links()

    total_items = len(item_links)
    print(f"Found {total_items} items in the collection.\n")

    print(item_links[0])

if __name__ == "__main__":
    main()