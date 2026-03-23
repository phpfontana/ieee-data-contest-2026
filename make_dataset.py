import pystac
import stac_asset.blocking
from tqdm import tqdm

collection = pystac.Collection.from_file(
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json"
)

item_links = collection.get_item_links()
item = pystac.Item.from_file(item_links[0].absolute_href)

config = stac_asset.blocking.Config(include=["thumbnail"])

with tqdm(total=1, desc="Downloading thumbnail", unit="file") as pbar:
    item = stac_asset.blocking.download_item(item, directory=".", config=config)
    pbar.update(1)

print("Done!")