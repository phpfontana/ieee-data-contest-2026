import pystac
import stac_asset.blocking

# Load the collection
collection = pystac.Collection.from_file(
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json"
)

# Get the first item
item_links = collection.get_item_links()
item = pystac.Item.from_file(item_links[0].absolute_href)

# Download a single specific asset (e.g. "thumbnail")
config = stac_asset.blocking.Config(include=["thumbnail"])
item = stac_asset.blocking.download_item(item, config=config)

print("Downloaded to:", list(item.assets["thumbnail"].href))