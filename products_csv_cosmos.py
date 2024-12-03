#  pip install azure-cosmos
# read from csv, upload to cosmos db
import os
import csv
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import sys
import pathlib
from dotenv import load_dotenv
import json 
import pprint
from datetime import datetime

load_dotenv()


# Environment variables
ENDPOINT = os.getenv("AZURE_COSMOS_DB_ENDPOINT")
KEY = os.getenv("AZURE_COSMOS_DB_KEY")
DATABASE_NAME = os.getenv("AZURE_COSMOS_DB_NAME")
CONTAINER_NAME = os.getenv("AZURE_COSMOS_DB_COLLECTION_NAME")
PARTITION_KEY_FIELD = "category"

ASSET_PATH = pathlib.Path("./data").resolve()
PRODUCTS_DATA_PATH = ASSET_PATH  / "hosting" / "products" / "products.csv"

print (ASSET_PATH, PRODUCTS_DATA_PATH)

CATEGORY = "VPS"

# Initialize Cosmos Client
client = CosmosClient(ENDPOINT, KEY)

# Ensure database exists
database = None
try:
    database = client.create_database(DATABASE_NAME)
    print(f"Database '{DATABASE_NAME}' created.")
except exceptions.CosmosResourceExistsError:
    database = client.get_database_client(DATABASE_NAME)
    print(f"Database '{DATABASE_NAME}' already exists.")



# Ensure container exists
container = None
try:
    container = database.create_container(
        id=CONTAINER_NAME,
        partition_key=PartitionKey(path=f"/{PARTITION_KEY_FIELD}"),
        offer_throughput=400
    )
    print(f"Container '{CONTAINER_NAME}' created.")
except exceptions.CosmosResourceExistsError:
    container = database.get_container_client(CONTAINER_NAME)
    print(f"Container '{CONTAINER_NAME}' already exists.")

# Helper function to convert date strings to datetime
def parse_date(date_string):
    if not date_string or date_string.strip() == "":
        return None  # Handle optional Expired field
    return datetime.strptime(date_string, "%Y-%m-%d").isoformat() + "Z"

with open(PRODUCTS_DATA_PATH, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Convert column names to lower_case_with_underscores
        item = {k.lower().replace(" ", "_"): v for k, v in row.items()}
        pprint.pprint(item)


        # Convert numeric fields to their native types
        item["id"] = str(item["id"])  # Ensure ID is a string
        item["cores"] = int(item["cores"])
        item["ram"] = int(item["ram"])
        item["storage"] = int(item["storage"])
        item["bandwidth"] = int(item["bandwidth"])
        item["monthly_cost"] = float(item["monthly_cost"])
        item["hourly_cost"] = float(item["hourly_cost"])
        item["active"] = item["active"].lower() == "true"  # Convert to boolean

        item["category"] = CATEGORY

        # Convert date fields
        item["since_from"] = parse_date(item["since_from"])
        item["expired"] = parse_date(item["expired"])

        pprint.pprint(item)

         # Upsert item: insert or update if ID already exists
        try:
            existing_item = container.read_item(item["id"], partition_key=item[PARTITION_KEY_FIELD])
            container.replace_item(existing_item, item)
            print(f"Updated item with ID: {item['id']}")
        except exceptions.CosmosResourceNotFoundError:
            container.create_item(item)
            print(f"Inserted new item with ID: {item['id']}")