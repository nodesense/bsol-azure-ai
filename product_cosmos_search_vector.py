# Inderer like code pull data from cosmos db and add to search index
# pip install azure-cosmos azure-search-documents

#  pip install azure-cosmos

import os
import csv
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import sys
import pathlib
from dotenv import load_dotenv
import json 
import pprint
from datetime import datetime

from openai import AzureOpenAI

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient 

import pandas as pd
from azure.search.documents.indexes.models import (
    SemanticSearch,
    SearchField,
    # VectorField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    SearchIndex,
)
 

load_dotenv()


# Environment variables
ENDPOINT = os.getenv("AZURE_COSMOS_DB_ENDPOINT")
KEY = os.getenv("AZURE_COSMOS_DB_KEY")
DATABASE_NAME = os.getenv("AZURE_COSMOS_DB_NAME")
CONTAINER_NAME = os.getenv("AZURE_COSMOS_DB_COLLECTION_NAME")
PARTITION_KEY_FIELD = "category"

AZURE_SEARCH_AI_ENDPOINT= os.getenv("AZURE_SEARCH_AI_ENDPOINT")
AZURE_SEARCH_AI_KEY= os.getenv("AZURE_SEARCH_AI_KEY")
PRODUCTS_SEARCH_INDEX_NAME= os.getenv("PRODUCTS_SEARCH_INDEX_NAME")

ASSET_PATH = pathlib.Path("./data").resolve()


# Initialize Cosmos Client
cosmos_client = CosmosClient(ENDPOINT, KEY)
database = cosmos_client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# create a vector embeddings client that will be used to generate vector embeddings
embeddings = project.inference.get_embeddings_client()

openai_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        # azure_ad_token_provider=token_provider,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

from azure.core.credentials import AzureKeyCredential

search_index_client = SearchIndexClient(endpoint=AZURE_SEARCH_AI_ENDPOINT, credential=AzureKeyCredential(AZURE_SEARCH_AI_KEY))
search_client = SearchClient(endpoint=AZURE_SEARCH_AI_ENDPOINT, index_name=PRODUCTS_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_AI_KEY))

from azure.core.exceptions import ResourceNotFoundError

def delete_search_index(index_name):
    """Delete a search index if it exists."""
    try:
        # Check if the index exists
        search_index_client.get_index(index_name)
        print(f"Index '{index_name}' exists. Deleting it...")
        # Delete the index
        search_index_client.delete_index(index_name)
        print(f"Index '{index_name}' has been successfully deleted.")
    except ResourceNotFoundError:
        print(f"Index '{index_name}' does not exist. Nothing to delete.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def create_search_index(index_name):
    """Create or update the Azure Search index with combined content and vector support."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="name", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="cores", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SimpleField(name="ram", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SimpleField(name="storage", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SimpleField(name="bandwidth", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
        SearchableField(name="cpu", type=SearchFieldDataType.String, facetable=True),
        SearchableField(name="category", type=SearchFieldDataType.String, facetable=True),
        SearchableField(name="processor", type=SearchFieldDataType.String),
        SimpleField(name="monthly_cost", type=SearchFieldDataType.Double, filterable=True, facetable=True),
        SimpleField(name="hourly_cost", type=SearchFieldDataType.Double, filterable=True, facetable=True),
        SearchableField(name="plan_type", type=SearchFieldDataType.String, facetable=True),
        SimpleField(name="active", type=SearchFieldDataType.Boolean, filterable=True, facetable=True),
        SimpleField(name="since_from", type=SearchFieldDataType.DateTimeOffset, filterable=True),
        SimpleField(name="expired", type=SearchFieldDataType.DateTimeOffset, filterable=True),
        SearchableField(name="description", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),  #   combined content field
        # VectorField(name="content_vector", vector_dim=1536, searchable=True, filterable=False, retrievable=True)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            # Size of the vector created by the text-embedding-ada-002 model.
            vector_search_dimensions= 1536,  # text-embedding-ada-002
            vector_search_profile_name="myHnswProfile",
        ),
    ]

 # The "content" field should be prioritized for semantic ranking.
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            keywords_fields=[],
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    
    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=1000,
                    ef_search=1000,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            ),
        ],
    )


    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=semantic_search,
        vector_search=vector_search,
    )

    import traceback
    try:
        search_index_client.create_index(index)
        print(f"Index '{PRODUCTS_SEARCH_INDEX_NAME}' created successfully.")
    except Exception as e:
        print(f"Index creation failed or already exists: {str(e)}")
        print(traceback.format_exc())


def generate_combined_content(item):
    """Generate a human-readable combined content string."""
    return f"name: {item['name']}, cores: {item['cores']} cores, ram: {item['ram']}GB, storage: {item['storage']}GB, " \
           f"bandwidth: {item['bandwidth']}TB, category: {item.get('category', 'VPS')}, " \
           f"plan type: {item['plan_type']}, description: {item['description']}"

 
def generate_combined_title(item):
    """Generate a human-readable combined content string."""
    return f"{item['plan_type']} {item['cpu']} Processor | {item['name']} | {item['cores']} cores | {item['ram']} GB RAM | {item['storage']} GB Storage, " \
           f"{item['bandwidth']} TB Bandwidth {item.get('category', 'VPS')}, " 
           

def generate_content_vector(text):
    """Generate a vector embedding for the given text using OpenAI."""
    try:
        emb =  embeddings.embed(input=text, model="text-embedding-ada-002")
        return emb.data[0].embedding 
    except Exception as e:
        print(f"Failed to generate vector: {e}")
        return None

def populate_search_index(items):
    """Insert or update data in Azure Search Index."""
    documents = []
    for item in items:
        combined_content = generate_combined_content(item)
        combined_title = generate_combined_title(item)
        print ("title ", combined_title)
        print ("content", combined_content)
        
        vector = generate_content_vector(combined_content)

        documents.append({
            "id": item["id"],
            "name": item["name"],
            "title": combined_title,
            "cores": item["cores"],
            "ram": item["ram"],
            "storage": item["storage"],
            "bandwidth": item["bandwidth"],
            "cpu": item["cpu"],
            "category": item.get("category", "VPS"),
            "processor": item["processor"],
            "monthly_cost": item["monthly_cost"],
            "hourly_cost": item["hourly_cost"],
            "plan_type": item["plan_type"],
            "active": item["active"],
            "since_from": item["since_from"],
            "expired": item.get("expired", None),
            "description": item["description"],
            "content": combined_content,  # Add combined content field
            "content_vector": vector  # Add the generated vector
        })

    try:
        results = search_client.merge_or_upload_documents(documents)
        print("Documents uploaded or updated:")
        for result in results:
            print(f"  - {result.key}: {result.succeeded}")
    except Exception as e:
        print(f"Error uploading documents: {str(e)}")



def fetch_cosmos_data():
    """Fetch data from Cosmos DB."""
    query = "SELECT * FROM c"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    return items

if __name__ == "__main__":
    # Step 1: Create or update the search index
    delete_search_index(PRODUCTS_SEARCH_INDEX_NAME)

    create_search_index(PRODUCTS_SEARCH_INDEX_NAME)
    

    # # Step 2: Fetch data from Cosmos DB
    cosmos_items = fetch_cosmos_data()

    print (cosmos_items)
    # # Step 3: Insert or update data in Azure Cognitive Search
    populate_search_index(cosmos_items)
