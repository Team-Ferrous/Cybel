'''import weaviate
from weaviate.classes.init import Auth
import os

# Best practice: store your credentials in environment variables
weaviate_url     = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
cohere_api_key   = os.environ["COHERE_APIKEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
    headers={"X-Cohere-Api-Key": cohere_api_key},           # Replace with your Cohere API key
)

questions = client.collections.get("Question")

response = questions.generate.near_text(
    query="biology",
    limit=2,
    grouped_task="Write a tweet with emojis about these facts."
)

print(response.generated)  # Inspect the generated text

client.close()  # Free up resources'''

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
import requests, json, os

def load_json_files(directory):
    """Load and validate JSON files from a directory."""
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    documents = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                documents.append(json.loads(json_file))

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return documents

# Best practice: store your credentials in environment variables
weaviate_url = weaviate_url     = "ishkbitntd7ll5xcxf8gw.c0.us-east1.gcp.weaviate.cloud" #os.environ["WEAVIATE_URL"]
weaviate_api_key = "a5QfMY4qjkZyFicQBVqbi5GPCS6oUkTByqwE" #os.environ["WEAVIATE_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
)

questions = client.collections.create(
    name="Introduction",
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(), # Configure the Weaviate Embeddings integration
    generative_config=Configure.Generative.cohere()             # Configure the Cohere generative AI integration
)

# Get absolute path to the JSON file
file_path = os.path.abspath('/Users/sasori/Downloads/automat-llm-main/Cleaned_JSONs/cleaned_SupercellAMemory0.json')

# Open the file and load its JSON contents
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

    # Extract list of 'Entry' strings if the JSON is a list of dicts
    entries = [item['Entry'] for item in data if 'Entry' in item]

    with questions.batch.fixed_size(batch_size=200) as batch:
        for d in entries:
            print(d)
            batch.add_object(
                {
                    "entry": d
                }
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = questions.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")

    client.close()  # Free up resources
