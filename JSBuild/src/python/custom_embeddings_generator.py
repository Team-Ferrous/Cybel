
import json
from   json import JSONEncoder

import chromadb
import pandas as pd
import supabase
import numpy as np
from   pypdf import PdfReader
from   sentence_transformers import SentenceTransformer
from   chromadb.utils import embedding_functions

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
# Extract text from PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


# Split text into chunks for embeddings
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def create_embeddings_from_pdf(user_hash, pdf_path):

    # Extract PDF text
    text = extract_pdf_text(pdf_path)

    if not text:
        print("PDF had no readable text.")
        return

    # Chunk text
    chunks = chunk_text(text)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embeddings = model.encode(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        memory_id = f"{user_hash}_pdf_{i}"

        data = {
            "memory_id": memory_id,
            "user_id": user_hash,
            "text": chunk,
            "embedding": embedding.tolist(),
        }

        response = supabase.table("Embeddings")\
            .select("memory_id")\
            .eq("memory_id", memory_id)\
            .limit(1)\
            .execute()

        if len(response.data) > 0:
            supabase.table("Embeddings")\
                .update(data)\
                .eq("memory_id", memory_id)\
                .execute()
        else:
            supabase.table("Embeddings")\
                .insert(data)\
                .execute()

    print(f"Processed {len(chunks)} chunks from PDF.")

def chromadb_embeddings_creator(user_hash):
    # Fetch data from Supabase
    response = supabase.table("MemoryRecommendationCollection").select("*").execute()
    df = pd.json_normalize(response.data)[1:]  # Convert to DataFrame and skip header

    # Initialize model and ChromaDB client
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="chroma_data/")
    collection_name = "cdb_memories"

    # Retrieve or create ChromaDB collection
    try:
        collection = client.get_collection(name=collection_name)
    except:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

    # Prepare and encode memory texts
    ids = df["Id"].tolist()
    texts = df["Memory"].tolist()
    embeddings = model.encode(texts)

    # Store embeddings in Supabase
    existing_embeddings = supabase.table("Embeddings").select("embedding_id").execute()
    embeddings_json = json.dumps({"embeddings": embeddings.tolist()})
    pyi = NumpyArrayEncoder()

    if not existing_embeddings.data:
        supabase.table("Embeddings").insert({"embedding_id": str(user_hash), "embeddings": embeddings_json}).execute()
    else:
        supabase.table("Embeddings").update({"embedding_id": str(user_hash), "embeddings": embeddings_json}).eq("embedding_id", user_hash).execute()

    # Add embeddings to ChromaDB
    for id, text, embedding in zip(ids, texts, embeddings):
        if not collection.get(id):
            collection.add(
                documents=[text],
                embeddings=[embedding.tolist()],
                metadatas=[{"source": id}],
                ids=[id]
            )
    
    print(f"Memories have been successfully added to ChromaDB for {user_hash}")


#TBA: not completed and needs refactoring
def create_embeddings_from_json(user_hash):
    # Fetch data from Supabase using the table name "UserMemories" and user_hash
    memories = supabase.table("Memories").select("associated_data").eq("user_id", user_hash).execute()

    associated_data = json.loads(memories.data[0]['associated_data'])

    # check if the associated_data is not empty
    if not associated_data:
        print(f"No memories found for {user_hash}")
        return

    entries = associated_data['Entries']

    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #texts = [entry['summary'] for entry in entries]
    texts = [entry.get('summary') if entry.get('summary') not in [None, "", '', '\"', "\"", "None"] else 'No summary' for entry in entries]
    ids   = [entry['id'] for entry in entries]

    # Compute embeddings for all memory texts
    embeddings = model.encode(texts)

    for text, embedding, id in zip(texts, embeddings, ids):
        data = {
            "memory_id": id,  # Generate a unique ID for the embedding entry
            "user_id":   user_hash,  # Associate with the user
            "text":      text,
            "embedding": embedding.tolist(),  # Convert numpy array to list for JSON 
        }
        
        response = supabase.table("Embeddings").select("memory_id").eq("memory_id", id).limit(1).execute()

        if len(response.data) > 0:
            supabase.table('Embeddings').update(data).eq('memory_id', id).execute()
        else:
            supabase.table('Embeddings').insert(data).execute()

def chromadb_embeddings_creator_raw(user_hash, supabase_collection_name):
    #pull memory collection, valid collection names: "MemoryRecommendationCollection", "CommunityActionsCollection", "GeneralCardsCollection"
    response = supabase.table(supabase_collection_name).select("*").execute() #GeneralRecommendationCollection

    #read response into dataframe
    df = pd.json_normalize(response)

    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=f"chroma_data/")

    # Create a collection in ChromaDB
    collection_name = "categories"
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

    # Check if the collection exists
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

    # Prepare the data for embedding
    ids      = []
    texts    = []

    # remove header
    df = df[1:]

    # Iterate rows in the DataFrame
    for index, row in df.iterrows():
        memory_text = row["Memory"]
        id = row["Id"]
        ids.append(id)
        texts.append(memory_text)

    # Compute embeddings for all memory texts
    embeddings = model.encode(texts)
    res = supabase.table("Embeddings").select("embedding_id").execute()
    if(len(res.data) <= 0):
        embeddings_json  = json.dumps({"embeddings": embeddings.tolist()})
        (supabase.table("Embeddings").insert({"embedding_id": str(user_hash), "embeddings": str(embeddings_json)}).execute())
    else:
        embeddings_json  = json.dumps({"embeddings": embeddings.tolist()})
        (supabase.table("Embeddings").update({"embedding_id": str(user_hash), "embeddings": str(embeddings_json)}).eq("embedding_id", user_hash).execute())
    
    # Store embeddings in ChromaDB
    for id, text, embedding in zip(ids, texts, embeddings):
        col = collection.get(id) 
        if(col is not None):
            continue
        else:
            collection.add(
                documents=[text],
                embeddings=[embedding.tolist()],
                metadatas=[{"source": f"{id}"}],
                ids=[id]
            )
    print("Activities have been successfully added to ChromaDB.")