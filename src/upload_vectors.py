import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from vector_store import VectorStore

client = OpenAI()
vec = VectorStore(
    table_name="vectors",
    num_dimensions=1536,
    distance_type="cosine"
)

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    prepared = []
    for idx, row in df.iterrows():
        frage = row['frage']
        antwort = row['antwort']
        embedding = generate_embedding(frage)
        metadata = {"quelle": "facts.csv"}
        prepared.append({
            "text": frage,
            "embedding": embedding,
            "metadata": metadata
        })
    return prepared

def push_to_vector_db(prepared_data):
    for item in prepared_data:
        vec.upsert(
            text=item["text"],
            embedding=item["embedding"],
            metadata=item["metadata"]
        )

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "docs", "facts.csv")
    vec.create_tables()
    vec.create_index()
    prepared_data = prepare_data(csv_path)
    push_to_vector_db(prepared_data)
    print("Alle Vektoren wurden hochgeladen.")

if __name__ == "__main__":
    main()
