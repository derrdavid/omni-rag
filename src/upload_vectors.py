import os
import pandas as pd
from vector_store import VectorStore
from timescale_vector.client import uuid_from_time
import datetime

vec = VectorStore(
    table_name="vectors",
    num_dimensions=1536,
    distance_type="cosine"
)

def prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    prepared = []
    for idx, row in df.iterrows():
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        embedding = vec.generate_embedding(content)
        metadata = {
            "source": "facts.csv",
            "created_at": datetime.datetime.now().isoformat()
            }
        prepared.append({
            "id": uuid_from_time(datetime.datetime.now()),
            "text": content,
            "embedding": embedding,
            "metadata": metadata,
        })
    return prepared

def push_to_vector_db(prepared_data):
    for item in prepared_data:
        vec.upsert(
            id=item["id"],
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
