import pandas as pd
from vector_store import VectorStore
from timescale_vector.client import uuid_from_time
import uuid
import datetime
from datasets import load_dataset
from pydantic import BaseModel
from typing import List

class Row(BaseModel):
    id: uuid.UUID
    metadata: dict[str, str]
    text: str
    embedding: List[float]

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

def prepare_dataset(name="HuggingFaceFW/fineweb", dump="CC-MAIN-2014-10", max_datasets=50):
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds_stream = load_dataset(name, dump, streaming=True)
    first_examples = ds_stream['train'].take(max_datasets)
    
    for example in first_examples:
        embedding = vec.generate_embedding(example["text"])
        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "dump": example["dump"],
            "date": example["date"],
            "url": example["url"],
        }
        row = Row(
            id= uuid_from_time(datetime.datetime.now()),
            metadata=metadata,
            text=example["text"],
            embedding=embedding
        )
        push_to_vector_db(row)
        
        
def push_to_vector_db(item: Row):
    vec.upsert(
        id=item.id,
        text=item.text,
        embedding=item.embedding,
        metadata=item.metadata
    )

def main():
    # csv_path = os.path.join(os.path.dirname(__file__), "docs", "facts.csv")
    # vec.create_tables()
    # vec.create_index()
    prepare_dataset()
    # prepared_data = prepare_data(csv_path)
    # push_to_vector_db(prepared_data)
    print("Alle Vektoren wurden hochgeladen.")

if __name__ == "__main__":
    main()
