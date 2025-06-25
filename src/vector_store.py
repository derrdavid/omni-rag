from timescale_vector import client
from dotenv import load_dotenv
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Any, Dict, List

class VectorSearchResult(BaseModel):
    id: Any
    metadata: Dict[str, Any]
    contents: str
    embedding: List[float]
    distance: float

class VectorStore:
    def __init__(self, table_name="vectors", num_dimensions=1536, distance_type="cosine"):
        load_dotenv()
        service_url = os.getenv("TIMESCALE_SERVICE_URL")
        self.vec = client.Sync(service_url=service_url, table_name=table_name, num_dimensions=num_dimensions, distance_type=distance_type)
        self.openai_client = OpenAI()

    def create_tables(self):
        self.vec.create_tables()

    def create_index(self):
        self.vec.create_embedding_index(client.DiskAnnIndex())

    def generate_embedding(self, text):
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
            )
        return response.data[0].embedding

    def upsert(self, text, embedding, metadata=None, id=None):
        if id is None:
            id = uuid.uuid1()
        if metadata is None:
            metadata = {}
        self.vec.upsert([
            (id, metadata, text, embedding)
        ])

    def search(self, query, top_k=5, metadata_filter=None) -> List[VectorSearchResult]:
        embedding = self.generate_embedding(query)
        results = self.vec.search(embedding, limit=top_k, filter=metadata_filter)
        # Gib eine Liste von VectorSearchResult-Objekten zurück
        return [
            VectorSearchResult(
                id=r[client.SEARCH_RESULT_ID_IDX],
                metadata=r[client.SEARCH_RESULT_METADATA_IDX],
                contents=r[client.SEARCH_RESULT_CONTENTS_IDX],
                embedding=r[client.SEARCH_RESULT_EMBEDDING_IDX],
                distance=r[client.SEARCH_RESULT_DISTANCE_IDX],
            )
            for r in results
        ]