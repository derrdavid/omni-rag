from timescale_vector import client
from dotenv import load_dotenv
import os
import uuid


class VectorStore:
    def __init__(self, table_name="vectors", num_dimensions=1536, distance_type="cosine"):
        load_dotenv()
        service_url = os.getenv("TIMESCALE_SERVICE_URL")
        self.vec = client.Sync(service_url=service_url, table_name=table_name, num_dimensions=num_dimensions, distance_type=distance_type)

    def create_tables(self):
        self.vec.create_tables()

    def create_index(self):
        # Standard: DiskANN Index, kann angepasst werden
        self.vec.create_embedding_index(client.DiskAnnIndex())

    def upsert(self, text, embedding, metadata=None, id=None):
        # id: UUID, metadata: dict, text: str, embedding: list[float]
        if id is None:
            id = uuid.uuid1()
        if metadata is None:
            metadata = {}
        self.vec.upsert([
            (id, metadata, text, embedding)
        ])

    def search(self, embedding, top_k=3, metadata_filter=None):
        # metadata_filter: dict, z.B. {"action": "jump"}
        return self.vec.search(embedding, limit=top_k, filter=metadata_filter)

    def delete_by_id(self, id):
        self.vec.delete_by_ids([id])

    def delete_by_metadata(self, metadata_filter):
        self.vec.delete_by_metadata(metadata_filter)

    def delete_all(self):
        self.vec.delete_all()