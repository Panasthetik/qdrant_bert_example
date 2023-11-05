from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models.models import Filter
from sentence_transformers import SentenceTransformer



class NeuralSearcher:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.qdrant_client = QdrantClient(host='localhost', port=6333)

    def search(self, text: str) -> List[dict]:
        vector = self.model.encode(text).tolist()
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            # query_filter=Filter(**filter_) if filter_ else None,
            # top=5
        )
        return [hit.payload for hit in hits]