from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import numpy as np
import json

qdrant_client = QdrantClient(host='localhost', port=6333)

qdrant_client.recreate_collection(
    collection_name='startups',
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

fd = open('./startups_demo.json')
payload = map(json.loads, fd)

vectors = np.load('./ventors.npy')

qdrant_client.upload_collection(
    collection_name='startups',
    vectors=vectors,
    payload=payload,
    ids=None,
    batch_size=256
)