from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from tqdm.notbook import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# PART 1
df = pd.read_json('./startups_demo.json', lines=True)

# vectors = model.encode([
#     row.alt + ". " + row.description
#     for row in df.itertuples()
# ], show_progress_bar=True)

# vectors.shape
# np.save('ventors.npy', vectors, allow_pickle=False)

# PART 2
vectors = np.load('./ventors.npy')

sample_query = df.iloc[12345].description
print(sample_query)

query_vector = model.encode(sample_query)
scores = cosine_similarity([query_vector], vectors)[0]
top_scores_ids = np.argsort(scores)[-5:][::-1]

for top_id in top_scores_ids:
    print(df.iloc[top_id].description)
    print("-----")