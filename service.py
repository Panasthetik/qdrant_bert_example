from fastapi import FastAPI

from search_api import NeuralSearcher
# from qdrant_demo.text_searcher import TextSearcher

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name='startups')
# text_searcher = TextSearcher()

result = neural_searcher.search("Berlin")
print(result)

@app.get("/api/search")
async def read_item(q: str):
    return {
        "result": neural_searcher.search(text=q)
        # if neural else text_searcher.search(query=q)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)