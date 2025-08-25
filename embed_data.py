"""
Script to embed example research papers and upload them to MongoDB for vector search.

Before running this script, ensure you have set the following environment variables:

  - `MONGO_URI`: MongoDB connection string (e.g. mongodb+srv://user:password@cluster.mongodb.net/)

The script uses a sentence embedding model from the `sentence-transformers` library.  You can change the model name to suit your needs.  After generating embeddings, it inserts the documents into a MongoDB collection with an `embedding` field containing the dense vector.  You may also create a vector index in MongoDB manually via the Atlas UI or using the driver (see comments below).
"""

import json
import os
from typing import List

import pymongo
from sentence_transformers import SentenceTransformer



def load_papers(path: str) -> List[dict]:
    """Load papers from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def embed_documents(papers: List[dict], model_name: str = "all-MiniLM-L6-v2") -> List[dict]:
    """Compute sentence embeddings for each paper's content."""
    model = SentenceTransformer(model_name)
    docs = []
    for paper in papers:
        embedding = model.encode(paper["content"], convert_to_numpy=True).tolist()
        docs.append({
            "_id": paper["id"],
            "title": paper["title"],
            "content": paper["content"],
            "embedding": embedding,
        })
    return docs



def main():
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("Please set the MONGO_URI environment variable.")

    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client["research_agent"]
    collection = db["papers"]

    # Load and embed papers
    papers = load_papers(os.path.join("data", "papers.json"))
    docs = embed_documents(papers)

    # Insert into MongoDB (replace existing documents)
    print(f"Inserting {len(docs)} documents into MongoDB...")
    collection.delete_many({})
    collection.insert_many(docs)
    print("Documents inserted.")

    # To create a vector index, you can run the following (MongoDB server must support vector search):
    # collection.create_index([("embedding", "cosine")])
    # For instructions, see: https://www.mongodb.com/docs/atlas/atlas-vector-search/


if __name__ == "__main__":
    main()
