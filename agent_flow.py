"""
Definition of the LangGraph agent workflow used for the research assistant.

This module defines two nodes:

1. `retrieve` – performs a hybrid search over the stored paper embeddings and returns the most relevant documents.
2. `summarise` – uses a language model to generate a concise summary of the retrieved documents and optionally an “explain like I’m five” (ELI5) explanation.

To keep the example self‑contained, the retrieval node performs a simple cosine similarity search in Python.  For production use you should use a vector database (e.g. MongoDB’s Atlas Vector Search or AstraDB) to perform efficient similarity search.

To run the agent, import `build_agent` and call it with your question.  For example:

```python
from agent_flow import build_agent

agent = build_agent()
response = agent.invoke({"question": "Explain the volatility of Bitcoin"})
print(response)
```
"""

import os
from typing import List, Dict, Any

import numpy as np
import openai

from langgraph.graph import Graph


def load_documents() -> List[Dict[str, Any]]:
    """Load the embedded documents from MongoDB or a local cache."""
    # For demonstration purposes we load the JSON file and compute embeddings on the fly.
    # In a real application you would retrieve the documents and their pre‑computed embeddings from MongoDB.
    import json
    from sentence_transformers import SentenceTransformer

    with open(os.path.join("data", "papers.json"), "r", encoding="utf-8") as f:
        papers = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []
    for paper in papers:
        embedding = model.encode(paper["content"], convert_to_numpy=True)
        docs.append({
            "title": paper["title"],
            "content": paper["content"],
            "embedding": embedding,
        })
    return docs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve(question: str, **kwargs) -> List[Dict[str, Any]]:
    """Retrieve the top‑k documents most similar to the question."""
    docs = load_documents()
    # Encode the question using the same embedding model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode(question, convert_to_numpy=True)

    # Compute similarities and return top 3
    scored = []
    for doc in docs:
        score = cosine_similarity(query_vec, doc["embedding"])
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [item[1] for item in scored[:3]]
    return top_docs


def summarise(docs: List[Dict[str, Any]], question: str, **kwargs) -> str:
    """Summarise the retrieved documents and answer the user’s question."""
    # Concatenate the documents into a single context string
    context = "\n\n".join([f"Title: {d['title']}\nContent: {d['content']}" for d in docs])
    prompt = (
        "You are a helpful research assistant.  Given the following research paper abstracts and a question, "
        "write a concise answer (3–5 sentences) and an explanation suitable for a beginner.\n\n"
        f"Question: {question}\n\n"
        f"Research Papers:\n{context}\n\n"
        "Answer:"
    )

    # Use OpenAI’s chat completion API to generate the summary
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or a different model if you prefer
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message["content"].strip()


def build_agent() -> Graph:
    """Construct and compile the LangGraph agent."""
    graph = Graph()

    # Define the nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("summarise", summarise)

    # Edges: question goes to retrieve; output of retrieve goes to summarise
    graph.set_entrypoint("retrieve")
    graph.connect("retrieve", "summarise")
    graph.set_finish("summarise")

    return graph.compile()


if __name__ == "__main__":
    # Example usage: run the agent in the console
    question = input("Enter your question: ")
    agent = build_agent()
    result = agent.invoke({"question": question})
    print("\nResponse:\n", result)
