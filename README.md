# Research Assistant Agent with Hybrid Search

This repository contains a small demonstration of a research assistant built with **LangGraph** and **LangChain**.  The agent uses a **hybrid search** strategy – combining semantic embeddings and keyword matching – to retrieve relevant research papers from a local database and then summarise them for a user.

## Features

- **Hybrid search**: Uses both dense (vector) and sparse (keyword) representations to find the most relevant documents.
- **Small sample dataset**: Includes a few example research abstracts so you can try the system without needing to supply your own data.
- **LangGraph orchestration**: Demonstrates how to build a multi‑step workflow that retrieves, summarises and explains research topics.
- **Streamlit interface**: Simple web UI for asking questions and displaying the agent’s responses (optional).

## Repository layout

| Path | Description |
| --- | --- |
| `data/papers.json` | Example research papers with titles and abstracts.  You can replace this with your own dataset. |
| `embed_data.py` | Script to embed the papers and store them in a MongoDB collection with a vector field. |
| `agent_flow.py` | Defines the LangGraph workflow used to answer questions by retrieving and summarising documents. |
| `app.py` | Streamlit application for interacting with the agent. |
| `requirements.txt` | Python dependencies. |

## Getting started

1. **Install dependencies**

   You should create a virtual environment and install the dependencies listed in `requirements.txt`:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set environment variables**

   - `MONGO_URI` – connection string for your MongoDB Atlas or local MongoDB instance.  This is used to store the indexed documents and perform vector search.  If you prefer, you can adapt the code to use another vector database (e.g. Pinecone, AstraDB).
   - `OPENAI_API_KEY` – API key for the OpenAI models used to perform summarisation.  Alternatively, you can swap in any other LLM supported by LangChain.

3. **Embed and index the papers**

   Run the embedding script to compute sentence embeddings for each paper and upload them to your MongoDB collection:

   ```bash
   python embed_data.py
   ```

   This script uses `sentence-transformers` to generate embeddings.  You can change the model in `embed_data.py` if you prefer another embedding model.

4. **Run the agent in Streamlit**

   Start the Streamlit app to interact with the research assistant via a web interface:

   ```bash
   streamlit run app.py
   ```

   Enter a question about a research topic (e.g. *“Explain the volatility of Bitcoin”*).  The agent will perform a hybrid search over the indexed documents, retrieve the most relevant abstracts and generate a summary and simplified explanation.

## Notes

This project is meant as a **portfolio demonstration**.  It uses a tiny set of sample papers for convenience.  For a real application, substitute `data/papers.json` with a larger corpus of your own documents and adjust the indexing/retrieval code accordingly.  You can also experiment with adding additional agents (e.g. for critique or explanation) using LangGraph’s modular architecture.
