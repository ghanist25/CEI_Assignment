# Document Question Answering System (RAG)

Mini project for DBMS - Week 7. Retrieval-Augmented Generation pipeline built
from scratch (no LangChain) with a Streamlit UI on top.

## What's in here

- `rag_pipeline.py` - the actual RAG logic: document loading, chunking,
  embeddings, FAISS vector store, generation. No UI code in here on purpose.
- `app.py` - Streamlit UI, imports rag_pipeline.py and wires it up to a
  browser interface.
- `dbms_notes.txt` - sample dataset (my own DBMS notes), used by default if
  you don't upload your own file.
- `requirements.txt` - dependencies.

## How to run this

This needs to run locally on your own machine (or any real terminal), not
inside Google Colab - a UI app is a persistent local web server, and Colab
notebooks aren't really built for that.

**1. Install the dependencies:**

```
pip install -r requirements.txt
```

**2. Run the app:**

```
streamlit run app.py
```

This should open a browser tab automatically at `http://localhost:8501`. If
it doesn't open on its own, just go to that address manually.

**3. First run will take a bit longer** - it downloads the embedding model
(`all-MiniLM-L6-v2`) and the generation model (`flan-t5-small`) the first
time, then caches them locally so later runs are fast.

## Using it

- Upload your own `.txt` or `.pdf`, or just leave it blank to use the
  bundled DBMS notes.
- Type a question in the text box and hit "Get answer".
- Expand "See retrieved chunks" to see exactly what context the answer was
  based on.
- The "Run validation check" button tests retrieval against a small
  question bank with known correct keywords, and reports an accuracy
  percentage. This part is written specifically against the DBMS notes
  dataset, so it won't make sense if you swap in a different document.
- The system metrics report at the bottom shows the actual pipeline config -
  chunk size, embedding model + dimension, vector store, generation model.

## If a model fails to load (no internet)

Both the embedding model and the generation model fall back automatically -
embeddings fall back to TF-IDF, generation falls back to picking the most
relevant sentences straight out of the retrieved chunks instead of actually
generating new text. You'll see a warning in the sidebar if either fallback
kicks in. Everything still works either way, just with slightly lower answer
quality in fallback mode.
