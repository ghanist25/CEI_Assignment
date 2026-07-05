"""
app.py

Streamlit UI for the RAG project. Run this with:
    streamlit run app.py

Kept all the actual RAG logic in rag_pipeline.py, this file is just the UI -
takes user input (uploaded doc, question, settings) and calls into that
module to get results.
"""

import random

import pandas as pd
import streamlit as st

import rag_pipeline as rp

st.set_page_config(page_title="Document QA System (RAG)", layout="wide")

st.title("Document Question Answering System")
st.caption("Mini project - DBMS notes as the custom dataset, RAG pipeline built from scratch")


# ---------------------------------------------------------------------------
# sidebar - pipeline settings
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 800, 400, step=50)
overlap = st.sidebar.slider("Chunk overlap (characters)", 0, 150, 50, step=10)
top_k = st.sidebar.slider("Chunks to retrieve (top k)", 1, 5, 3)

st.sidebar.caption(
    "Bigger chunk size = more context per chunk but less precise retrieval. "
    "These get re-applied every time you change them, the doc gets re-chunked "
    "and re-indexed automatically."
)


# ---------------------------------------------------------------------------
# load models once - cached so streamlit doesn't reload them on every rerun
# (streamlit reruns the whole script top to bottom on every interaction,
# without caching this would reload the embedding model + flan-t5 on every
# single click, which would be painfully slow)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return rp.load_embedding_model()


@st.cache_resource(show_spinner="Loading generation model (flan-t5)...")
def get_generator():
    return rp.load_generator()


embed_model = get_embedding_model()
generator = get_generator()

if embed_model is None:
    st.sidebar.warning("Embedding model couldn't load (no internet?) - using TF-IDF fallback instead.")
if generator is None:
    st.sidebar.warning("flan-t5 couldn't load (no internet?) - using extractive fallback instead.")


# ---------------------------------------------------------------------------
# document loading
# ---------------------------------------------------------------------------

st.subheader("1. Document")

uploaded_file = st.file_uploader(
    "Upload a .txt or .pdf file, or leave empty to use the bundled DBMS notes",
    type=["txt", "pdf"],
)

if uploaded_file is not None:
    raw_text = rp.load_document(file_obj=uploaded_file)
    doc_name = uploaded_file.name
else:
    raw_text = rp.load_document(file_path="dbms_notes.txt")
    doc_name = "dbms_notes.txt (bundled sample)"

st.write(f"Loaded **{doc_name}** — {len(raw_text)} characters")


# ---------------------------------------------------------------------------
# chunk + build the vector index
# cached on the actual text + settings, so it only rebuilds when the doc or
# the sliders actually change, not on every single rerun
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chunking document and building the vector index...")
def process_document(text, chunk_size, overlap):
    chunks = rp.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    index, vectorizer, dim = rp.build_vector_index(chunks, embed_model)
    return chunks, index, vectorizer, dim


chunks, index, vectorizer, embed_dim = process_document(raw_text, chunk_size, overlap)

st.write(f"Split into **{len(chunks)}** chunks, embedding dimension = **{embed_dim}**")


# ---------------------------------------------------------------------------
# question answering
# ---------------------------------------------------------------------------

st.subheader("2. Ask a question")

query = st.text_input("Type your question about the document here")
ask_clicked = st.button("Get answer", type="primary")

if ask_clicked and query.strip():
    with st.spinner("Retrieving context and generating an answer..."):
        answer, retrieved = rp.answer_question(
            query, chunks, index, embed_model, vectorizer, generator, top_k=top_k
        )

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("See retrieved chunks (what the answer is based on)"):
        for r in retrieved:
            st.write(f"**similarity score: {r['score']}**")
            st.write(r["chunk"])
            st.divider()

elif ask_clicked:
    st.warning("Type a question first.")


# ---------------------------------------------------------------------------
# validation log - dynamic sample questions with keyword based accuracy check
# only relevant for the bundled dbms notes, since it's the question bank the
# accuracy check was written against
# ---------------------------------------------------------------------------

st.subheader("3. Validation check (optional)")
st.caption("Runs a random sample of test questions against the document and checks retrieval accuracy. Built for the bundled DBMS notes specifically.")

question_bank = [
    ("What is an entity in the ER model?", ["entity", "attribute"]),
    ("Explain functional dependency", ["functional dependency", "determine"]),
    ("What is two phase locking?", ["two phase locking", "growing phase", "shrinking phase"]),
    ("What does BCNF mean?", ["boyce codd", "super key"]),
    ("What happens during a deadlock?", ["deadlock", "waiting"]),
    ("What is normalization?", ["normalization", "redundancy"]),
    ("What are the ACID properties?", ["atomicity", "consistency", "isolation", "durability"]),
    ("What is a transaction?", ["transaction", "logical unit"]),
    ("What is serializability?", ["serializable", "schedule"]),
    ("What is relational algebra?", ["relational algebra", "selection", "projection"]),
]

if st.button("Run validation check"):
    sample_questions = random.sample(question_bank, 5)
    log_rows = []
    for q, expected_keywords in sample_questions:
        top_chunk = rp.retrieve_chunks(q, chunks, index, embed_model, vectorizer, top_k=1)[0]
        chunk_lower = top_chunk["chunk"].lower()
        hit = any(kw in chunk_lower for kw in expected_keywords)
        log_rows.append({
            "query": q,
            "retrieval_score": top_chunk["score"],
            "expected_keywords": ", ".join(expected_keywords),
            "keyword_found": hit,
        })

    log_df = pd.DataFrame(log_rows)
    accuracy = log_df["keyword_found"].mean() * 100

    st.dataframe(log_df, use_container_width=True)
    st.write(f"**Retrieval accuracy on this sample: {accuracy:.1f}%**")
    st.caption(
        "Note - BCNF tends to fail here since the source notes spell out "
        "'Boyce Codd Normal Form' and never use the acronym anywhere, so "
        "there's no shared word for keyword based retrieval to latch onto."
    )


# ---------------------------------------------------------------------------
# system metrics report
# ---------------------------------------------------------------------------

st.subheader("4. System metrics report")

metrics_report = {
    "document": doc_name,
    "document_length_chars": len(raw_text),
    "chunking_strategy": "fixed length with overlap",
    "chunk_size_chars": chunk_size,
    "chunk_overlap_chars": overlap,
    "num_chunks": len(chunks),
    "embedding_model": rp.EMBEDDING_MODEL_NAME if embed_model is not None else "TF-IDF (fallback, no internet)",
    "embedding_dimension": embed_dim,
    "vector_store": "FAISS (IndexFlatIP, cosine similarity via normalized vectors)",
    "vectors_indexed": index.ntotal,
    "generation_model": rp.GENERATION_MODEL_NAME if generator is not None else "extractive fallback (no model, no internet)",
    "retrieval_top_k": top_k,
}

report_df = pd.DataFrame(list(metrics_report.items()), columns=["metric", "value"])
st.table(report_df)
