"""
rag_pipeline.py

Core RAG logic, kept separate from the UI on purpose so app.py stays clean
and this file could technically be reused for a CLI version or a different
UI later without touching the actual pipeline code.

Same logic as the notebook version, just split into functions instead of
notebook cells, and adjusted a bit to work with file uploads (streamlit gives
you a file object, not a path, when someone uploads something through the
browser).
"""

import re
import faiss
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"


# ---------------------------------------------------------------------------
# document loading
# ---------------------------------------------------------------------------

def load_document(file_path=None, file_obj=None):
    """
    loads text either from a path on disk (file_path) or from an uploaded
    file object (file_obj) - streamlit's file_uploader gives you a file-like
    object, not an actual path, so needed to handle both cases
    """
    if file_obj is not None:
        name = file_obj.name.lower()
        if name.endswith(".pdf"):
            reader = PdfReader(file_obj)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            return file_obj.read().decode("utf-8")

    # otherwise treat it as a normal file path
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=400, overlap=50):
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 20:   # skip tiny leftover pieces at the end
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# embeddings + vector store
# ---------------------------------------------------------------------------

def load_embedding_model():
    """
    tries to load the sentence-transformers model. if there's no internet
    (or first-time download fails for some other reason) this falls back to
    returning None, and the rest of the code switches to tf-idf instead.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model
    except Exception as e:
        print("embedding model failed to load:", e)
        return None


def build_vector_index(chunks, embed_model):
    """
    embeds every chunk and stores the vectors in a faiss index.
    returns (index, vectorizer, embedding_dim) - vectorizer is only used
    when embed_model is None (tf-idf fallback), otherwise it's None
    """
    if embed_model is not None:
        vectors = embed_model.encode(chunks, convert_to_numpy=True).astype("float32")
        vectorizer = None
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(chunks).toarray().astype("float32")

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, vectorizer, vectors.shape[1]


def retrieve_chunks(query, chunks, index, embed_model, vectorizer, top_k=3):
    if embed_model is not None:
        query_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    else:
        query_vec = vectorizer.transform([query]).toarray().astype("float32")

    faiss.normalize_L2(query_vec)
    scores, indexes = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indexes[0]):
        results.append({
            "chunk": chunks[idx],
            "score": round(float(score), 4)
        })
    return results


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def load_generator():
    """same idea as load_embedding_model - falls back to None if it can't load"""
    try:
        from transformers import pipeline
        gen = pipeline("text2text-generation", model=GENERATION_MODEL_NAME)
        return gen
    except Exception as e:
        print("generation model failed to load:", e)
        return None


def generate_with_llm(query, retrieved, generator):
    context = " ".join([r["chunk"] for r in retrieved])[:800]
    prompt = f"Answer the question using only the given context.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    output = generator(prompt, max_new_tokens=80)
    return output[0]["generated_text"].strip()


def generate_extractive(query, retrieved):
    """
    fallback answer generator, no model needed - scores each sentence in the
    retrieved chunks by word overlap with the question and returns the best
    matching ones. not real generation, just extraction, but keeps things
    working if flan-t5 can't load
    """
    full_context = " ".join([r["chunk"] for r in retrieved])
    sentences = re.split(r'(?<=[.!?])\s+', full_context)

    q_words = set(re.findall(r"\w+", query.lower()))
    stopwords = {"what", "is", "are", "the", "a", "an", "of", "in", "to",
                 "how", "does", "do", "explain", "define"}
    q_words = q_words - stopwords

    scored = []
    for sent in sentences:
        sent_words = set(re.findall(r"\w+", sent.lower()))
        overlap = len(q_words & sent_words)
        if overlap > 0:
            scored.append((overlap, sent.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return "Sorry, I couldn't find a relevant answer for this in the document."

    return " ".join([s for _, s in scored[:3]])


def answer_question(query, chunks, index, embed_model, vectorizer, generator, top_k=3):
    """
    ties the whole pipeline together - retrieve, then generate (with fallback
    if the model isn't loaded or generation fails for some reason)
    """
    retrieved = retrieve_chunks(query, chunks, index, embed_model, vectorizer, top_k=top_k)

    if generator is not None:
        try:
            answer = generate_with_llm(query, retrieved, generator)
        except Exception as e:
            print("generation failed, falling back to extractive method:", e)
            answer = generate_extractive(query, retrieved)
    else:
        answer = generate_extractive(query, retrieved)

    return answer, retrieved
