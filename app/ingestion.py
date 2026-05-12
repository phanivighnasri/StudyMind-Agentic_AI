from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import json
from datetime import datetime


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _load_metadata(persist_directory: str) -> dict:
    path = os.path.join(persist_directory, "metadata.json")
    if not os.path.exists(path):
        return {"files": []}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"files": []}


def get_uploaded_files(persist_directory: str) -> list:
    """Return list of uploaded file metadata dicts (filename, chunks, uploaded_at)."""
    return _load_metadata(persist_directory).get("files", [])


# ── Vectorstore builder (append mode) ────────────────────────────────────────

def build_vectorstore(file_path, persist_directory):
    """
    Index a new PDF and APPEND its content to the existing vectorstore.
    Re-fits TF-IDF on all accumulated texts so vocabulary stays current.
    """
    # Resolve filename first — used in both sources tracking and metadata
    filename = os.path.basename(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    new_texts = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
    new_texts = [t for t in new_texts if len(t) > 20]

    if not new_texts:
        raise ValueError("No valid text found in document. PDF may be empty or scanned.")

    os.makedirs(persist_directory, exist_ok=True)
    texts_path = os.path.join(persist_directory, "texts.pkl")

    # Load existing corpus and append — preserves previously uploaded PDFs
    existing_texts = []
    if os.path.exists(texts_path):
        with open(texts_path, "rb") as f:
            existing_texts = pickle.load(f)

    all_texts = existing_texts + new_texts

    # Load existing per-chunk source map (chunk_index → filename)
    sources_path = os.path.join(persist_directory, "sources.pkl")
    existing_sources = []
    if os.path.exists(sources_path):
        with open(sources_path, "rb") as f:
            existing_sources = pickle.load(f)
    all_sources = existing_sources + [filename] * len(new_texts)

    # Re-fit vectorizer on entire corpus so new vocabulary is included
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_texts)

    with open(os.path.join(persist_directory, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(persist_directory, "vectors.pkl"), "wb") as f:
        pickle.dump(vectors, f)
    with open(texts_path, "wb") as f:
        pickle.dump(all_texts, f)
    with open(sources_path, "wb") as f:
        pickle.dump(all_sources, f)

    # Update metadata — deduplicate by filename so re-uploads replace old entry
    meta = _load_metadata(persist_directory)
    meta["files"] = [f for f in meta["files"] if f["filename"] != filename]
    meta["files"].append({
        "filename": filename,
        "chunks": len(new_texts),
        "uploaded_at": datetime.now().isoformat(),
    })
    with open(os.path.join(persist_directory, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return len(new_texts)
