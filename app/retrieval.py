import pickle
import os
import re
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity

# Chunks below this score are irrelevant noise — filtered out
MIN_CHUNK_SCORE = 0.03


def _normalize_query(query: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation noise."""
    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", " ", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def _correct_spelling(query: str, vocab: set) -> str:
    """
    Replace misspelled words with the closest vocabulary match.
    Skips short words (<=3 chars) to avoid over-correcting abbreviations.
    cutoff=0.82 is conservative — only fixes clear typos.
    """
    corrected = []
    for word in query.split():
        if len(word) <= 3 or word in vocab:
            corrected.append(word)
        else:
            matches = get_close_matches(word, vocab, n=1, cutoff=0.82)
            corrected.append(matches[0] if matches else word)
    return " ".join(corrected)


def retrieve_content(
    query: str,
    persist_directory: str,
    top_k: int = 5,
    source_filter: str = None,
):
    """
    Returns (context_text: str, max_similarity_score: float).

    source_filter: if set, restricts retrieval to chunks from that filename.
    Falls back to global search if the filter matches nothing.
    """
    vec_path    = os.path.join(persist_directory, "vectorizer.pkl")
    vecs_path   = os.path.join(persist_directory, "vectors.pkl")
    texts_path  = os.path.join(persist_directory, "texts.pkl")
    sources_path = os.path.join(persist_directory, "sources.pkl")

    if not all(os.path.exists(p) for p in [vec_path, vecs_path, texts_path]):
        return "", 0.0

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(vecs_path, "rb") as f:
        vectors = pickle.load(f)
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    sources = None
    if os.path.exists(sources_path):
        with open(sources_path, "rb") as f:
            sources = pickle.load(f)

    vocab = set(vectorizer.get_feature_names_out())
    query = _normalize_query(query)
    query = _correct_spelling(query, vocab)
    query_vector = vectorizer.transform([query])

    def _rank(vec_matrix, text_list):
        """Rank rows of vec_matrix, return (top_indices, max_score, texts_subset)."""
        sims = cosine_similarity(query_vector, vec_matrix).flatten()
        ranked = sims.argsort()[-top_k:][::-1]
        ranked = [r for r in ranked if sims[r] >= MIN_CHUNK_SCORE]
        score = float(sims[ranked[0]]) if ranked else 0.0
        ctx = "\n\n---\n\n".join(text_list[r] for r in ranked)
        return ctx, score

    # ── Per-file filtering ────────────────────────────────────────────────────
    if source_filter and sources and len(sources) == len(texts):
        allowed = [i for i, s in enumerate(sources)
                   if source_filter.lower() in s.lower()]
        if allowed:
            filtered_vectors = vectors[allowed]
            filtered_texts   = [texts[i] for i in allowed]
            return _rank(filtered_vectors, filtered_texts)
        # Filter matched nothing — fall through to global search

    # ── Global search across all uploaded PDFs ────────────────────────────────
    return _rank(vectors, texts)
