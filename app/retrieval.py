import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_content(query, persist_directory, top_k=3):

    with open(os.path.join(persist_directory, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(persist_directory, "vectors.pkl"), "rb") as f:
        vectors = pickle.load(f)

    with open(os.path.join(persist_directory, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)

    # Convert query → vector
    query_vector = vectorizer.transform([query])

    # Compute similarity
    similarities = cosine_similarity(query_vector, vectors).flatten()

    # Get top_k results
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Return joined text
    return "\n\n".join([texts[i] for i in top_indices])
