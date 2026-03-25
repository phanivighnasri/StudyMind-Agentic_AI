from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


def build_vectorstore(file_path, persist_directory):

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    texts = [doc.page_content.strip() for doc in docs if          doc.page_content.strip()]

# Remove very small/invalid chunks
    texts = [t for t in texts if len(t) > 20]

# If still empty → raise clear error
    if not texts:
      raise ValueError("No valid text found in document. PDF may be empty or scanned.")

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)

    os.makedirs(persist_directory, exist_ok=True)

    with open(os.path.join(persist_directory, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(persist_directory, "vectors.pkl"), "wb") as f:
        pickle.dump(vectors, f)

    with open(os.path.join(persist_directory, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    return True
