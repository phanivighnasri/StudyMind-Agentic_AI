import os
from dotenv import load_dotenv

load_dotenv()

# API — env file uses GROQ_API_KEY (not GROK)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

# Vector DB paths
CONTENT_DB_PATH = "./db/content_db"

# Quiz results persist here across sessions
QUIZ_RESULTS_PATH = "./db/quiz_results.json"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# If max TF-IDF cosine similarity < this threshold → "Out of syllabus"
OUT_OF_SYLLABUS_THRESHOLD = 0.05
