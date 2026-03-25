import os
from dotenv import load_dotenv

load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

BASE_URL = "https://api.x.ai/v1"

CONTENT_DB_PATH = "./db/content_db"
SYLLABUS_DB_PATH = "./db/syllabus_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SIMILARITY_THRESHOLD = 0.65
