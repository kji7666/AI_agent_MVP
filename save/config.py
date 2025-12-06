import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
API_HOST = os.getenv("API_HOST")
MODEL_NAME = os.getenv("MODEL_NAME")

# ChromaDB 的儲存路徑 (在 Docker 內)
DB_PERSIST_DIRECTORY = "/app/data/chroma_db"