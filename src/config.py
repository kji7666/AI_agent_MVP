import os
from dotenv import load_dotenv

# 強制載入 .env，如果找不到會報錯提醒
if not load_dotenv():
    print("Warning: .env file not found. Ensure environment variables are set.")

class Config:
    # LLM Settings
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_HOST = os.getenv("LLM_HOST")
    LLM_MODEL = os.getenv("LLM_MODEL")

    # local 小模型設定 (用於 Scoring)
    FAST_LLM_HOST = "http://localhost:11434" # 指向本地 Docker
    FAST_LLM_MODEL = "llama3.2:1b"
    
    # Chroma Settings
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    CHROMA_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}"

    # Embedding Model (Local)
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def validate(self):
        """簡單的驗證邏輯，確保關鍵變數存在"""
        if not self.LLM_API_KEY:
            raise ValueError("Missing LLM_API_KEY in .env")
        if not self.LLM_HOST:
            raise ValueError("Missing LLM_HOST in .env")

config = Config()
config.validate()