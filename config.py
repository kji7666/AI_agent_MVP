# config.py
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 讀取環境變數，如果沒讀到則拋出錯誤或使用預設值
API_HOST = os.getenv("API_HOST")
LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss:20b")

if not API_HOST or not LLM_API_KEY:
    raise ValueError("❌ 錯誤: 未偵測到 API_HOST 或 LLM_API_KEY，請檢查 .env 檔案")