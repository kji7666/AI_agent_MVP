import sys
import os
import time

# 將專案根目錄加入 Path，確保可以 import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_factory import get_llm, get_embeddings
from src.config import config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

def test_environment():
    print("========================================")
    print("🛠️  GEN AGENT ENVIRONMENT SANITY CHECK")
    print("========================================")
    
    # [新增] Debug 區塊：檢查變數是否讀取成功
    print(f"🔍 DEBUG: LLM_HOST = {config.LLM_HOST}")
    if config.LLM_API_KEY:
        # 顯示前幾碼確保有讀到，不要印出完整的 key
        print(f"🔍 DEBUG: LLM_API_KEY = {config.LLM_API_KEY[:5]}... (Length: {len(config.LLM_API_KEY)})")
    else:
        print("❌ DEBUG: LLM_API_KEY is None or Empty! Check your .env file.")
        return # 直接結束

    # 1. 測試 Embedding 模型 (Local)
    print("\n[1/4] 📥 Loading Local Embeddings...")
    start_time = time.time()
    try:
        embeddings = get_embeddings()
        vector = embeddings.embed_query("測試向量化 Test Vector")
        print(f"   ✅ Success! Dimension: {len(vector)} (Time: {time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return

    # 2. 測試 Vector DB (Docker Chroma)
    print("\n[2/4] 💾 Connecting to ChromaDB (Docker)...")
    try:
        # Client Setup
        db = Chroma(
            collection_name="sanity_check_collection",
            embedding_function=embeddings,
            client_settings=None, # 使用預設 HTTP Client
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # 寫入測試
        docs = [
            Document(page_content="John likes pizza.", metadata={"source": "test"}),
            Document(page_content="John is a software engineer.", metadata={"source": "test"})
        ]
        db.add_documents(docs)
        
        # 讀取測試
        results = db.similarity_search("What is John's job?", k=1)
        if results and "engineer" in results[0].page_content:
            print(f"   ✅ Success! Retrieved: {results[0].page_content}")
        else:
            print(f"   ⚠️ Warning: Retrieval content mismatch. Got: {results}")

    except Exception as e:
        print(f"   ❌ Failed. Is Docker running? Error: {e}")
        return

    # 3. 測試 NCKU LLM API 連線
    print(f"\n[3/4] ☁️  Connecting to NCKU LLM ({config.LLM_MODEL})...")
    try:
        llm = get_llm(temperature=0.7)
        response = llm.invoke("Say 'Hello Engineer' only.")
        print(f"   ✅ Success! Response: {response.content}")
    except Exception as e:
        print(f"   ❌ Failed. Check API Key or Network. Error: {e}")
        return
        
    # 4. 測試中文/Unicode 處理
    print("\n[4/4] 🔣 Testing Unicode/Chinese Handling...")
    try:
        prompt = ChatPromptTemplate.from_template("請簡短翻譯成英文: {text}")
        chain = prompt | llm
        res = chain.invoke({"text": "人工智慧代理人"})
        print(f"   ✅ Success! Response: {res.content}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    print("\n========================================")
    print("🎉 ALL SYSTEMS GO! Ready for Phase 2.")
    print("========================================")

if __name__ == "__main__":
    test_environment()