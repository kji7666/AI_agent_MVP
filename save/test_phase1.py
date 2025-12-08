import uuid
from datetime import datetime
from src.schema import MemoryObject
from src.utils import get_llm, get_vector_store

def test_infrastructure():
    print("=== 開始測試 Phase 1 基礎設施 ===")

    # 1. 測試 LLM 連線
    print("\n1. 測試 LLM 連線 (School Ollama API)...")
    try:
        llm = get_llm()
        response = llm.invoke("Hello! Are you working?")
        print(f"✅ LLM 回應成功: {response.content}")
    except Exception as e:
        print(f"❌ LLM 連線失敗: {e}")
        return

    # 2. 測試 Vector DB 與 Schema
    print("\n2. 測試 ChromaDB 與 Memory Schema...")
    try:
        vector_store = get_vector_store()
        
        # 建立一個測試記憶物件
        test_memory = MemoryObject(
            id=str(uuid.uuid4()),
            content="Agent saw a red ball in the park.",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance_score=2,
            type="observation"
        )
        
        print(f"   準備寫入記憶: {test_memory.content}")

        # 寫入 DB (add_texts 接受 text 和 metadata)
        vector_store.add_texts(
            texts=[test_memory.content],
            metadatas=[test_memory.to_metadata()],
            ids=[test_memory.id]
        )
        print("✅ 寫入成功")

        # 測試檢索 (Embedding Search)
        print("   執行語意搜尋: 'What did the agent see?'")
        results = vector_store.similarity_search("What did the agent see?", k=1)
        
        if results:
            doc = results[0]
            print(f"✅ 檢索成功! 內容: {doc.page_content}")
            print(f"   Metadata: {doc.metadata}")
        else:
            print("❌ 檢索失敗: 未找到相關文件")
            
    except Exception as e:
        print(f"❌ DB 測試失敗: {e}")

if __name__ == "__main__":
    test_infrastructure()