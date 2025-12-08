import sys
import os
from datetime import datetime, timedelta

# 加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.retriever import GenerativeRetriever

def test_memory_logic():
    print("========================================")
    print("🧠 TESTING MEMORY STREAM LOGIC")
    print("========================================")

    # 1. 初始化
    retriever = GenerativeRetriever(collection_name="test_memory_logic")
    
    # 為了測試方便，我們先清空舊資料 (Chroma 的 delete collection 比較麻煩，這裡我們用 unique collection name 即可)
    
    print("\n[Step 1] Adding Memories (with LLM scoring)...")
    
    now = datetime.now()
    
    # 情境 A: 很久以前的重要記憶 (分手)
    retriever.add_memory(
        "I broke up with my girlfriend, it was devastating.", 
        created_at=now - timedelta(days=7)
    )
    
    # 情境 B: 剛剛發生的瑣事 (吃早餐)
    retriever.add_memory(
        "I had oatmeal for breakfast.", 
        created_at=now - timedelta(hours=1)
    )
    
    # 情境 C: 剛剛發生的工作 (寫程式)
    retriever.add_memory(
        "I am writing Python code for the agent project.", 
        created_at=now - timedelta(minutes=10)
    )

    print("\n[Step 2] Retrieving Context...")
    query = "How are you feeling recently?"
    print(f"❓ Query: {query}")
    
    results = retriever.retrieve(query, k=2)
    
    print("\n[Analysis]")
    if any("broke up" in d.page_content for d in results):
        print("✅ SUCCESS: The agent remembered the 'break up' despite it being old (High Importance wins!)")
    else:
        print("❌ FAIL: The agent forgot the important event.")

if __name__ == "__main__":
    test_memory_logic()