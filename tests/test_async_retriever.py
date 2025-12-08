import asyncio
import sys
import os
import time
from datetime import datetime, timedelta

# 加入專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.retriever import GenerativeRetriever

async def main():
    print("========================================")
    print("🚀 TESTING ASYNC RETRIEVER (P0 & P2)")
    print("========================================")

    # 1. 初始化
    print("\n[Init] Starting Retriever...")
    # 使用一個全新的 collection 避免髒資料
    retriever = GenerativeRetriever(collection_name="test_async_v1")
    
    # 給它一點時間啟動背景任務
    await asyncio.sleep(1)

    # 2. 測試寫入速度 (使用 Local LLM)
    print("\n[Step 1] Adding Memories (Benchmarking Local LLM)...")
    
    memories = [
        "I am brushing my teeth.",           # 應該低分
        "I found a lost puppy in the rain.", # 應該高分
        "I am coding a new AI project.",     # 中等
    ]

    start_time = time.time()
    
    for mem in memories:
        print(f"   📝 Adding: '{mem}'")
        await retriever.add_memory(mem)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / len(memories)
    print(f"   ⏱️  Average Time per Memory: {avg_time:.2f}s")
    
    if avg_time < 1.0:
        print("   ✅ Local LLM is FAST! (System 2 would take >2s)")
    else:
        print("   ⚠️ Local LLM is a bit slow. Check if GPU is enabled or CPU is busy.")

    # 3. 測試檢索與背景更新
    print("\n[Step 2] Retrieving & Background Update...")
    query = "What meaningful things happened?"
    
    # 檢索
    results = await retriever.retrieve(query, k=2)
    
    for i, doc in enumerate(results):
        print(f"   🔍 Rank {i+1}: {doc.page_content} (Imp Score: {doc.metadata.get('importance')})")

    print("\n   💤 Waiting 2 seconds to let Background Flusher work...")
    await asyncio.sleep(2)
    
    # 驗證 Flusher 是否有運作 (這部分只能看 Console Log 是否有噴錯，或是看 Docker Log)
    print("   ✅ Retrieval loop finished without blocking.")

    # 4. 結束測試
    # 取消背景任務 (在真實 Server 中不需要這步，但在 Script 中要優雅退出)
    retriever.flusher_task.cancel()
    try:
        await retriever.flusher_task
    except asyncio.CancelledError:
        print("\n👋 Flusher task stopped properly.")

if __name__ == "__main__":
    asyncio.run(main())