import asyncio
import uuid
import numpy as np
from datetime import datetime
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.memory.models import Memory
from src.memory.importance import get_importance_scorer
from src.llm_factory import get_embeddings

class GenerativeRetriever:
    """
    add_memory (新增記憶) ---> 寫入 DB

    retriever (回想) ---> 更新 last_access_time
                           |
                           +--> push memory.id 到 update_queue (可選)
    _background_flusher (背景守護) ---> 取出 queue 中的 id
                                        |
                                        +--> asyncio.to_thread(_batch_update_access_time)
    _batch_update_access_time (同步) ---> 讀取 metadata -> 更新 last_accessed_at -> 寫回 DB
    """
    def __init__(self, collection_name: str, decay_factor: float = 0.995):
        """
        初始化檢索器
        Args:
            collection_name: ChromaDB 的集合名稱
            decay_factor: 記憶遺忘係數 (論文預設 0.995)
        """
        # 用來將文字轉成向量 (vector) 儲存於向量資料庫中。
        self.embeddings = get_embeddings()
        
        # 初始化 Chroma Vector Database (向量搜尋使用 cosine similarity)
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client_settings=None, 
            collection_metadata={"hnsw:space": "cosine"} # 使用餘弦相似度
        )
        
        # 使用本地小模型的評分器
        self.importance_scorer = get_importance_scorer()
        # 記憶衰退係數
        self.decay_factor = decay_factor
        
        # 存放待更新記憶，不阻塞主執行流程
        self.update_queue = asyncio.Queue()
        # 建立背景工作任務，持續處理更新佇列 → 將更新後的記憶批量寫回 Chroma
        self.flusher_task = asyncio.create_task(self._background_flusher())
        print(f"🚀 [Retriever] Initialized with Async Write-back & Local LLM Scoring.")

    async def _background_flusher(self):
        """
        [Background Task] 定期將 last_accessed_at 寫回 DB
        避免檢索時因為寫入 DB 而變慢。
        """
        while True:
            try:
                ids_to_update = []
                # 嘗試將 Queue 中的所有任務取出
                while not self.update_queue.empty():
                    ids_to_update.append(self.update_queue.get_nowait())
                
                if ids_to_update:
                    # 避免同一個 ID 被多次更新, 去重複
                    unique_ids = list(set(ids_to_update))
                    current_time = datetime.now().timestamp()
                    
                    # ChromaDB 寫入是 同步 & 阻塞式 I/O, 要 await (需放入其他 thread 避免阻塞)
                    await asyncio.to_thread(self._batch_update_access_time, unique_ids, current_time)
                    
                # 每 5 秒 loop 一次
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                print("Flusher task cancelled.")
                break
            except Exception as e:
                print(f"Flusher Error: {e}")
                await asyncio.sleep(5)

    def _batch_update_access_time(self, ids: List[str], timestamp: float):
        """同步的 Chroma 批量更新邏輯 (被上面的 async 包裝)"""
        try:
            # 必須先取出 metadata 的其他欄位，因為 update time 會覆蓋整個 metadata
            existing_data = self.vector_store.get(ids=ids)
            
            if existing_data and existing_data['ids']:
                new_metadatas = []
                for meta in existing_data['metadatas']:
                    # 更新時間戳
                    meta['last_accessed_at'] = timestamp
                    new_metadatas.append(meta)
                
                # 寫回 DB
                self.vector_store.update_documents(
                    ids=existing_data['ids'],
                    metadatas=new_metadatas
                )
        except Exception as e:
            print(f"   ⚠️ Chroma Update Failed: {e}")

    async def add_memory(self, content: str, created_at: datetime = None, type: str = "observation"):
        """
        [Async] 新增記憶
        1. 呼叫本地 LLM 評分 (Fast)
        2. 寫入 Vector DB
        """
        if created_at is None:
            created_at = datetime.now()

        # 計算重要性
        # 使用 to_thread 避免 invoke 阻塞 Event Loop
        try:
            score = await asyncio.to_thread(
                self.importance_scorer.invoke, # blocking method
                {"memory_content": content} # method param
            )
        except Exception as e:
            print(f"   ⚠️ Scoring failed, defaulting to 1. Error: {e}")
            score = 1

        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            created_at=created_at,
            last_accessed_at=created_at,
            importance=score,
            type=type
        )
        
        # 寫入 Vector DB (Async)
        payload = memory.to_chroma_payload()
        await asyncio.to_thread(
            self.vector_store.add_documents,
            [Document(page_content=payload["page_content"], metadata=payload["metadata"])]
        )

        

    async def retrieve(self, query: str, now: datetime = None, k: int = 5, fetch_k: int = 100) -> List[Document]:
        """
        [Async] 混合檢索核心邏輯
        """
        if now is None:
            now = datetime.now()

        # 向量檢索 (Relevance) - 抓取較大範圍的候選集
        # 使用 to_thread 因為 similarity_search 是同步且耗時的
        candidates = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query,
            k=fetch_k
        )

        if not candidates:
            return []

        # 計算混合分數
        # 論文公式: Score = a*Recency + b*Importance + c*Relevance
        docs = [doc for doc, _ in candidates]
        # A. Relevance (Similarity)
        # Chroma 回傳的是 Distance (0~2)，轉為 Similarity
        relevance_scores = [1.0 - dist for _, dist in candidates]
        # B. Importance (1-10 -> 0-1)
        importance_scores = [doc.metadata.get("importance", 1) / 10.0 for doc in docs]
        # C. Recency (Decay Factor)
        recency_scores = []
        for doc in docs:
            last_accessed_ts = doc.metadata.get("last_accessed_at", now.timestamp())
            last_accessed = datetime.fromtimestamp(last_accessed_ts)
            hours_passed = (now - last_accessed).total_seconds() / 3600
            hours_passed = max(0, hours_passed)
            recency = self.decay_factor ** hours_passed
            recency_scores.append(recency)

        #  (Min-Max Scaling)
        def normalize(arr):
            a = np.array(arr)
            if np.max(a) == np.min(a):
                return a # 如果數值都一樣，就不縮放
            return (a - np.min(a)) / (np.max(a) - np.min(a))

        norm_recency = normalize(recency_scores)
        norm_importance = normalize(importance_scores)
        norm_relevance = normalize(relevance_scores)

        # 加權總分 (權重可調整)
        alpha, beta, gamma = 1.0, 1.0, 1.0
        total_scores = (alpha * norm_recency) + (beta * norm_importance) + (gamma * norm_relevance)

        # 5. 排序並取出 Top-K
        # argsort 是從小到大，所以用 [::-1] 反轉
        top_indices = np.argsort(total_scores)[::-1][:k]
        
        final_results = []
        for idx in top_indices:
            doc = docs[idx]
            final_results.append(doc)
            
            # 將此 ID 加入更新佇列
            # 我們不等待它寫入，直接繼續
            doc_id = doc.metadata.get("id")
            if doc_id:
                await self.update_queue.put(doc_id)

        return final_results