import uuid
from datetime import datetime
from typing import List
import numpy as np

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.memory.models import Memory
from src.memory.importance import get_importance_scorer
from src.llm_factory import get_embeddings

class GenerativeRetriever:
    def __init__(self, collection_name: str, decay_factor: float = 0.995):
        self.embeddings = get_embeddings()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client_settings=None, # 使用 Docker 預設
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.importance_scorer = get_importance_scorer()
        self.decay_factor = decay_factor # 論文中的遺忘係數

    def add_memory(self, content: str, created_at: datetime = None, type: str = "observation"):
        """
        新增記憶：自動計算 Embedding 與 Importance
        """
        if created_at is None:
            created_at = datetime.now()

        # 1. 呼叫 LLM 計算重要性
        try:
            score = self.importance_scorer.invoke({"memory_content": content})
        except Exception as e:
            print(f"   ⚠️ Scoring failed, defaulting to 5. Error: {e}")
            score = 5

        # 2. 建立記憶物件
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            created_at=created_at,
            last_accessed_at=created_at,
            importance=score,
            type=type
        )

        # 3. 寫入 Vector DB
        payload = memory.to_chroma_payload()
        self.vector_store.add_documents([
            Document(page_content=payload["page_content"], metadata=payload["metadata"])
        ])
        print(f"   ✅ Saved Memory (Score: {score}): {content}")

    def retrieve(self, query: str, now: datetime = None, k: int = 5, fetch_k: int = 100) -> List[Document]:
        """
        混合檢索核心邏輯：
        1. 先用 Vector Search 抓取 Top-100 (Relevance)
        2. 計算 Recency 與 Importance
        3. 加權總分排序，回傳 Top-K
        """
        if now is None:
            now = datetime.now()

        # 1. 向量檢索 (Relevance) - 抓取較大範圍的候選集
        # results_with_score 回傳 (Document, distance)
        # Cosine Distance 越小越好，我們轉成 Similarity (1 - dist)
        candidates = self.vector_store.similarity_search_with_score(query, k=fetch_k)

        if not candidates:
            return []

        # 2. 準備特徵向量
        relevance_scores = []
        recency_scores = []
        importance_scores = []
        
        docs = []

        for doc, distance in candidates:
            docs.append(doc)
            
            # A. Relevance (歸一化到 0-1)
            # Chroma Cosine distance 範圍通常是 0~2，這裡簡單轉為相似度 sim
            sim = 1 - distance
            relevance_scores.append(sim)
            
            # B. Importance (正規化 1-10 -> 0-1)
            imp = doc.metadata.get("importance", 1)
            importance_scores.append(imp / 10.0)
            
            # C. Recency (指數衰減)
            # 論文公式：decay_factor ^ (hours_passed)
            last_accessed = datetime.fromtimestamp(doc.metadata.get("last_accessed_at"))
            hours_passed = (now - last_accessed).total_seconds() / 3600
            recency = self.decay_factor ** hours_passed
            recency_scores.append(recency)

        # 3. 歸一化 (Min-Max Scaling)
        # 讓三個指標都在 0-1 之間，這樣加權才有意義
        def normalize(arr):
            arr = np.array(arr)
            if np.max(arr) == np.min(arr): return arr # 避免除以零
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # 注意：雖然上面已經做了簡單正規化，但為了讓混合分數分佈更廣，
        # 我們通常會對這群 candidates 再做一次 min-max
        norm_recency = normalize(recency_scores)
        norm_importance = normalize(importance_scores)
        norm_relevance = normalize(relevance_scores)

        # 4. 計算總分
        # 論文權重：alpha=1, beta=1, gamma=1 (可調整)
        alpha, beta, gamma = 1.0, 1.0, 1.0
        total_scores = (alpha * norm_recency) + (beta * norm_importance) + (gamma * norm_relevance)

        # 5. 排序並取出 Top-K
        # argsort 回傳的是從小到大的 index，所以要反轉 [::-1]
        top_indices = np.argsort(total_scores)[::-1][:k]
        
        final_results = []
        for idx in top_indices:
            doc = docs[idx]
            # 更新 last_accessed_at (因為被想起來了)
            # 這裡暫時不寫回 DB 以免拖慢速度，但在完整系統中應該要更新
            # doc.metadata['last_accessed_at'] = now.timestamp()
            final_results.append(doc)
            
            # Debug 輸出，讓你看到分數是怎麼算出來的
            print(f"   🔍 Rank {len(final_results)}: {doc.page_content}")
            print(f"      Scores -> Recency: {norm_recency[idx]:.2f}, Imp: {norm_importance[idx]:.2f}, Rel: {norm_relevance[idx]:.2f} | Total: {total_scores[idx]:.2f}")

        return final_results