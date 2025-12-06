from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    # --- 靜態資訊 ---
    agent_name: str
    agent_summary: str
    
    # --- 動態環境 ---
    current_time: str
    observations: List[str]  # 這一輪看到的環境變化
    
    # --- 內部狀態 ---
    relevant_memories: List[Document] # 這一輪檢索到的記憶
    daily_plan: List[Dict] # 當天的計畫 (從 Planning 模組生成)
    
    # --- 輸出 ---
    current_action: Optional[str] # 代理人決定做什麼
    current_emoji: Optional[str]  # 動作對應的表情符號 (增加趣味性)