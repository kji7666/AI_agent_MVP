from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    # --- 靜態資訊 ---
    agent_name: str
    agent_summary: str
    
    # --- 動態環境 ---
    current_time: str # 格式必須為 "%Y-%m-%d %I:%M %p"
    observations: List[str]
    
    # --- 內部狀態 ---
    relevant_memories: List[Document]
    daily_plan: List[Dict]      # L1 長期計畫
    short_term_plan: List[Dict] # L2 短期細節
    
    # 格式: "2025-06-01 09:30 AM"
    busy_until: Optional[str] 
    
    # 用於內部 Graph 流程控制 (不會存入 DB)
    skip_thinking: Optional[bool]

    # --- 輸出 ---
    current_action: Optional[str]
    current_emoji: Optional[str]

    current_daily_block_activity: Optional[str]

    