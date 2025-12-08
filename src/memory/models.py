from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any

class Memory(BaseModel):
    """
    代表一條記憶單元 (Observation, Reflection, or Plan)
    """
    # field : Metadata 描述, 可以給預設值 / 驗證規則, default_factory 
    # description : 在 LangChain 解析 JSON 時，Pydantic 會自動輸出這些描述文字
    id: str = Field(description="Unique UUID for the memory")
    content: str = Field(description="The natural language description")
    created_at: datetime = Field(description="When this memory was formed")
    last_accessed_at: datetime = Field(description="When this memory was last retrieved")
    importance: int = Field(description="Score from 1 (mundane) to 10 (poignant)")
    type: str = Field(default="observation", description="observation, reflection, or plan")
    
    # 用於儲存額外資訊 (例如 reflection 的 pointers)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_chroma_payload(self):
        """轉換為 ChromaDB 可儲存的格式 (Metadata 只能是 str, int, float)"""
        return {
            "page_content": self.content,
            "metadata": {
                "id": self.id,
                "created_at": self.created_at.timestamp(), # 轉為 float timestamp
                "last_accessed_at": self.last_accessed_at.timestamp(),
                "importance": self.importance,
                "type": self.type,
                **self.metadata
            }
        }