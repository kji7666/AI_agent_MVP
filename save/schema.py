from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Literal

class MemoryObject(BaseModel):
    """
    代表單一條記憶 (Observation, Thought, or Reflection)。
    這是存入 VectorDB 的基本單位。
    """
    id: str = Field(..., description="Unique UUID for the memory")
    content: str = Field(..., description="The textual content of the memory")
    
    # 時間戳記
    created_at: datetime
    last_accessed: datetime
    
    # 核心屬性
    importance_score: int = Field(..., description="1-10 score, 10 is most important")
    type: Literal["observation", "reflection", "plan"] = "observation"
    
    def to_metadata(self):
        """
        轉換為 ChromaDB 接受的 metadata 格式 (Chroma 不支援 datetime 物件，需轉字串)
        """
        return {
            "type": self.type,
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            # 可以在此加入其他 metadata，例如關聯的 agent_id
        }

    class Config:
        # 允許 datetime 序列化
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }