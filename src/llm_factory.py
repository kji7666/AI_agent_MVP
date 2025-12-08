import os
import ollama
from typing import Any, List, Optional, Dict
from pydantic import Field, PrivateAttr

# LangChain Core Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import config

class NCKUCustomLLM(BaseChatModel):
    """
    LangChain → NCKUCustomLLM() → 直接觸發 Ollama API → NCKU Server
    直接使用官方 ollama library 連接 NCKU, 完全繞過 langchain-ollama 的連線邏輯。
    這保證了 Header 一定會被發送。
    1. 繼承 BaseChatModel
    2. 實作 generate
    """
    # LangChain 會嘗試將 模型的屬性轉成 JSON / dict (序列化)
    model_name: str = Field(default=config.LLM_MODEL)
    temperature: float = Field(default=0.7)
    _client: ollama.Client = PrivateAttr() # 設定不被序列化, 因為有 key

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = ollama.Client(
            host=config.LLM_HOST,
            headers={'Authorization': f'Bearer {config.LLM_API_KEY}'},
            timeout=180
        )
        # client 就是一個「用來連線到遠端服務的物件」把「發送請求 → 收到回應」這件事包裝起來

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        實作 LangChain 的生成介面
        把 langChain 的參數格式改成 ollama dict
        塞進 client.chat 取得 response
        包裝成 LangChain 格式回傳
        """
        
        # 轉換訊息格式 (LangChain Message -> Ollama Dict)
        ollama_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            
            ollama_messages.append({
                "role": role,
                "content": msg.content
            })

        # 呼叫 NCKU API (使用官方 Client)
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=ollama_messages,
                options={
                    "temperature": self.temperature,
                }
            )
            
            generated_text = response['message']['content']
            
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=generated_text))]
            )
            
        except Exception as e:
            print(f"NCKU API Error: {e}")
            raise e

    @property
    def _llm_type(self) -> str:
        return "ncku-custom-wrapper"

# ==========================================
# factory function
# ==========================================

def get_llm(temperature=0.7, json_mode=False):
    """
    回傳我們自製的 NCKU Wrapper
    """
    return NCKUCustomLLM(
        model_name=config.LLM_MODEL,
        temperature=temperature
    )

def get_embeddings():
    """回傳本地 Embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )