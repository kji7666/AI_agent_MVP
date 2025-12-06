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
    model_name: str = Field(default=config.LLM_MODEL)
    temperature: float = Field(default=0.7)
    _client: ollama.Client = PrivateAttr() # 不備序列化, 因為有 key

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = ollama.Client(
            host=config.LLM_HOST,
            headers={'Authorization': f'Bearer {config.LLM_API_KEY}'},
            timeout=120
        )

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
            # 錯誤處理：印出詳細資訊方便除錯
            print(f"❌ NCKU API Error: {e}")
            raise e

    @property
    def _llm_type(self) -> str:
        return "ncku-custom-wrapper"

# ==========================================
# 工廠函數
# ==========================================

def get_llm(temperature=0.7, json_mode=False):
    """
    回傳我們自製的 NCKU Wrapper。
    註：gpt-oss:120b 目前對於 JSON mode 的支援可能不一，
    這裡我們先回傳一般模式，靠 Prompt Engineering 強制它輸出 JSON。
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