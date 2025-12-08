# src/memory/importance.py

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.config import config

class ImportanceScore(BaseModel):
    score: int = Field(description="分數介於 1 到 10 之間")

def get_importance_scorer():
    llm = ChatOllama(
        base_url=config.FAST_LLM_HOST,
        model=config.FAST_LLM_MODEL,
        temperature=0,
        format="json"
    ) # LangChain 提供的 LLM 介面，用來跟 Ollama server 溝通
    
    # 把 LLM response json 格式轉成 pydantic 格式
    parser = PydanticOutputParser(pydantic_object=ImportanceScore)
    
    template = """
    請評估這段記憶的重要性，範圍從 1 (瑣碎日常，如刷牙) 到 10 (極度重要，如分手)。
    
    記憶內容: {memory_content}
    
    請只回傳 JSON 格式: {{ "score": int }}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | parser | (lambda x: x.score) # Pydantic Model 中取出 score (int)
    # output score (int)
    return chain