from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.llm_factory import get_llm

# 定義 LLM 的輸出格式
class ImportanceScore(BaseModel):
    score: int = Field(description="An integer score between 1 and 10")

def get_importance_scorer():
    """
    回傳一個 Chain，輸入 {"memory_content": "..."}，輸出 int 分數
    """
    llm = get_llm(temperature=0.0, json_mode=True) # 評分需要精準，temp=0
    parser = PydanticOutputParser(pydantic_object=ImportanceScore) # 當 JSON 格式錯誤時：自動 raise Exception
    template = """
    On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) 
    and 10 is extremely poignant (e.g., a break up, college acceptance), 
    rate the likely poignancy of the following piece of memory.
    
    Memory: {memory_content}
    
    Respond in JSON format: {{ "score": <int> }}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 建立 Chain: Prompt -> LLM -> JSON Parser -> 取出 score 欄位
    chain = prompt | llm | parser | (lambda x: x.score)
    
    return chain