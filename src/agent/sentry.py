from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.llm_factory import get_llm # 記得要用本地小模型
from src.config import config
import ollama

class Sentry:
    def __init__(self):
        # 使用本地小模型 (Fast System 1)
        # 這裡我們直接用 ollama client 或 langchain wrapper，為了方便我們用 langchain
        self.llm = get_llm(temperature=0, json_mode=True) 
        # 注意：要在 get_llm 支援切換 model，或直接在這裡用 ChatOllama 指定 config.FAST_LLM_MODEL

    async def check_urgency(self, observations: list[str]) -> bool:
        """
        判斷這些觀察是否包含緊急事件
        """
        obs_text = "\n".join(observations)
        
        prompt = ChatPromptTemplate.from_template("""
        你是一個 AI 代理的「感知過濾器」。
        請評估以下觀察到的環境資訊，判斷是否發生了「需要立即注意或中斷當前動作」的事件。
        
        [觀察內容]
        {obs_text}
        
        [判斷標準]
        - 緊急 (True): 火災、有人向我搭話、有人呼救、巨大的聲響、突發意外。
        - 平凡 (False): 靜態的環境描述、別人在做不相關的事(睡覺、讀書)、物品狀態正常改變。
        
        請輸出 JSON: {{ "is_urgent": true/false, "reason": "簡短原因" }}
        """)
        
        # 為了速度，這裡其實可以用更簡單的關鍵字過濾 + LLM 輔助
        # 但我們先用 LLM 展示泛用性
        try:
            # 這裡為了演示用 langchain，實務上建議換成 src.config 裡的 FAST_LLM_HOST
            # 假設 get_llm 支援傳入 model_name，或者我們這裡手動建立一個 fast chain
            from langchain_ollama import ChatOllama
            fast_llm = ChatOllama(base_url=config.FAST_LLM_HOST, model=config.FAST_LLM_MODEL, format="json", temperature=0)
            
            chain = prompt | fast_llm | JsonOutputParser()
            result = await chain.ainvoke({"obs_text": obs_text})
            
            if result.get("is_urgent"):
                print(f"   ⚡ [哨兵] 觸發打斷！原因: {result.get('reason')}")
                return True
            return False
            
        except Exception as e:
            print(f"   ⚠️ 哨兵失效: {e}")
            return False