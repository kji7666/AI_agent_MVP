from typing import List
from langchain_core.prompts import ChatPromptTemplate
from src.llm_factory import get_llm
from src.memory.retriever import GenerativeRetriever

class Reflector:
    def __init__(self, retriever: GenerativeRetriever):
        self.retriever = retriever
        self.llm = get_llm(temperature=0.5)

    async def run(self, agent_name: str, last_k: int = 20):
        print(f"🤔 {agent_name} 正在反思最近發生的事...")
        
        recent_memories = await self.retriever.retrieve(
            query=f"{agent_name} 最近發生了什麼事?",
            k=last_k,
            fetch_k=last_k * 2
        )
        
        if not recent_memories:
            print("   沒有足夠的記憶可供反思。")
            return

        observations = [m.page_content for m in recent_memories]
        observations_str = "\n".join([f"- {o}" for o in observations])

        prompt = ChatPromptTemplate.from_template("""
        {observations}
        
        僅根據以上資訊，我們可以推斷出關於 {agent_name} 的哪 3 個最重要的高層次洞察 (Insights)？
        請用繁體中文回答，列出 3 個不同的句子，每行一句。不要包含編號。
        """)
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "observations": observations_str, 
                "agent_name": agent_name
            })
            insights = response.content.strip().split('\n') # 列出 3 個不同的句子，每行一句 => \n split
            
            for insight in insights:
                insight = insight.strip()
                if insight and len(insight) > 5: 
                    print(f"   💡 生成洞察: {insight}")
                    await self.retriever.add_memory(
                        content=insight,
                        type="reflection"
                    )
                    
        except Exception as e:
            print(f"❌ 反思失敗: {e}")