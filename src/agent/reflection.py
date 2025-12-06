from typing import List
from langchain_core.prompts import ChatPromptTemplate
from src.llm_factory import get_llm
from src.memory.retriever import GenerativeRetriever

class Reflector:
    def __init__(self, retriever: GenerativeRetriever):
        self.retriever = retriever
        self.llm = get_llm(temperature=0.5) # 反思需要一點創意

    def run(self, agent_name: str, last_k: int = 20):
        """
        執行反思程序：
        1. 撈取最近 k 條尚未反思過的記憶
        2. 請 LLM 歸納
        3. 將歸納結果 (Insight) 寫回記憶庫
        """
        print(f"🤔 {agent_name} is reflecting on recent events...")
        
        # 1. 為了簡化 MVP，我們直接撈取最近的記憶 (不論是否反思過)
        # 在完整版中，我們應該記錄一個 'last_reflected_time' 指標
        recent_memories = self.retriever.retrieve(
            query=f"What happened to {agent_name} recently?",
            k=last_k,
            fetch_k=last_k * 2
        )
        
        if not recent_memories:
            print("   No memories to reflect on.")
            return

        # 將記憶轉為文字清單
        observations = [m.page_content for m in recent_memories]
        observations_str = "\n".join([f"- {o}" for o in observations])

        # 2. 呼叫 LLM 進行歸納
        # 論文技巧：Ask "What high-level insights can you infer?"
        prompt = ChatPromptTemplate.from_template("""
        {observations}
        
        Given only the information above, what are 3 most salient high-level insights 
        we can infer about {agent_name}?
        
        Respond with 3 distinct sentences, one per line. Do not include numbering.
        """)
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "observations": observations_str, 
                "agent_name": agent_name
            })
            insights = response.content.strip().split('\n')
            
            # 3. 將 Insight 存回記憶庫
            for insight in insights:
                insight = insight.strip()
                # 去除可能的編號 (1. 2. - 等)
                if insight and len(insight) > 10: 
                    print(f"   💡 Insight generated: {insight}")
                    # 寫入時標記 type='reflection'
                    self.retriever.add_memory(
                        content=insight,
                        type="reflection"
                    )
                    
        except Exception as e:
            print(f"❌ Reflection failed: {e}")