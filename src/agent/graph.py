import json
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.agent.state import AgentState
from src.memory.retriever import GenerativeRetriever
from src.agent.planning import Planner
from src.agent.reflection import Reflector
from src.llm_factory import get_llm

class GenerativeAgent:
    def __init__(self, name: str, summary: str, collection_name: str):
        self.name = name
        self.summary = summary
        
        # 初始化各模組
        self.retriever = GenerativeRetriever(collection_name=collection_name)
        self.planner = Planner(self.retriever)
        self.reflector = Reflector(self.retriever)
        self.llm = get_llm(temperature=0.4, json_mode=True) # 用於決策
        
        # 編譯 Graph
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 定義節點
        workflow.add_node("perceive", self.perceive_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("react", self.react_node)
        
        # 定義邊 (流程)
        workflow.set_entry_point("perceive")
        workflow.add_edge("perceive", "retrieve")
        workflow.add_edge("retrieve", "react")
        workflow.add_edge("react", END)

        return workflow.compile()

    # --- Nodes 實作 ---

    def perceive_node(self, state: AgentState):
        """
        1. 接收觀察
        2. 存入記憶庫
        3. (選擇性) 檢查是否需要產生今天的計畫
        """
        print(f"\n👀 {state['agent_name']} perceives the world...")
        
        # 1. 儲存觀察
        for obs in state["observations"]:
            self.retriever.add_memory(obs)
            
        # 2. 檢查是否有計畫 (簡化版：如果 state 裡沒計畫且是早上，就生成一個)
        # 實務上這可以做更複雜，例如每天 6:00 AM 自動觸發
        current_plan = state.get("daily_plan", [])
        if not current_plan:
            print("   📅 No plan found. Generating dynamic schedule...")
            # 呼叫 Phase 3 的 Planner
            plan_items = self.planner.create_initial_plan(
                state["agent_name"], state["agent_summary"], state["current_time"]
            )
            # 轉成 dict 存入 state
            current_plan = [item.dict() for item in plan_items]
        
        return {"daily_plan": current_plan}

    def retrieve_node(self, state: AgentState):
        """
        根據最近的觀察，檢索相關記憶來決定如何反應
        """
        print(f"   🧠 Retrieving context...")
        
        # 查詢組裝：結合觀察 + 當前正在做的事(計畫)
        observations_str = ", ".join(state["observations"])
        query = f"Context: {observations_str}. What should {state['agent_name']} do next?"
        
        memories = self.retriever.retrieve(query, k=5)
        return {"relevant_memories": memories}

    def react_node(self, state: AgentState):
        """
        核心決策：根據 (計畫 + 記憶 + 觀察) 決定當下動作
        """
        print(f"   🤔 Deciding action...")
        
        # 準備 Prompt Context
        memories_text = "\n".join([f"- {m.page_content}" for m in state["relevant_memories"]])
        plan_text = json.dumps(state["daily_plan"][:3], indent=2) # 只看接下來的幾個行程
        
        prompt = ChatPromptTemplate.from_template("""
        You are {agent_name}.
        Summary: {agent_summary}
        Current Time: {current_time}
        
        [Relevant Memories]
        {memories}
        
        [Your Original Plan]
        {plan}
        
        [Current Observations]
        {observations}
        
        Based on the observations, should you stick to your plan or react to the new situation?
        Output a JSON with:
        - "action": What you are doing now (1 sentence).
        - "emoji": A fitting emoji.
        - "reason": Why you chose this action.
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({
                "agent_name": state["agent_name"],
                "agent_summary": state["agent_summary"],
                "current_time": state["current_time"],
                "memories": memories_text,
                "plan": plan_text,
                "observations": state["observations"]
            })
            
            print(f"   🎬 ACTION: {result['emoji']} {result['action']}")
            print(f"      (Reason: {result['reason']})")
            
            # 將動作存回記憶 (這樣他才知道自己做過這件事)
            self.retriever.add_memory(
                f"{state['agent_name']} is {result['action']}", 
                type="observation"
            )
            
            return {
                "current_action": result['action'],
                "current_emoji": result['emoji']
            }
            
        except Exception as e:
            print(f"❌ React failed: {e}")
            return {"current_action": "Idling", "current_emoji": "😴"}