import json
import asyncio
from datetime import datetime, timedelta
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
        
        # 決策用模型 (通常是慢思考/大模型)
        self.llm = get_llm(temperature=0.4, json_mode=True)
        
        # 編譯 Graph
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 定義 node
        workflow.add_node("perceive", self.perceive_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("react", self.react_node)
        
        # 定義 edge
        workflow.set_entry_point("perceive")
        # 是否跳過思考
        def should_retrieve(state):
            if state.get("skip_thinking", False): # perceive return
                return END # 如果還在忙，直接結束，不進行檢索與反應
            return "retrieve"
        workflow.add_conditional_edges(
            "perceive",
            should_retrieve
        )
        workflow.add_edge("retrieve", "react")
        workflow.add_edge("react", END)

        return workflow.compile()

    async def perceive_node(self, state: AgentState):
        """
        感知節點：
        1. 儲存觀察。
        2. 檢查是否還在執行 plan
        3. 沒有的話, "填入"下一個 plan
        """
        print(f"\n👀 {state['agent_name']} 正在感知世界...")
        
        # 1. 儲存觀察到記憶庫 (world 在 main 抽取後放入)
        for obs in state["observations"]:
            await self.retriever.add_memory(obs)

        # 2. 檢查是否還在做事 (要做到 busy_until 結束)
        busy_until = state.get("busy_until")
        current_time_str = state["current_time"]
        
        if busy_until:
            try:
                # 解析時間 (必須與 main.py 格式一致)
                time_fmt = "%Y-%m-%d %I:%M %p"
                curr_dt = datetime.strptime(current_time_str, time_fmt)
                busy_dt = datetime.strptime(busy_until, time_fmt)
                
                # 如果現在時間 < 忙碌結束時間
                if curr_dt < busy_dt:
                    # 檢查是否有「重大事件」打斷
                    # 簡單判定：如果觀察只有"環境描述" ("你現在位於...", "這裡有一個...")，就不打斷
                    # 如果有其他訊息 (e.g. "Fire!", "Maria is talking to you")，視為打斷
                    is_routine = all("你現在位於" in o or "這裡有一個" in o or "You are" in o or "There is" in o for o in state["observations"])
                    
                    if is_routine:
                        print(f"   ⏳ {state['agent_name']} 正在忙於上一個動作 (直到 {busy_until})，跳過思考。")
                        return {"skip_thinking": True} # 給 conditional edge
                    else:
                        print(f"   ⚡ 偵測到新事件！中斷目前的動作！")
                        # 清空 busy_until，強制重新思考
                        # 注意：這裡不 return skip，而是繼續往下走
            except ValueError as e:
                print(f"   ⚠️ 時間格式解析錯誤: {e}，強制重新思考。")
        
        # --- 如果決定要思考，繼續執行下一個 plan(先看有沒有 plan) ---

        current_daily_plan = state.get("daily_plan", [])
        short_term_plan = state.get("short_term_plan", [])
        
        # 3. 處理 粗略計畫 (Daily Plan)
        if not current_daily_plan:
            print("   📅 沒找到計畫。正在生成動態行程...")
            plan_items = await self.planner.create_initial_plan(
                state["agent_name"], state["agent_summary"], state["current_time"]
            )
            current_daily_plan = [item.dict() for item in plan_items]

        # 4. 處理 細分分解 (Decomposition)
        if current_daily_plan and not short_term_plan:
            current_block = current_daily_plan[0]
            print(f"   🔍 嘗試細分活動: {current_block['activity']}")
            
            subtasks = await self.planner.decompose_activity(
                state["agent_name"],
                current_block['activity'],
                current_block['start_time'],
                "Unknown End" 
            )
            if subtasks:
                short_term_plan = [t.dict() for t in subtasks]

        # 清除 busy_until (因為要重新裝入下一個 plan)
        return {
            "daily_plan": current_daily_plan,
            "short_term_plan": short_term_plan,
            "busy_until": None, 
            "skip_thinking": False
        }

    async def retrieve_node(self, state: AgentState):
        """
        檢索節點
        1. observations => retrieve
        """
        print(f"   🧠 正在檢索相關記憶...")
        
        observations_str = ", ".join(state["observations"])
        query = f"情境: {observations_str}. {state['agent_name']} 接下來該做什麼?"
        
        memories = await self.retriever.retrieve(query, k=5)
        return {"relevant_memories": memories}

    async def react_node(self, state: AgentState):
        """
        反應節點：決定行動與持續時間
        1. get 上一步的 memory
        2. 檢查有沒有填入 plan
        3. prompt -> LLM -> return action
        """
        print(f"   🤔 正在決定行動...")
        
        memories_text = "\n".join([f"- {m.page_content}" for m in state["relevant_memories"]])
        
        short_term = state.get("short_term_plan", [])
        daily = state.get("daily_plan", [])
        
        if short_term:
            current_focus = short_term[0]
            plan_context = f"[當前執行細項]\n時間: {current_focus['start_time']} - {current_focus['end_time']}\n任務: {current_focus['description']}"
        elif daily:
            plan_context = f"[當前大方向]\n{json.dumps(daily[:1], indent=2, ensure_ascii=False)}"
        else:
            plan_context = "目前沒有具體計畫。"

        prompt = ChatPromptTemplate.from_template("""
        你是 {agent_name}。
        背景: {agent_summary}
        目前時間: {current_time}
        
        [計畫狀態]
        {plan_context}
        
        [相關記憶]
        {memories}
        
        [目前的觀察]
        {observations}
        
        請決定你現在的行動。
        同時估計這個行動大約需要多久 (分鐘)，以及是否需要重規劃。
        
        請輸出 JSON (不要包含 Markdown):
        {{
            "action": "繁體中文描述行動 (1句話)",
            "emoji": "表情符號",
            "reason": "原因",
            "duration": 整數 (分鐘, 例如: 15, 30, 60),
            "should_replan": true 或 false
        }}
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({
                "agent_name": state["agent_name"],
                "agent_summary": state["agent_summary"],
                "current_time": state["current_time"],
                "memories": memories_text,
                "plan_context": plan_context,
                "observations": state["observations"]
            })
            
            # --- 計算 busy_until ---
            duration = result.get("duration", 15)
            # 確保 duration 至少 15 分鐘
            if duration < 15: duration = 15
            
            time_fmt = "%Y-%m-%d %I:%M %p"
            curr_dt = datetime.strptime(state["current_time"], time_fmt)
            end_dt = curr_dt + timedelta(minutes=duration)
            busy_until_str = end_dt.strftime(time_fmt)
            
            print(f"   🎬 行動: {result['emoji']} {result['action']}")
            print(f"      (預計耗時: {duration} 分鐘, 直到 {busy_until_str})")
            
            # observation 會影響 plan -> LLM think should replan (呼叫 planner update)
            final_daily_plan = daily
            if result.get("should_replan", False):
                print(f"   ⚠️ 偵測到計畫變更需求，正在重規劃...")
                new_schedule = await self.planner.update_plan(
                    agent_name=state["agent_name"],
                    current_plan=daily,
                    current_time=state["current_time"],
                    reason=result['action']
                )
                if new_schedule:
                    final_daily_plan = [item.dict() for item in new_schedule]
                    short_term = []

            # --- 邏輯 B: 推進短期計畫 ---
            # 假設完成此動作後，就移除第一個細項
            if short_term and not result.get("should_replan", False):
                # 這裡簡單移除，實際應用可比對時間
                # short_term.pop(0) 
                pass

            # 存入記憶
            await self.retriever.add_memory(
                f"{state['agent_name']} 正在 {result['action']}", 
                type="observation"
            )
            
            return {
                "current_action": result['action'],
                "current_emoji": result['emoji'],
                "daily_plan": final_daily_plan,
                "short_term_plan": short_term,
                "busy_until": busy_until_str # 更新忙碌狀態
            }
            
        except Exception as e:
            print(f"❌ 決策失敗: {e}")
            return {
                "current_action": "發呆", 
                "current_emoji": "😳", 
                "busy_until": None
            }

    def interview(self, question: str):
        # 簡單的同步接口，實際應使用 async
        pass