import json
import asyncio
import re 
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

# 輔助方法 (請放在 class 內)
    def _get_current_block(self, daily_plan: list, current_time_str: str):
        """找出當下應該執行的 Daily Plan Block (包含結束時間計算)"""
        try:
            curr_dt = datetime.strptime(current_time_str, "%Y-%m-%d %I:%M %p")
            today_str = curr_dt.strftime("%Y-%m-%d")
            active_block = None
            
            for i, block in enumerate(daily_plan):
                t_str = block['start_time'].replace("：", ":")
                try:
                    block_dt = datetime.strptime(f"{today_str} {t_str}", "%Y-%m-%d %H:%M")
                except: continue
                
                if block_dt <= curr_dt:
                    active_block = block
                    # 計算結束時間：拿「下一個 block」的開始時間當作結束
                    if i + 1 < len(daily_plan):
                        next_t = daily_plan[i+1]['start_time'].replace("：", ":")
                        active_block["calculated_end_time"] = next_t
                    else:
                        active_block["calculated_end_time"] = (block_dt + timedelta(hours=2)).strftime("%H:%M")
                else:
                    # 遇到未來的任務就停止，因為 active_block 已鎖定最新的過去任務
                    break
            return active_block
        except: return None

    # Perceive Node 核心
    async def perceive_node(self, state: AgentState):
        print(f"\n👀 {state['agent_name']} 正在感知世界...")
        
        # 1. 儲存觀察
        for obs in state["observations"]:
            await self.retriever.add_memory(obs)

        # 2. 檢查是否忙碌 (Persistence Check)
        # 目前使用簡單字串規則判斷是否為例行公事 (is_routine)
        busy_until = state.get("busy_until")
        if busy_until:
            try:
                curr_dt = datetime.strptime(state["current_time"], "%Y-%m-%d %I:%M %p")
                busy_dt = datetime.strptime(busy_until, "%Y-%m-%d %I:%M %p")
                
                if curr_dt < busy_dt:
                    # 簡單判定：如果觀察只有基本環境描述，就不打斷
                    # (這裡是你提到的"脆弱"判斷，未來可用 Sentry 替換)
                    is_routine = all("你現在位於" in o or "這裡有一個" in o or "You are" in o or "There is" in o for o in state["observations"])
                    
                    if is_routine:
                        print(f"   ⏳ {state['agent_name']} 正在忙於上一個動作 (直到 {busy_until})，跳過思考。")
                        return {"skip_thinking": True}
                    else:
                        print(f"   ⚡ 偵測到新事件！中斷目前的動作！")
                        # 不 return skip，繼續往下走 (重置 busy_until)
            except ValueError:
                pass # 時間解析失敗則忽略忙碌狀態

        # 3. 準備狀態變數
        daily = state.get("daily_plan", [])
        short = state.get("short_term_plan", [])
        last_activity = state.get("current_daily_block_activity")

        # 4. 處理 L1 粗略計畫 (Daily Plan)
        if not daily:
            print("   📅 沒找到計畫。正在生成動態行程...")
            plan_items = await self.planner.create_initial_plan(
                state["agent_name"], state["agent_summary"], state["current_time"]
            )
            daily = [item.dict() for item in plan_items]

        # 5. 處理 L2 細分分解 (Decomposition) & 任務切換
        curr_block = self._get_current_block(daily, state["current_time"])
        current_activity_name = None

        if curr_block:
            current_activity_name = curr_block['activity']
            
            # [關鍵修正] 偵測任務是否切換
            if current_activity_name != last_activity:
                print(f"   🔄 任務切換偵測: '{last_activity}' -> '{current_activity_name}'")
                print(f"   🗑️ 清空過期的短期計畫，準備重新細分...")
                short = [] # 強制清空，觸發下方的分解邏輯

        # 如果沒有短期計畫 (或剛被清空)，進行分解
        if curr_block and not short:
            print(f"   🔍 鎖定任務: {current_activity_name}")
            subtasks = await self.planner.decompose_activity(
                state["agent_name"],
                current_activity_name,
                curr_block['start_time'],
                curr_block.get("calculated_end_time", "Unknown")
            )
            if subtasks:
                short = [t.dict() for t in subtasks]

        return {
            "daily_plan": daily,
            "short_term_plan": short,
            "busy_until": None, 
            "skip_thinking": False,
            "current_daily_block_activity": current_activity_name # 更新當前任務
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
        print(f"   🤔 決定行動...")
        
        # 1. 準備 Context
        memories_text = "\n".join([f"- {m.page_content}" for m in state["relevant_memories"]])
        world_desc = state.get("world_map_desc", "")
        
        short = state.get("short_term_plan", [])
        daily = state.get("daily_plan", [])
        
        # [修改] 將 Planner 指定的「建議地點」加入 Context
        if short:
            current_focus = short[0]
            suggested_loc = current_focus.get('location', '未指定') # 取得地點
            plan_ctx = (
                f"[當前執行細項]\n"
                f"時間: {current_focus['start_time']} - {current_focus['end_time']}\n"
                f"任務: {current_focus['description']}\n"
                f"建議地點: {suggested_loc}" # 明確告訴 LLM 該去哪
            )
        elif daily:
            plan_ctx = f"[當前大方向]\n{json.dumps(daily[:1], indent=2, ensure_ascii=False)}"
        else:
            plan_ctx = "目前沒有具體計畫。"

        # 2. 構建 Prompt
        prompt = ChatPromptTemplate.from_template("""
        你是 {agent_name}。背景: {agent_summary}。時間: {current_time}。
        
        [地圖資訊]
        {world_desc}
        
        [當前計畫]
        {plan_ctx}
        
        [相關記憶]
        {memories}
        
        [目前的觀察]
        {observations}
        
        請決定你現在的行動。
        
        **導航與行動規則 (請嚴格遵守)**:
        1. **優先檢查地點**：看一眼 [當前計畫] 的「建議地點」。如果你現在不在那個地點，請優先設定 `target_location_id` 移動過去。
        2. **到達後操作**：如果你已經在正確地點，則尋找該地點的物品進行操作 (設定 `target_object_id`)。
        3. **填寫 JSON**:
           - 移動時: `target_location_id` 填 ID (如 'bedroom'), `target_object_id` 填 null。
           - 操作時: `target_location_id` 填 null, `target_object_id` 填 ID (如 'bed')。
        
        請輸出 JSON (不要包含 Markdown):
        {{
            "action": "繁體中文描述行動 (1句話)",
            "emoji": "表情",
            "reason": "原因",
            "target_location_id": "ID or null", 
            "target_object_id": "ID or null",
            "duration": 整數 (分鐘),
            "should_replan": true 或 false
        }}
        """)
        
        # 3. 執行 LLM (包含重試機制)
        chain = prompt | self.llm 
        
        import re # 確保有引入 re
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Invoke
                raw_response = chain.invoke({
                    "agent_name": state["agent_name"], "agent_summary": state["agent_summary"],
                    "current_time": state["current_time"], "memories": memories_text,
                    "plan_ctx": plan_ctx, "observations": state["observations"], "world_desc": world_desc
                })
                
                # Clean JSON
                content = raw_response.content
                if "```" in content:
                    content = re.sub(r"```json\s*", "", content)
                    content = re.sub(r"```", "", content)
                content = content.strip()

                # Parse
                parser = JsonOutputParser()
                res = parser.parse(content)
                
                # --- 邏輯處理 ---
                
                # 計算時間
                dur = res.get("duration", 15)
                if dur < 15: dur = 15
                
                time_fmt = "%Y-%m-%d %I:%M %p"
                curr_dt = datetime.strptime(state["current_time"], time_fmt)
                action_end_dt = curr_dt + timedelta(minutes=dur)
                busy_until = action_end_dt.strftime(time_fmt)
                
                print(f"   🎬 {res.get('emoji', '🤖')} {res['action']} ({dur}min)")
                
                # 存記憶
                await self.retriever.add_memory(f"{state['agent_name']} {res['action']}", type="observation")
                
                # A. 處理重規劃
                final_daily_plan = daily
                if res.get("should_replan"):
                    print(f"   ⚠️ 偵測到重規劃需求...")
                    new_schedule = await self.planner.update_plan(
                        state["agent_name"], daily, state["current_time"], res['action']
                    )
                    if new_schedule:
                        final_daily_plan = [item.dict() for item in new_schedule]
                        short = [] 
                
                # B. 處理任務推進 (比對時間)
                elif short:
                    current_subtask = short[0]
                    try:
                        task_end_str = current_subtask['end_time'].replace("：", ":")
                        today_str = curr_dt.strftime("%Y-%m-%d")
                        # 這裡假設 end_time 格式正確，若有跨日需額外處理，目前簡化
                        task_end_dt = datetime.strptime(f"{today_str} {task_end_str}", "%Y-%m-%d %H:%M")
                        
                        # 如果動作結束時間 >= 任務結束時間，視為完成
                        if action_end_dt >= task_end_dt:
                            removed = short.pop(0)
                            print(f"   ✅ 完成細項: {removed['description']} (地點: {removed.get('location', '未指定')})")
                            if short: print(f"   🔜 下一項: {short[0]['description']} @ {short[0].get('location')}")
                        else:
                            print(f"   ▶️ 任務進行中: {current_subtask['description']}")
                    except ValueError:
                        pass
                
                return {
                    "current_action": res['action'], 
                    "current_emoji": res.get("emoji", "🤖"),
                    "target_location_id": res.get("target_location_id"),
                    "target_object_id": res.get("target_object_id"),
                    "busy_until": busy_until,
                    "daily_plan": final_daily_plan,
                    "short_term_plan": short
                }

            except Exception as e:
                print(f"   ⚠️ JSON 解析或執行失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"   ❌ 放棄思考，執行發呆。")
                    return {"current_action": "發呆", "busy_until": None}
    
    def interview(self, question: str):
        # 簡單的同步接口，實際應使用 async
        pass