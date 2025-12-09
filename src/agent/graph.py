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

    def _get_current_block(self, daily_plan: list, current_time_str: str):
        """
        [修正版] 找出當下應該執行的 Daily Plan Block
        """
        time_fmt = "%Y-%m-%d %I:%M %p"
        try:
            curr_dt = datetime.strptime(current_time_str, time_fmt)
            today_str = curr_dt.strftime("%Y-%m-%d")
            
            active_block = None
            
            # 我們需要找到一個 block，它的 start_time <= current_time
            # 且它是所有符合條件中「最晚開始」的一個 (也就是最新的)
            
            for i, block in enumerate(daily_plan):
                try:
                    # 處理時間格式 (容錯中文全形冒號)
                    t_str = block['start_time'].replace("：", ":")
                    
                    # 補上日期進行比對
                    block_dt = datetime.strptime(f"{today_str} {t_str}", "%Y-%m-%d %H:%M")
                    
                    if block_dt <= curr_dt:
                        # 找到了候選人
                        active_block = block
                        
                        # 順便計算結束時間 (拿「下一個 block」的開始時間當作結束)
                        if i + 1 < len(daily_plan):
                            next_t_str = daily_plan[i+1]['start_time'].replace("：", ":")
                            active_block["calculated_end_time"] = next_t_str
                        else:
                            # 如果是最後一個任務，假設 2 小時後結束
                            end_dt = block_dt + timedelta(hours=2)
                            active_block["calculated_end_time"] = end_dt.strftime("%H:%M")
                            
                    else:
                        # 因為 daily_plan 是照時間排序的
                        # 一旦遇到一個 "未來" 的任務，就可以停止搜尋了
                        # 此時 active_block 裡面存的就是「當下正在進行」的任務
                        break
                        
                except ValueError:
                    continue
            
            return active_block

        except Exception as e:
            print(f"   ⚠️ 時間解析錯誤: {e}")
            return None
        
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
        
        #  取得上一次紀錄正在做的大任務，用於比對
        last_activity = state.get("current_daily_block_activity")

        # 3. 處理 粗略計畫 (Daily Plan)
        if not current_daily_plan:
            print("   📅 沒找到計畫。正在生成動態行程...")
            plan_items = await self.planner.create_initial_plan(
                state["agent_name"], state["agent_summary"], state["current_time"]
            )
            current_daily_plan = [item.dict() for item in plan_items]

        # 4. [修正] 處理 細分分解 (Decomposition)
        # 先找出現在時間對應的大任務
        current_block = self._get_current_block(current_daily_plan, current_time_str)
        
        current_activity_name = None
        if current_block:
            current_activity_name = current_block['activity']
            
            # [邏輯修正] 關鍵判斷：任務是否切換了？
            # 如果 (有新任務) 且 (新任務 != 舊任務)
            if current_activity_name != last_activity:
                print(f"   🔄 任務切換偵測: '{last_activity}' -> '{current_activity_name}'")
                print(f"   🗑️ 清空過期的短期計畫，準備重新細分...")
                short_term_plan = [] # 強制清空舊細節！

        # 如果短期計畫是空的 (包含剛剛被我們強制清空的)，且有當前任務，就進行細分
        if current_block and not short_term_plan:
            print(f"   🔍 鎖定當前時段任務: {current_block['activity']}")
            end_time = current_block.get("calculated_end_time", "Unknown")
            
            subtasks = await self.planner.decompose_activity(
                state["agent_name"],
                current_block['activity'],
                current_block['start_time'],
                end_time # 傳入計算出的結束時間
            )
            if subtasks:
                short_term_plan = [t.dict() for t in subtasks]

        # 清除 busy_until (因為要重新裝入下一個 plan)
        return {
            "daily_plan": current_daily_plan,
            "short_term_plan": short_term_plan,
            "busy_until": None, 
            "skip_thinking": False,
            # 更新當前任務名稱到 State，供下一輪比對
            "current_daily_block_activity": current_activity_name 
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
        memories_text = "\n".join([f"- {m.page_content}" for m in state["relevant_memories"]])
        
        # 取得短期計畫與每日計畫
        short = state.get("short_term_plan", [])
        daily = state.get("daily_plan", [])
        
        plan_ctx = f"當前細項: {short[0]['description']}" if short else "無具體細項"
        world_desc = state.get("world_map_desc", "")

        prompt = ChatPromptTemplate.from_template("""
        你是 {agent_name}。背景: {agent_summary}。時間: {current_time}。
        
        [地圖]
        {world_desc}
        [計畫]
        {plan_ctx}
        [記憶]
        {memories}
        [觀察]
        {observations}
        
        請決定你現在的行動。
        
        **JSON 填寫範例 (請嚴格參考)**:
        - 情況 A (移動): {{ "action": "前往廚房準備早餐", "target_location_id": "kitchen", "target_object_id": null, ... }}
        - 情況 B (操作物品): {{ "action": "使用咖啡機", "target_location_id": null, "target_object_id": "coffee_machine", ... }}
        - 情況 C (原地發呆): {{ "action": "發呆", "target_location_id": null, "target_object_id": null, ... }}
        
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
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            res = chain.invoke({
                "agent_name": state["agent_name"], "agent_summary": state["agent_summary"],
                "current_time": state["current_time"], "memories": memories_text,
                "plan_ctx": plan_ctx, "observations": state["observations"], "world_desc": world_desc
            })
            
            # --- 1. 計算時間與狀態 ---
            dur = res.get("duration", 15)
            if dur < 15: dur = 15 # 確保最小耗時
            
            time_fmt = "%Y-%m-%d %I:%M %p"
            curr_dt = datetime.strptime(state["current_time"], time_fmt)
            
            # 計算動作結束的時間點
            action_end_dt = curr_dt + timedelta(minutes=dur)
            busy_until = action_end_dt.strftime(time_fmt)
            
            print(f"   🎬 {res['emoji']} {res['action']} ({dur}min)")
            await self.retriever.add_memory(f"{state['agent_name']} {res['action']}", type="observation")
            
            # --- 2. 處理計畫變更 (重規劃 vs 任務推進) ---
            final_daily_plan = daily # 預設維持原樣
            
            # 情況 A: LLM 決定重規劃
            if res.get("should_replan"):
                print(f"   ⚠️ 偵測到重規劃需求...")
                new_schedule = await self.planner.update_plan(
                    state["agent_name"], daily, state["current_time"], res['action']
                )
                if new_schedule:
                    final_daily_plan = [item.dict() for item in new_schedule]
                    short = [] # 重規劃後，舊的短期細節作廢
            
            # 情況 B: [新增] 推進短期計畫
            # 如果沒有重規劃，且手上有短期任務，檢查是否完成
            elif short:
                current_subtask = short[0]
                try:
                    # 解析任務結束時間 (格式通常是 HH:MM)
                    task_end_str = current_subtask['end_time'].replace("：", ":")
                    today_str = curr_dt.strftime("%Y-%m-%d")
                    task_end_dt = datetime.strptime(f"{today_str} {task_end_str}", "%Y-%m-%d %H:%M")
                    
                    # 判定：如果「動作做完的時間」 >= 「任務表定結束時間」
                    if action_end_dt >= task_end_dt:
                        removed = short.pop(0) # 移除第一項
                        print(f"   ✅ 完成細項: {removed['description']} (進度: {busy_until})")
                        
                        if short:
                            print(f"   🔜 下一項: {short[0]['description']}")
                    else:
                        print(f"   ▶️ 任務進行中: {current_subtask['description']}")
                        
                except ValueError:
                    # 如果時間格式解析失敗，保守起見不移除，讓下一次 perceive_node 決定
                    pass
            
            return {
                "current_action": res['action'], 
                "current_emoji": res['emoji'],
                "target_location_id": res.get("target_location_id"),
                "target_object_id": res.get("target_object_id"),
                "busy_until": busy_until,
                "daily_plan": final_daily_plan, # 回傳可能更新過的每日計畫
                "short_term_plan": short        # 回傳可能更新過的短期計畫
            }
            
        except Exception as e:
            print(f"❌ React Error: {e}")
            return {"current_action": "發呆", "busy_until": None}

    def interview(self, question: str):
        # 簡單的同步接口，實際應使用 async
        pass