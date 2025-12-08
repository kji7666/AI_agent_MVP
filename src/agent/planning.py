from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.llm_factory import get_llm
from src.memory.retriever import GenerativeRetriever

class PlanItem(BaseModel):
    start_time: str = Field(description="Time in HH:MM format (e.g., 08:00)")
    activity: str = Field(description="Description of the activity")
    location: str = Field(description="Where this activity takes place")

class DailyPlan(BaseModel):
    schedule: List[PlanItem] = Field(description="The full day schedule")

class SubTask(BaseModel):
    start_time: str = Field(description="HH:MM")
    end_time: str = Field(description="HH:MM")
    description: str = Field(description="具體的細項動作")

class DetailedRoutine(BaseModel):
    subtasks: List[SubTask]

class Planner:
    def __init__(self, retriever: GenerativeRetriever):
        self.retriever = retriever
        self.llm = get_llm(temperature=0.4, json_mode=True) 

    # ==========================================
    # Step 1: 獲取昨日脈絡 (Temporal Context)
    # ==========================================
    async def _get_yesterday_context(self, agent_name: str) -> str:
        """檢索昨天發生了什麼，以決定今天的延續性"""
        # 這裡用模糊查詢，依賴語意搜尋找到相關的時間點
        query = f"{agent_name} 昨天做了什麼？有哪些未完成的事？"
        memories = await self.retriever.retrieve(query, k=3)
        if not memories:
            return "沒有關於昨天的特別紀錄。"
        return "\n".join([f"- {m.page_content}" for m in memories])

    # ==========================================
    # Step 2: 獲取內在狀態 (Reflection Context)
    # ==========================================
    async def _get_internal_state(self, agent_name: str) -> str:
        """檢索最近的反思與心情"""
        query = f"{agent_name} 最近的心情、感覺與反思洞察"
        # 這裡我們希望抓到 'reflection' 類型的記憶
        memories = await self.retriever.retrieve(query, k=3)
        if not memories:
            return "心情平靜，沒有特別的想法。"
        return "\n".join([f"- {m.page_content}" for m in memories])

    # ==========================================
    # Step 3: 獲取目標進度 (Goal Context)
    # ==========================================
    async def _get_goal_context(self, agent_name: str, agent_summary: str) -> str:
        """先從 Summary 提取核心目標，再檢索該目標的進度"""
        
        # 3.1 先問 LLM 核心目標是什麼 (簡單提取)
        extract_prompt = ChatPromptTemplate.from_template("""
        根據以下描述，{agent_name} 目前人生中最重要的 1 個長期目標是什麼？
        (例如：寫完論文、準備馬拉松、交到女朋友)
        請用 JSON 回傳: {{ "goal": "目標描述" }}
        
        描述: {summary}
        """)
        try:
            chain = extract_prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"agent_name": agent_name, "summary": agent_summary})
            core_goal = result.get("goal", "過好每一天")
        except:
            core_goal = "日常雜務"

        # 3.2 檢索該目標的狀態
        query = f"{agent_name} 的 '{core_goal}' 目前進度與相關活動"
        memories = await self.retriever.retrieve(query, k=3)
        
        context_str = f"核心目標: {core_goal}\n相關記憶:\n"
        if memories:
            context_str += "\n".join([f"- {m.page_content}" for m in memories])
        else:
            context_str += "目前還沒有開始執行此目標。"
            
        return context_str

    # ==========================================
    # 主流程: 綜合生成計畫
    # ==========================================
    async def create_initial_plan(self, agent_name: str, agent_summary: str, current_time: str):
        print(f"📅 {agent_name} 正在進行深度規劃 (Context-Aware)...")
        
        # 平行執行三個檢索任務
        # 同時發出三個查詢，不用一個等一個
        import asyncio
        yesterday_ctx, state_ctx, goal_ctx = await asyncio.gather(
            self._get_yesterday_context(agent_name),
            self._get_internal_state(agent_name),
            self._get_goal_context(agent_name, agent_summary)
        )
        
        print(f"   🔍 [昨日] 檢索完成")
        print(f"   🔍 [狀態] 檢索完成")
        print(f"   🔍 [目標] 檢索完成")
        # 把 LLM response json 格式轉成 pydantic 格式
        parser = PydanticOutputParser(pydantic_object=DailyPlan)

        template = """
        你是 {agent_name}。
        背景設定: {agent_summary}
        目前時間: {current_time}
        
        為了制定今天的計畫，請參考以下資訊：
        
        === 1. 昨日回顧 (Yesterday) ===
        {yesterday_ctx}
        (如果昨天有未完成的事，今天請優先安排)
        
        === 2. 內在狀態 (Internal State) ===
        {state_ctx}
        (如果最近很累，請安排休息；如果很有動力，請安排困難工作)
        
        === 3. 目標進度 (Core Goal) ===
        {goal_ctx}
        (請確保今天的行程能推進這個目標)
        
        --- 任務 ---
        請綜合以上資訊，為今天制定一個具體且連貫的行程表。
        行程應該涵蓋從起床到睡覺的時間 (5-8 個主要時段)。
        請使用繁體中文回答。
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | parser
        
        try:
            plan = chain.invoke({
                "agent_name": agent_name,
                "agent_summary": agent_summary,
                "current_time": current_time,
                "yesterday_ctx": yesterday_ctx,
                "state_ctx": state_ctx,
                "goal_ctx": goal_ctx,
                "format_instructions": parser.get_format_instructions()
            })
            
            # 合併 Str and 存入記憶
            plan_text = f"{current_time} 的每日計畫 (基於昨日與目標):\n"
            for item in plan.schedule:
                line = f"{item.start_time}: {item.activity} (地點: {item.location})"
                plan_text += line + "\n"
                print(f"   📌 {line}")
            
            await self.retriever.add_memory(content=plan_text, type="plan")
            return plan.schedule
            
        except Exception as e:
            print(f"❌ 計畫生成失敗: {e}")
            return []
        
    async def update_plan(self, agent_name: str, current_plan: List[dict], current_time: str, reason: str):
        """
        重規劃功能
        當代理人偏離原訂計畫時，呼叫此方法來修正剩餘的行程表。
        """
        print(f"🔄 {agent_name} 正在修正行程表 (原因: {reason})...")
        
        parser = PydanticOutputParser(pydantic_object=DailyPlan)

        # 將舊計畫轉成字串方便 LLM 閱讀
        old_plan_str = "\n".join([f"{p['start_time']}: {p['activity']}" for p in current_plan])

        template = """
        你是 {agent_name}。
        目前時間: {current_time}。
        
        [原本的計畫]
        {old_plan_str}
        
        [發生的狀況]
        你剛剛偏離了計畫，原因: {reason}。
        
        請根據目前時間和狀況，**重新安排今天剩餘的行程**。
        1. 移除已經過去的時間段。
        2. 根據新的狀況調整接下來的活動（例如：如果遲到了，可能要取消某些事，或是順延）。
        3. 保持行程的連貫性。
        
        請使用繁體中文回答。
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | parser
        
        try:
            new_plan = chain.invoke({
                "agent_name": agent_name,
                "current_time": current_time,
                "old_plan_str": old_plan_str,
                "reason": reason,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Log 並存入記憶
            plan_text = f"{current_time} 的修正計畫 (因 {reason}):\n"
            for item in new_plan.schedule:
                line = f"{item.start_time}: {item.activity} (地點: {item.location})"
                plan_text += line + "\n"
                print(f"   🔄 [修正] {line}")
            
            await self.retriever.add_memory(content=plan_text, type="plan")
            
            return new_plan.schedule
            
        except Exception as e:
            print(f"❌ 重規劃失敗: {e}")
            # 如果失敗，回傳原本的計畫避免崩潰
            return []
        
    async def decompose_activity(self, agent_name: str, activity: str, start_time: str, end_time: str):
        """
        遞迴分解：將一個長時間的粗略活動，細分為短時間的具體執行步驟。
        """
        print(f"🔨 {agent_name} 正在細分活動: '{activity}' ({start_time} - {end_time})...")
        
        parser = PydanticOutputParser(pydantic_object=DetailedRoutine)

        template = """
        你是 {agent_name}。
        你原本的計畫是在 {start_time} 到 {end_time} 進行 "{activity}"。
        
        請將這個時段細分為更具體、可執行的子任務 (Sub-tasks)。
        每個子任務大約 15-60 分鐘。
        確保子任務加總起來的時間涵蓋整個時段。
        
        請使用繁體中文回答。
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "agent_name": agent_name,
                "activity": activity,
                "start_time": start_time,
                "end_time": end_time,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Log
            for task in result.subtasks:
                print(f"   ↳ 🔨 {task.start_time}-{task.end_time}: {task.description}")
            
            # 存入記憶 (讓 Agent 記得自己規劃了細節)
            detail_text = f"針對 {start_time} 的 '{activity}'，我規劃了細節:\n" + \
                          "\n".join([f"- {t.start_time}: {t.description}" for t in result.subtasks])
            await self.retriever.add_memory(content=detail_text, type="plan")

            return result.subtasks
            
        except Exception as e:
            print(f"❌ 細分失敗: {e}")
            return []