from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.llm_factory import get_llm
from src.memory.retriever import GenerativeRetriever

# 定義行程表的資料結構
class PlanItem(BaseModel):
    start_time: str = Field(description="Time in HH:MM format (e.g., 08:00)")
    activity: str = Field(description="Description of the activity")
    location: str = Field(description="Where this activity takes place")

class DailyPlan(BaseModel):
    schedule: List[PlanItem] = Field(description="The full day schedule")

class Planner:
    def __init__(self, retriever: GenerativeRetriever):
        self.retriever = retriever
        # Planning 需要非常嚴謹的格式，所以 Temperature 設低一點
        self.llm = get_llm(temperature=0.2, json_mode=True) 

    def create_initial_plan(self, agent_name: str, agent_summary: str, current_time: str):
        """
        產生一天的粗略計畫
        """
        print(f"📅 {agent_name} is creating a daily plan...")
        
        parser = PydanticOutputParser(pydantic_object=DailyPlan)

        template = """
        You are {agent_name}. 
        Here is your background: {agent_summary}
        Current time: {current_time}
        
        Based on your background, create a broad daily schedule for today.
        The schedule should cover from waking up to sleeping.
        Break it down into 5-8 major blocks.
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = prompt | self.llm | parser
        
        try:
            plan = chain.invoke({
                "agent_name": agent_name,
                "agent_summary": agent_summary,
                "current_time": current_time,
                "format_instructions": parser.get_format_instructions()
            })
            
            # 將計畫存入記憶 (這很重要，這樣代理人之後才會記得自己有計畫)
            plan_text = f"Daily Plan for {current_time}:\n"
            for item in plan.schedule:
                line = f"{item.start_time}: {item.activity} at {item.location}"
                plan_text += line + "\n"
                print(f"   📌 {line}")
            
            self.retriever.add_memory(content=plan_text, type="plan")
            
            return plan.schedule
            
        except Exception as e:
            print(f"❌ Planning failed: {e}")
            return []