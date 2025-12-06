import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.graph import GenerativeAgent

def run_simulation():
    print("========================================")
    print("🤖 STARTING AGENT SIMULATION")
    print("========================================")
    
    # 1. 初始化代理人
    # 我們沿用 Klaus，但給一個新的 collection 以免被舊測試干擾
    klaus = GenerativeAgent(
        name="Klaus",
        summary="Klaus is a sociology student who loves reading and is precise with his schedule.",
        collection_name="sim_klaus_01"
    )
    
    # 2. 定義模擬時間軸與事件
    timeline = [
        {
            "time": "08:00 AM",
            "obs": ["Klaus wakes up in his dorm room.", "The sun is shining."]
        },
        {
            "time": "09:00 AM",
            "obs": ["Klaus's stomach is growling loud.", "The fridge is empty."]
        },
        {
            "time": "10:00 AM",
            "obs": ["Maria knocks on the door.", "Maria says: 'Hey Klaus, want to study together?'"]
        }
    ]
    
    # 狀態傳遞 (保留上一輪的計畫)
    current_plan = [] 
    
    # 3. 執行迴圈
    for step in timeline:
        print(f"\n⏰ TIME: {step['time']}")
        
        initial_state = {
            "agent_name": klaus.name,
            "agent_summary": klaus.summary,
            "current_time": step["time"],
            "observations": step["obs"],
            "daily_plan": current_plan, # 傳入上一輪的計畫
            "relevant_memories": [],
            "current_action": None,
            "current_emoji": None
        }
        
        # 執行 LangGraph
        result = klaus.graph.invoke(initial_state)
        
        # 更新計畫 (如果有變動)
        if result.get("daily_plan"):
            current_plan = result["daily_plan"]
            
        print(f"----------------------------------------")

if __name__ == "__main__":
    run_simulation()