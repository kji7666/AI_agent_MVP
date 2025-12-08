import asyncio
import sys
import os
from datetime import datetime, timedelta

# 修正路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import GenerativeAgent
from src.world.environment import World

async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("========================================")
    print("🌍 小鎮模擬：多代理人版 (Multi-Agent)")
    print("========================================")
    
    world = World()
    
    # --- 1. 初始化 Agents ---
    print("🤖 正在初始化居民...")
    
    # 定義 Agent 清單
    agents_config = [
        {
            "name": "Klaus",
            "summary": "Klaus 是社會系學生。他喜歡整潔，正在寫論文，常去圖書館。",
            "collection": "agent_klaus_multi_v1",
            "start_loc": "Bedroom"
        },
        {
            "name": "Maria",
            "summary": "Maria 是一個熱愛物理的學生。她喜歡喝咖啡，經常在圖書館唸書，個性開朗。",
            "collection": "agent_maria_multi_v1",
            "start_loc": "Library" # Maria 一開始在圖書館
        }
    ]
    
    agents = []
    # 用字典來儲存每個 Agent 的執行狀態 (Plan, Busy, etc.)
    agent_states_memory = {} 

    for cfg in agents_config:
        print(f"   ➕ 建立 {cfg['name']}...")
        agent = GenerativeAgent(
            name=cfg["name"],
            summary=cfg["summary"],
            collection_name=cfg["collection"]
        )
        agents.append(agent)
        
        # 設定初始位置
        world.move_agent(cfg["name"], cfg["start_loc"])
        
        # 初始化記憶體狀態
        agent_states_memory[cfg["name"]] = {
            "daily_plan": [],
            "short_term_plan": [],
            "busy_until": None
        }

    current_time = datetime.strptime("2025-06-01 08:00", "%Y-%m-%d %H:%M")
    
    print(f"\n✅ 模擬開始！Klaus 在臥室，Maria 在圖書館。")
    
    # --- 2. 主迴圈 ---
    while True:
        print(f"\n⏰ {current_time.strftime('%I:%M %p')}")
        print("-" * 50)
        
        # 每個 Agent 輪流行動
        for agent in agents:
            name = agent.name
            
            # 取得當前位置
            loc_id = world.agent_positions[name]
            loc_name = world.locations[loc_id].name
            print(f"\n👤 {name} (位於: {loc_name})")
            
            # A. 感知 (從 World 拿，現在包含「看到其他人」)
            observations = world.get_observations(name)
            print(f"   👀 觀察: {observations}")

            # B. 讀取上一輪的狀態
            mem = agent_states_memory[name]
            
            # C. 組裝 State
            input_state = {
                "agent_name": name,
                "agent_summary": agent.summary,
                "current_time": current_time.strftime("%Y-%m-%d %I:%M %p"),
                "observations": observations,
                "daily_plan": mem["daily_plan"],
                "short_term_plan": mem["short_term_plan"],
                "busy_until": mem["busy_until"],
                "relevant_memories": []
            }
            
            # D. 思考 (Async)
            # print(f"   🧠 思考中...")
            result = await agent.graph.ainvoke(input_state)
            
            # E. 更新狀態記憶
            mem["daily_plan"] = result.get("daily_plan", [])
            mem["short_term_plan"] = result.get("short_term_plan", [])
            mem["busy_until"] = result.get("busy_until") # 這裡會拿到 "skip_thinking" 時的 None 或 原值
            
            # 處理顯示
            if result.get("skip_thinking"):
                print(f"   ⏳ (繼續執行上一個動作...)")
                action = "BUSY" # 標記為忙碌，不觸發規則引擎
            else:
                action = result.get("current_action", "發呆")
                emoji = result.get("current_emoji", "😐")
                print(f"   🎬 決定: {emoji} {action}")
            
            # F. 規則引擎 (處理移動與互動)
            if action != "BUSY":
                action_lower = action.lower()
                
                # 移動邏輯 (更新 World 的位置表)
                target_loc = None
                if "廚房" in action or "kitchen" in action_lower: target_loc = "Kitchen"
                elif "圖書館" in action or "library" in action_lower: target_loc = "Library"
                elif "臥室" in action or "bedroom" in action_lower: target_loc = "Bedroom"
                elif "講堂" in action or "lecture" in action_lower: target_loc = "Lecture Hall"
                
                if target_loc:
                    world.move_agent(name, target_loc)
                    print(f"   🚶 移動到了 {world.locations[target_loc].name}")

                # 物件互動邏輯 (簡化版)
                if "整理" in action and loc_id == "Bedroom":
                    world.update_object_state("desk", "整潔")
                elif "咖啡" in action and loc_id == "Kitchen":
                    world.update_object_state("coffee_machine", "沖泡中")
        
        # 時間流逝
        await asyncio.sleep(1) 
        current_time += timedelta(minutes=15)
        # input("Press Enter...") # Debug 用

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 模擬結束。")