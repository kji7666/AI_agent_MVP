from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import GenerativeAgent
from src.world.environment import World

simulation_data = {
    "world": None,
    "agents": {},
    "current_time": datetime.strptime("2025-06-01 08:00", "%Y-%m-%d %H:%M"),
    "agent_states": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🌍 [Server] 初始化 Data-Driven World...")
    simulation_data["world"] = World("world_config.json")
    
    # 初始化 Klaus，初始位置設為 bedroom
    simulation_data["world"].move_agent("Klaus", "bedroom")
    
    klaus = GenerativeAgent(
        name="Klaus",
        summary="Klaus 是成大學生，住在宿舍。生活規律，喜歡整潔。",
        collection_name="godot_klaus_final_v4"
    )
    simulation_data["agents"]["Klaus"] = klaus
    simulation_data["agent_states"]["Klaus"] = {
        "daily_plan": [],
        "short_term_plan": [],
        "busy_until": None,
        "last_location": "bedroom" # 必須與 JSON ID 一致
    }
    print("✅ [Server] 系統就緒！")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 Godot 獲取地圖
@app.get("/world/map")
async def get_world_map():
    return simulation_data["world"].get_map_config()

@app.get("/agent/decide")
async def agent_decide():
    klaus = simulation_data["agents"]["Klaus"]
    world = simulation_data["world"]
    state = simulation_data["agent_states"]["Klaus"]
    current_time = simulation_data["current_time"]
    
    # 1. 準備輸入
    map_desc = world.get_location_description_for_llm()
    observations = world.get_observations(klaus.name)
    
    agent_input = {
        "agent_name": klaus.name,
        "agent_summary": klaus.summary,
        "current_time": current_time.strftime("%Y-%m-%d %I:%M %p"),
        "observations": observations,
        "world_map_desc": map_desc, # 傳入地圖清單
        "daily_plan": state["daily_plan"],
        "short_term_plan": state["short_term_plan"],
        "busy_until": state["busy_until"],
        "relevant_memories": []
    }
    
    print(f"\n🧠 Processing Tick: {current_time}")
    result = await klaus.graph.ainvoke(agent_input)
    
    # 2. 更新狀態
    state["daily_plan"] = result.get("daily_plan", [])
    state["short_term_plan"] = result.get("short_term_plan", [])
    state["busy_until"] = result.get("busy_until")
    
    # 3. 處理移動與互動
    target_loc_id = result.get("target_location_id")
    target_obj_id = result.get("target_object_id")
    action = result.get("current_action", "")
    
    final_target = None
    
    # 情況 A: 移動到房間
    if target_loc_id and target_loc_id in world.locations_map:
        final_target = target_loc_id
        if target_loc_id != state["last_location"]:
            print(f"   🚶 移動: {state['last_location']} -> {target_loc_id}")
            world.move_agent("Klaus", target_loc_id)
            state["last_location"] = target_loc_id

    # 情況 B: 操作物品 (視為精細移動)
    elif target_obj_id and target_obj_id in world.objects_map:
        final_target = target_obj_id
        print(f"   👉 操作物品: {target_obj_id}")
        
        # 簡單規則更新狀態 (可擴充)
        if "咖啡" in action or "coffee" in action:
            world.update_object_state(target_obj_id, "運作中")
        elif "整理" in action or "tidy" in action:
            world.update_object_state(target_obj_id, "整潔")
        elif "睡" in action:
            world.update_object_state(target_obj_id, "使用中")

    simulation_data["current_time"] += timedelta(minutes=15)

    return {
        "agent": "Klaus",
        "action": action,
        "emoji": result.get("current_emoji"),
        "target_id": final_target, # 統一回傳 ID (不論是地點還是物品)
        "time_display": current_time.strftime("%I:%M %p")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)