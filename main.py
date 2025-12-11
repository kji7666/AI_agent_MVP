import asyncio
import sys
import os
from datetime import datetime, timedelta

# 確保 Python 能找到 src 模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.graph import GenerativeAgent
from src.world.environment import World

async def main():
    # 清除螢幕
    os.system('cls' if os.name == 'nt' else 'clear')
    print("========================================")
    print("🌍 生成式代理：單人模擬模式 (Final Fixed)")
    print("========================================")
    
    # 1. 初始化世界
    print("Example: 正在讀取 world_config.json...")
    try:
        world = World("world_config.json")
    except FileNotFoundError:
        print("❌ 錯誤：找不到 world_config.json，請確保它在專案根目錄。")
        return

    # 2. 初始化 Klaus
    agent_name = "Klaus"
    print(f"🤖 正在喚醒 {agent_name}...")
    
    klaus = GenerativeAgent(
        name=agent_name,
        summary="Klaus 是成大學生，住在宿舍。生活規律，喜歡整潔，目前正致力於撰寫畢業論文。他喜歡在圖書館唸書，累了會喝咖啡。",
        collection_name="text_sim_fixed_v1" # 改個名字確保記憶乾淨
    )
    
    # 3. 設定初始狀態
    current_time = datetime.strptime("2025-06-01 08:00", "%Y-%m-%d %H:%M")
    
    # 初始位置
    start_location = "bedroom"
    world.move_agent(agent_name, start_location)
    
    # [關鍵修正] 狀態變數初始化
    agent_state = {
        "daily_plan": [],
        "short_term_plan": [],
        "busy_until": None,
        "last_location": start_location,
        "current_daily_block_activity": None # 用於紀錄當前正在執行的大任務名稱
    }

    print(f"\n✅ 模擬開始！(按 Ctrl+C 結束)")
    print("="*60)

    try:
        while True:
            # --- A. 顯示環境資訊 ---
            loc_id = agent_state["last_location"]
            loc_name = world.locations_map[loc_id]["name"]
            print(f"\n⏰ {current_time.strftime('%I:%M %p')} | 📍 {loc_name}")
            print("-" * 30)
            
            # --- B. 感知 (Perceive) ---
            observations = world.get_observations(agent_name)
            map_desc = world.get_location_description_for_llm()
            
            # --- C. 思考 (Think - Async) ---
            input_data = {
                "agent_name": klaus.name,
                "agent_summary": klaus.summary,
                "current_time": current_time.strftime("%Y-%m-%d %I:%M %p"),
                "observations": observations,
                "world_map_desc": map_desc,
                # 傳入上一輪的狀態
                "daily_plan": agent_state["daily_plan"],
                "short_term_plan": agent_state["short_term_plan"],
                "busy_until": agent_state["busy_until"],
                "current_daily_block_activity": agent_state["current_daily_block_activity"],
                "relevant_memories": []
            }
            
            # 執行 Graph
            result = await klaus.graph.ainvoke(input_data)
            
            # --- D. 更新狀態 (Update State) ---
            # [關鍵修正] 必須將所有狀態存回，包含 current_daily_block_activity
            agent_state.update({
                "daily_plan": result.get("daily_plan", []),
                "short_term_plan": result.get("short_term_plan", []),
                "busy_until": result.get("busy_until"),
                "current_daily_block_activity": result.get("current_daily_block_activity")
            })
            
            # --- E. 執行動作與物理互動 (Act) ---
            if result.get("skip_thinking"):
                print(f"   ⏳ ({agent_name} 正在忙碌...)")
            else:
                action = result.get("current_action", "")
                emoji = result.get("current_emoji", "")
                target_loc_id = result.get("target_location_id")
                target_obj_id = result.get("target_object_id")
                
                print(f"   🎬 {emoji} {action}")
                
                # --- [防呆補救機制] ---
                # 如果 LLM 忘了給 ID，嘗試從 Action 文字反推
                if not target_loc_id and ("前往" in action or "去" in action):
                    for lid, data in world.locations_map.items():
                        if data['name'] in action:
                            target_loc_id = lid
                            print(f"   🔧 補救導航: {lid}")
                            break
                
                if not target_obj_id and not target_loc_id:
                    # 嘗試補救物品操作
                    current_loc_data = world.locations_map.get(agent_state["last_location"])
                    if current_loc_data and "objects" in current_loc_data:
                        for obj in current_loc_data["objects"]:
                            if obj['name'] in action:
                                target_obj_id = obj['id']
                                print(f"   🔧 補救操作: {target_obj_id}")
                                break

                # 1. 移動邏輯 (Location ID)
                if target_loc_id and target_loc_id in world.locations_map:
                    if target_loc_id != agent_state["last_location"]:
                        target_name = world.locations_map[target_loc_id]["name"]
                        print(f"   🚶 移動前往: {target_name} ({target_loc_id})")
                        world.move_agent(agent_name, target_loc_id)
                        agent_state["last_location"] = target_loc_id
                        
                # 2. 物品互動邏輯 (Object ID)
                elif target_obj_id:
                    # 需從 world.objects_map 查找名稱
                    if target_obj_id in world.objects_map:
                        obj_name = world.objects_map[target_obj_id]["name"]
                        print(f"   👉 操作物品: {obj_name} ({target_obj_id})")
                        
                        # 簡單狀態更新規則
                        if "咖啡" in action or "coffee" in action:
                            world.update_object_state(target_obj_id, "運作中")
                        elif "睡" in action or "sleep" in action:
                            world.update_object_state(target_obj_id, "使用中")
                        elif "整理" in action or "tidy" in action:
                            world.update_object_state(target_obj_id, "整潔")
                        elif "吃" in action or "eat" in action:
                            world.update_object_state(target_obj_id, "空了")

            # --- F. 時間流逝 ---
            await asyncio.sleep(2) 
            current_time += timedelta(minutes=15)

    except KeyboardInterrupt:
        print("\n👋 模擬結束")

if __name__ == "__main__":
    asyncio.run(main())