import json
import os
from typing import List, Dict, Any

class World:
    def __init__(self, config_path="world_config.json"):
        if not os.path.exists(config_path):
            # 容錯：如果找不到，試著往上一層找
            config_path = os.path.join("..", config_path)
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.locations_map = {}
        self.objects_map = {}
        self.agent_positions: Dict[str, str] = {} 

        # 解析地點與物品
        for loc in self.config["locations"]:
            self.locations_map[loc["id"]] = loc
            if "objects" in loc:
                for obj in loc["objects"]:
                    obj["parent_location"] = loc["id"]
                    self.objects_map[obj["id"]] = obj

    def get_location_description_for_llm(self) -> str:
        descriptions = []
        for loc in self.config["locations"]:
            desc = f"- ID: {loc['id']} ({loc['name']}) | 功能: {', '.join(loc.get('affordances', []))}"
            if "objects" in loc:
                objs = [f"[{o['id']}] {o['name']}" for o in loc["objects"]]
                desc += f" | 物品: {', '.join(objs)}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def get_observations(self, agent_name: str) -> List[str]:
        current_loc_id = self.agent_positions.get(agent_name)
        if not current_loc_id or current_loc_id not in self.locations_map:
            return ["你目前不在任何已知地點。"]
        
        loc_data = self.locations_map[current_loc_id]
        obs = [f"你現在位於 {loc_data['name']}。"]
        
        if "objects" in loc_data:
            for obj in loc_data["objects"]:
                obs.append(f"這裡有一個 [{obj['id']}] {obj['name']}，狀態是: {obj['state']}。")
                
        # (這裡省略了看見其他人的邏輯，先專注單人)
        return obs

    def move_agent(self, agent_name: str, location_id: str):
        if location_id in self.locations_map:
            self.agent_positions[agent_name] = location_id
            return True
        return False

    def update_object_state(self, object_id: str, new_state: str):
        if object_id in self.objects_map:
            self.objects_map[object_id]["state"] = new_state
            return True
        return False