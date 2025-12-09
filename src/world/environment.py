import json
import os
from typing import List, Dict, Any

class World:
    def __init__(self, config_path="world_config.json"):
        # 容錯：嘗試在當前目錄或上一層目錄尋找設定檔
        if not os.path.exists(config_path):
            parent_path = os.path.join("..", config_path)
            if os.path.exists(parent_path):
                config_path = parent_path
            else:
                raise FileNotFoundError(f"找不到設定檔: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        # 建立快速查表 (Map)
        self.locations_map = {}
        self.objects_map = {}
        self.agent_positions: Dict[str, str] = {} # {agent_name: location_id}

        # 解析 JSON 結構
        for loc in self.config["locations"]:
            self.locations_map[loc["id"]] = loc
            
            # 處理地點內的物品
            if "objects" in loc:
                for obj in loc["objects"]:
                    obj["parent_location"] = loc["id"]
                    self.objects_map[obj["id"]] = obj

    def get_location_description_for_llm(self) -> str:
        """
        生成給 LLM 看的地圖與物品清單
        """
        descriptions = []
        for loc in self.config["locations"]:
            # 描述地點
            desc = f"- ID: {loc['id']} ({loc['name']}) | 功能: {', '.join(loc.get('affordances', []))}"
            
            # 描述該地點的物品
            objs = []
            if "objects" in loc:
                for obj in loc["objects"]:
                    objs.append(f"[{obj['id']}] {obj['name']}")
            
            if objs:
                desc += f" | 物品: {', '.join(objs)}"
            
            descriptions.append(desc)
            
        return "\n".join(descriptions)

    def get_observations(self, agent_name: str) -> List[str]:
        """
        生成 Agent 的觀察 (包含地點描述、物品狀態、其他 Agent)
        """
        current_loc_id = self.agent_positions.get(agent_name)
        
        # 異常狀態處理
        if not current_loc_id or current_loc_id not in self.locations_map:
            return ["你目前不在任何已知地點。"]
        
        loc_data = self.locations_map[current_loc_id]
        obs = [f"你現在位於 {loc_data['name']} ({loc_data['description']})。"]
        
        # 1. 觀察物品狀態
        if "objects" in loc_data:
            for obj in loc_data["objects"]:
                obs.append(f"這裡有一個 [{obj['id']}] {obj['name']}，狀態是: {obj['state']}。")
        
        # 2. 觀察其他人
        present_agents = []
        for name, position in self.agent_positions.items():
            if position == current_loc_id and name != agent_name:
                present_agents.append(name)
        
        if present_agents:
            obs.append(f"你看到 {', '.join(present_agents)} 也在這裡。")
            
        return obs

    def move_agent(self, agent_name: str, location_id: str):
        """更新 Agent 位置"""
        if location_id in self.locations_map:
            self.agent_positions[agent_name] = location_id
            return True
        return False

    def update_object_state(self, object_id: str, new_state: str):
        """更新物品狀態"""
        if object_id in self.objects_map:
            obj = self.objects_map[object_id]
            print(f"🌍 [物件更新] {obj['name']} ({object_id}): {obj['state']} -> {new_state}")
            obj["state"] = new_state
            return True
        return False