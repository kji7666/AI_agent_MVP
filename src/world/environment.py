# src/world/environment.py

from typing import Dict, List, Optional
from pydantic import BaseModel

class WorldObject(BaseModel):
    id: str
    name: str
    state: str = "閒置" # 改中文預設值
    position: str 

class Location(BaseModel):
    name: str
    objects: Dict[str, WorldObject] = {}

class World:
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        # 👇 [新增] 追蹤所有 Agent 的位置 {agent_name: location_id}
        self.agent_positions: Dict[str, str] = {} 
        self._init_smallville()

    def _init_smallville(self):
        # ... (地點初始化保持不變，確保有 Kitchen, Bedroom, Library, Lecture Hall) ...
        # 建議複製上一輪修改過的中文版 _init_smallville
        kitchen = Location(name="廚房")
        kitchen.objects["stove"] = WorldObject(id="stove", name="瓦斯爐", position="廚房", state="關閉")
        kitchen.objects["fridge"] = WorldObject(id="fridge", name="冰箱", position="廚房", state="滿的")
        kitchen.objects["coffee_machine"] = WorldObject(id="coffee_machine", name="咖啡機", position="廚房", state="閒置")
        self.locations["Kitchen"] = kitchen

        bedroom = Location(name="臥室")
        bedroom.objects["bed"] = WorldObject(id="bed", name="床", position="臥室", state="鋪好的")
        bedroom.objects["desk"] = WorldObject(id="desk", name="書桌", position="臥室", state="雜亂")
        self.locations["Bedroom"] = bedroom

        library = Location(name="圖書館")
        library.objects["bookshelf"] = WorldObject(id="bookshelf", name="書架", position="圖書館", state="滿的")
        self.locations["Library"] = library

        lecture_hall = Location(name="大學講堂")
        lecture_hall.objects["projector"] = WorldObject(id="projector", name="投影機", position="大學講堂", state="關閉")
        self.locations["Lecture Hall"] = lecture_hall

    # 👇 [新增] 用來設定 Agent 初始位置或移動 Agent
    def move_agent(self, agent_name: str, new_location_id: str):
        if new_location_id in self.locations:
            self.agent_positions[agent_name] = new_location_id
            return True
        return False

    # 👇 [修改] 感知功能：現在可以「看到」其他人了！
    def get_observations(self, agent_name: str) -> List[str]:
        # 取得該 Agent 的位置
        current_loc_id = self.agent_positions.get(agent_name)
        if not current_loc_id or current_loc_id not in self.locations:
            return ["你目前不在任何已知的地方。"]
        
        loc = self.locations[current_loc_id]
        obs = [f"你現在位於 {loc.name}。"]
        
        # 1. 看到物件
        for obj in loc.objects.values():
            obs.append(f"這裡有一個 {obj.name}，狀態是: {obj.state}。")
            
        # 2. 👇 [新增] 看到其他人
        # 遍歷所有 Agent，找出「也在同一個地點」且「不是自己」的人
        present_agents = []
        for name, position in self.agent_positions.items():
            if position == current_loc_id and name != agent_name:
                present_agents.append(name)
        
        if present_agents:
            obs.append(f"你看到 {', '.join(present_agents)} 也在這裡。")
            
        return obs

    def update_object_state(self, object_id: str, new_state: str) -> str:
        # ... (保持不變) ...
        for loc in self.locations.values():
            if object_id in loc.objects:
                obj = loc.objects[object_id]
                old_state = obj.state
                obj.state = new_state
                print(f"🌍 [世界事件] {obj.name} 的狀態從 '{old_state}' 變成了 '{new_state}'。")
                return f"你成功將 {obj.name} 變為 {new_state}。"
        return "找不到物件。"