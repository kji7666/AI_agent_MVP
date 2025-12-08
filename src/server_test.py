from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# 允許 Godot 連線 (CORS 設定)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模擬狀態
start_time = time.time()

@app.get("/snapshot")
def get_world_snapshot():
    # 簡單的邏輯：每 10 秒換一個目標地點
    # 假設你的 Godot 視窗大概是 1152x648
    # 我們設定兩個座標點：(100, 100) 和 (400, 300)
    
    elapsed = int(time.time() - start_time)
    
    if (elapsed // 10) % 2 == 0:
        target = {"x": 100, "y": 100, "location_name": "Kitchen"}
        action = "Cooking"
    else:
        target = {"x": 400, "y": 300, "location_name": "Library"}
        action = "Reading"

    return {
        "time": "08:00 AM",
        "agents": [
            {
                "name": "Klaus",
                "position": target,
                "action": action,
                "emoji": "🍳" if action == "Cooking" else "📚"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # 啟動伺服器在 port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)