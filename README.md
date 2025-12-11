# AI_agent_MVP

## TODO
graph.py
        # --- 邏輯 B: 推進短期計畫 ---
        # 假設完成此動作後，就移除第一個細項
        if short_term and not result.get("should_replan", False):
            # 這裡簡單移除，實際應用可比對時間
            # short_term.pop(0) 
            pass

        def interview(self, question: str):
            # 簡單的同步接口，實際應使用 async
            pass

        # 檢查是否有「重大事件」打斷
        # 簡單判定：如果觀察只有"環境描述" ("你現在位於...", "這裡有一個...")，就不打斷
        # 如果有其他訊息 (e.g. "Fire!", "Maria is talking to you")，視為打斷
        is_routine = all("你現在位於" in o or "這裡有一個" in o or "You are" in o or "There is" in o for o in state["observations"])
        conti 的判定太草率

        沒有加入 reflection

## new idea
身分 prompt 要能隨著 reflection 改變
從 plan 驅動改為 LLM 隨時決定
說話時要有裡外思考

問題是記憶