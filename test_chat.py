import sys
import os
import re
import time
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm_factory import get_llm
# ==========================================
# 0. 環境與 LLM 設定
# ==========================================

llm = get_llm(temperature=0.7)


# ==========================================
# 1. 資料結構定義
# ==========================================
@dataclass
class Persona:
    name: str
    base_desc: str
    inner_traits: str
    speaking_style: str

klaus_profile = Persona(
    name="Klaus",
    base_desc="15歲的普通女高中生，長相清秀但總是一臉沒睡飽。",
    inner_traits="輕微傲嬌、愛面子、固執、覺得很多事情很麻煩，但其實心地不壞。",
    speaking_style="直來直往、帶有一點攻擊性(吐槽)、不喜歡講長篇大論的廢話。",
)

# ==========================================
# 2. Agent Pipeline 實作
# ==========================================
class AgentPipeline:
    def __init__(self, persona: Persona, llm):
        self.p = persona
        self.llm = llm

    # Step 1: 歸納觀察
    def step_1_observe(self, history: str):
        prompt = ChatPromptTemplate.from_template("""
        對話紀錄:
        {history}
        
        請用一句話歸納現在發生什麼事。
        重點包含：對方說了什麼、當下的氣氛、以及任何顯著的非語言線索。
        """)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"history": history})

    # Step 2: 廣譜生成
    def step_2_generate_options(self, observation: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        你是 {name}。
        內在特質: {inner_traits}                                       
        情境: {observation}
        
        請列出 **10 個** 截然不同的接話方式。
        要求：
        - 列出動作簡述或回應 (例如: "不說話", "轉身離開", "給對方一個擁抱")。
        """)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "observation": observation,
            "name": self.p.name,
            "inner_traits": self.p.inner_traits,
        })
    
    # Step 3: 負向過濾 & 挑選
    def step_3_filter_and_select(self, options_text: str, history : str) -> str:
        
        # 1. [刪去]: 根據正常邏輯，刪掉那些「使對話不流暢」的選項。
        # print(f"\n[DEBUG Options Pool]:\n{options_text}\n") # 除錯用，不想看可以註解掉
        prompt = ChatPromptTemplate.from_template(
        """
        對話紀錄:
        {history}              
        眼前有這些人類可能的反應選項:
        {options}
        
        任務:
        [挑選]: 從剩下的選項中，選出一個使對話最流暢 的行動。
        
        請嚴格依照此格式回傳:
        [選定行動]: 你的行動
        [理由]: ...
        [繼續]: (是/否) -> 如果你想接著講別的、或者想吐槽更多，填"是"；如果講完了等對方回，填"否"。
        """)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "name": self.p.name, 
            "inner_traits": self.p.inner_traits,
            "options": options_text,
            "history": history
        })

    def _extract_action(self, text: str) -> str:
        match = re.search(r"選定行動:\s*(.*?)(\n|$)", text)
        if match:
            return match.group(1).strip()
        return text

    # Step 4: 執行演出
    def step_4_act(self, selected_action: str, history: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        你是 {name}。
        說話風格: {style}
        對話紀錄: {history}
        
        你決定執行的行動是: "{action}"
        
        請演出這個行動。
        
        格式要求:
        [動作]: (描述微表情或動作)
        [語言]: (你的台詞)
        [繼續]: (是/否) -> 如果你想接著講別的、或者想吐槽更多，填"是"；如果講完了等對方回，填"否"。
        """)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "name": self.p.name,
            "style": self.p.speaking_style,
            "history": history,
            "action": selected_action
        })

    # 新增：解析最終輸出的 helper
    def parse_final_output(self, text: str):
        action_match = re.search(r"\[選定行動\]:\s*(.*)", text)
        speech_match = re.search(r"\[理由\]:\s*(.*)", text)
        continue_match = re.search(r"\[繼續\]:\s*(是|否)", text)

        action = action_match.group(1).strip() if action_match else ""
        speech = speech_match.group(1).strip() if speech_match else ""
        should_continue = continue_match.group(1).strip() == "是" if continue_match else False

        return action, speech, should_continue

    def run_step(self, history: str):
        """執行一次完整的思考與回應流程"""
        print(f"\n{'='*10} 思考開始 {'='*10}")
        
        # Step 1
        # obs = self.step_1_observe(history)
        # print(f"👁️ 觀察: {obs}")
        obs = history
        # Step 2
        options = self.step_2_generate_options(obs)
        print(f"🛡️ 選項: {options}")
        # Step 3
        # decision_text = self.step_3_filter_and_select(options, history)
        # clean_action = self._extract_action(decision_text)
        # print(f"🛡️ 決策: {decision_text}")
        # print(f"🛡️ 行動: {clean_action}")
        
        # # Step 4
        # final_output = self.step_4_act(clean_action, history)
        # print(f"🎭 演出:\n{final_output}")
        final_output = self.step_3_filter_and_select(options, history)
        print(f"🎭 演出:\n{final_output}")
        return final_output

# ==========================================
# 3. 主對話迴圈 (The Chat Loop)
# ==========================================
def start_chat_session():
    # 初始化
    agent = AgentPipeline(klaus_profile, llm)
    
    # 初始歷史紀錄
    history = """
    User: (拿著一杯熱可可) 嘿，Klaus，妳修了一整晚都沒睡，喝點熱的吧。
    """
    
    print(f"🎬 對話開始！初始情境:\n{history}")
    
    # 設定一個安全閥，避免 AI 自己講話講到無限迴圈
    auto_loop_limit = 10 
    auto_loop_count = 0

    while True:
        # 1. Agent 執行一次思考與回應
        raw_output = agent.run_step(history)
        
        # 2. 解析輸出
        action, speech, should_continue = agent.parse_final_output(raw_output)
        
        # 3. 格式化 Agent 的回應並更新 History
        # 將 Agent 的反應寫入歷史，讓它下次知道自己做過什麼
        agent_entry = f"Klaus: {action}"
        history += f"\n{agent_entry}"
        
        print(f"\n🗣️ Klaus : {action}")

        # 4. 判斷是否繼續
        if should_continue and auto_loop_count < auto_loop_limit:
            print("\n⏳ Klaus 似乎還想說什麼... (自動繼續)")
            auto_loop_count += 1
            time.sleep(1) # 稍微停頓一下增加真實感
            # 直接進入下一次 while 迴圈，不請求用戶輸入
            continue 
            
        else:
            # 如果不繼續，或者超過自動次數上限，換用戶說話
            if auto_loop_count >= auto_loop_limit:
                print("\n(系統強制換手，避免 Klaus 碎碎念太久)")
            
            auto_loop_count = 0 # 重置計數器
            
            print("\n" + "-"*30)
            user_input = input("👉 換你了 (輸入回應): ")
            
            if user_input.lower() in ["exit", "quit", "掰掰"]:
                print("👋 對話結束。")
                break
                
            # 更新 History
            history += f"\nUser: {user_input}"

# ==========================================
# 4. 執行
# ==========================================
if __name__ == "__main__":
    start_chat_session()