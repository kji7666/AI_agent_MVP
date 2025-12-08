import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.retriever import GenerativeRetriever
from src.agent.reflection import Reflector
from src.agent.planning import Planner

def test_cognition():
    print("========================================")
    print("🧠 TESTING COGNITIVE MODULES")
    print("========================================")
    
    agent_name = "Klaus Mueller"
    
    # 1. 準備記憶庫
    retriever = GenerativeRetriever(collection_name="test_cognition")
    
    # 注入一些觀察，讓 Reflector 有東西可以歸納
    print("\n[Step 0] Seeding Memories...")
    observations = [
        "Klaus is reading a book about sociology.",
        "Klaus spends 3 hours writing his research paper.",
        "Klaus discusses social justice with Maria.",
        "Klaus checks out a book on gentrification from the library."
    ]
    for obs in observations:
        retriever.add_memory(obs)
        
    # 2. 測試反思 (Reflection)
    print("\n[Step 1] Testing Reflection...")
    reflector = Reflector(retriever)
    reflector.run(agent_name)
    
    # 3. 測試規劃 (Planning)
    print("\n[Step 2] Testing Planning...")
    planner = Planner(retriever)
    
    summary = "Klaus Mueller is a dedicated sociology student at Oak Hill College. He is passionate about social justice."
    planner.create_initial_plan(
        agent_name=agent_name,
        agent_summary=summary,
        current_time="07:00 AM"
    )

if __name__ == "__main__":
    test_cognition()