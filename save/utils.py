from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import src.config as config

def get_llm(temperature=0.7):
    """
    回傳 LangChain 的 ChatOllama 物件，已設定好學校的 API Header。
    """
    return ChatOllama(
        base_url=config.API_HOST,
        model=config.MODEL_NAME,
        temperature=temperature,
        # 關鍵：將 Authorization header 注入請求中
        headers={'Authorization': f'Bearer {config.LLM_API_KEY}'}
    )

def get_embedding_model():
    """
    回傳本地運行的 Embedding 模型 (不消耗學校 API Quota)。
    使用 'all-MiniLM-L6-v2'，這是一個速度快且效果好的標準模型。
    """
    print("🔄 Loading local embedding model (this may take a moment first time)...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store():
    """
    初始化或連接 ChromaDB。
    """
    embedding_function = get_embedding_model()
    
    vector_store = Chroma(
        collection_name="agent_memories",
        embedding_function=embedding_function,
        persist_directory=config.DB_PERSIST_DIRECTORY
    )
    return vector_store