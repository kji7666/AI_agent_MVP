from ollama import Client
import src.config as config

def sanitize_content(text: str) -> str:
    """
    [清洗函數] 移除不合法的 Unicode 代理字元 (Surrogates)。
    這通常發生在 LLM 生成了損壞的 Emoji 或截斷的字串時。
    """
    if not isinstance(text, str):
        return text
    # 原理：嘗試編碼成 utf-8，如果有錯誤(surrogates)就忽略(ignore)，然後再解碼回來
    return text.encode('utf-8', 'ignore').decode('utf-8')

class LLMClient:
    def __init__(self):
        self.client = Client( 
            host=config.API_HOST, # ollama server url
            headers={'Authorization': f'Bearer {config.LLM_API_KEY}'}  # custom https header : 附帶 f'Bearer {config.LLM_API_KEY}' 的 Authorization (驗證)
        )
        self.model = config.MODEL_NAME # for chat

    def chat(self, messages, temperature=0.7):
        try:
            # 在發送前，先清洗所有的訊息內容
            clean_messages = []
            for msg in messages:
                clean_messages.append({
                    "role": msg["role"],
                    "content": sanitize_content(msg["content"])
                })
            
            response = self.client.chat(
                model=self.model,
                messages=clean_messages, # 使用清洗過的訊息
                options={
                    "temperature": temperature,
                }
            )
            return response['message']['content']
            # chat return a dict(response) example
            # {
            #     "id": "unique-response-id",
            #     "model": "llama3.1",
            #     "object": "message",
            #     "message": {
            #         "role": "assistant",
            #         "content": "你好，我是 LLM，很高興幫你回答問題。",
            #         "metadata": {
            #             "tokens_used": 50
            #         }
            #     },
            #     "created": 1700000000
            # }
        except Exception as e:
            print(f"\n❌ [LLM Client Error]: {e}")
            if hasattr(e, 'response'):
                print(f"   Status: {e.response.status_code}")
                print(f"   Body: {e.response.text}")
            return ""

llm_client = LLMClient()

