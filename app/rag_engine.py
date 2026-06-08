import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

class SimpleDashScopeEmbeddings(Embeddings):
    def __init__(self, api_key, model="text-embedding-v2"):
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # 【核心修复】：严格按照阿里云兼容模式要求发送 input 为 list of strings
        payload = {
            "model": self.model,
            "input": {"texts": texts} 
        }
        response = requests.post(DEFAULT_BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 未找到本地向量索引文件夹 vector_store/")
        return None
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("⚠️ 警告: 未配置 DASHSCOPE_API_KEY，RAG 功能将不可用。")
        return None

    embeddings = SimpleDashScopeEmbeddings(api_key=api_key, model="text-embedding-v2")
    
    try:
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ 加载向量库失败: {e}")
        return None

print("🚀 正在初始化 RAG 引擎...")
retriever = None
try:
    print("📂 正在加载向量库...")
    vs = load_vector_store()
    if vs:
        print("🔍 正在创建检索器...")
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        print("✅ RAG 引擎初始化完成！")
except Exception as e:
    print(f"⚠️ 初始化检索器失败: {e}")