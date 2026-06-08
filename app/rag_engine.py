import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 未找到本地向量索引文件夹 vector_store/")
        return None
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("⚠️ 警告: 未配置 DASHSCOPE_API_KEY，RAG 功能将不可用。")
        return None

    # 【核心修改】：换回阿里云 text-embedding-v2，节省内存
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v2", 
        api_key=api_key,  # type: ignore
        base_url=DEFAULT_BASE_URL
    )
    
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