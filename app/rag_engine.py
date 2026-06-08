import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def load_vector_store():
    """直接加载本地已有的向量索引"""
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 未找到本地向量索引文件夹 vector_store/")
        return None
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    # 即使没有 Key，我们也尝试加载，因为 FAISS 索引本身已经包含向量数据
    # 但为了保持接口一致性，通常还是需要 embedding 对象来转换新查询
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3", # 必须与生成索引时使用的模型一致
        api_key=api_key or "dummy-key", # 如果仅用于加载，有时 dummy key 也能绕过部分检查，但建议配置真实 Key
        base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    )
    
    try:
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ 加载向量库失败: {e}")
        return None

# 初始化检索器
retriever = None
try:
    vs = load_vector_store()
    if vs:
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        print("✅ 成功加载本地向量知识库")
except Exception as e:
    print(f"⚠️ 初始化检索器失败: {e}")

# 如果不需要重新构建，可以注释掉 main 部分的调用
if __name__ == "__main__":
    print("当前模式：仅加载已有索引。如需重新构建，请手动调用 create_vector_store()")
    # create_vector_store() 
