import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")

# 【核心优化】：使用极小的中文本地模型，实现毫秒级向量化，彻底消除网络延迟
LOCAL_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 未找到本地向量索引文件夹 vector_store/")
        return None
    
    try:
        # 使用 CPU 运行，内存占用极低
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ 加载向量库失败: {e}")
        return None

retriever = None