import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 定义本地模型路径
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")

def create_vector_store_with_local_model():
    # ... (加载文档和切片的逻辑保持不变) ...
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

def load_vector_store():
    """加载本地已有的向量索引"""
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 未找到本地向量索引文件夹 vector_store/")
        return None
    
    # 【核心修改】：使用本地下载的模型
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"⚠️ 警告: 未找到本地模型文件夹 {LOCAL_MODEL_PATH}")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_PATH,
        model_kwargs={'device': 'cpu'} # 强制使用 CPU，避免 Render 环境问题
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
        print("✅ 成功加载本地向量知识库 (Local ModelScope)")
except Exception as e:
    print(f"⚠️ 初始化检索器失败: {e}")