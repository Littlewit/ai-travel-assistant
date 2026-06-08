import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings  # 切换回原生 SDK

load_dotenv()

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
DATA_DIR = os.path.join(os.getcwd(), "data")

def create_vector_store():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"已创建数据目录: {DATA_DIR}")

    documents = []
    for ext in ["*.txt", "*.md"]:
        try:
            loader = DirectoryLoader(DATA_DIR, glob=ext, loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            documents.extend(loader.load())
        except Exception as e:
            print(f"加载 {ext} 文件时出错: {e}")

    if not documents:
        print("警告: data 目录下未找到任何 .txt 或 .md 文件")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("❌ 错误: 未检测到 DASHSCOPE_API_KEY 环境变量")
    
    # 【核心修改】：使用原生 DashScopeEmbeddings 调用视觉模型
    try:
        embeddings = DashScopeEmbeddings(
            model="tongyi-embedding-vision-plus-2026-03-06", 
            dashscope_api_key=api_key
        )
        # 预测试连通性
        embeddings.embed_query("test")
        print("✅ Embedding 模型连通性测试成功")
    except Exception as e:
        print(f"⚠️ Embedding 模型初始化失败: {e}")
        raise e
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ 向量知识库构建完成，共处理 {len(chunks)} 个片段。")
    return vector_store

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return create_vector_store()
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("❌ 错误: 未检测到 DASHSCOPE_API_KEY 环境变量")

    embeddings = DashScopeEmbeddings(
        model="tongyi-embedding-vision-plus-2026-03-06",
        dashscope_api_key=api_key
    )
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# 初始化
retriever = None
try:
    vs = load_vector_store()
    if vs:
        retriever = vs.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"⚠️ 初始化检索器失败: {e}")

if __name__ == "__main__":
    create_vector_store()