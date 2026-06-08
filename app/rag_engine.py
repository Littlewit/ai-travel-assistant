# NEW_FILE_CODE
import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
DATA_DIR = os.path.join(os.getcwd(), "data")

def create_vector_store():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"已创建数据目录: {DATA_DIR}")

    documents = []
    # 加载 TXT 和 MD 文件
    for ext in ["*.txt", "*.md"]:
        try:
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            loader = DirectoryLoader(DATA_DIR, glob=ext, loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            documents.extend(loader.load())
        except Exception as e:
            print(f"加载 {ext} 文件时出错: {e}")

    if not documents:
        print("警告: data 目录下未找到任何 .txt 或 .md 文件")
        return None

    # 文本切片
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 向量化
    from langchain_community.embeddings import DashScopeEmbeddings
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("❌ 错误: 未检测到 DASHSCOPE_API_KEY 环境变量，请在 Render 后台配置。")
    
    try:
        embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
        # 测试一下连通性
        embeddings.embed_query("test")
    except Exception as e:
        print(f"⚠️ Embedding 模型初始化或测试失败: {e}")
        print("请检查 API Key 是否有效以及是否有 Embedding 模型的使用权限。")
        # 在构建阶段如果失败，可以选择抛出异常停止构建，或者返回 None
        raise e

    from langchain_community.vectorstores import FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ 向量知识库构建完成，共处理 {len(chunks)} 个片段。")
    return vector_store

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return create_vector_store()
    
    embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = load_vector_store().as_retriever(search_kwargs={"k": 3}) if load_vector_store() else None
if __name__ == "__main__":
    retriever = load_vector_store()