import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 【核心修改】：直接导入我们在 rag_engine 里写好的稳定版 Embedding 类
from app.rag_engine import SimpleDashScopeEmbeddings

load_dotenv()

# 配置
DATA_DIR = "./data"
VECTOR_STORE_PATH = "./vector_store"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

def rebuild():
    if not os.path.exists(DATA_DIR):
        print("❌ data 目录不存在")
        return

    # 1. 加载文档
    documents = []
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(loader.load())
    
    # 2. 切片
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 已切分为 {len(chunks)} 个片段")

    # 3. 使用我们手写的稳定版 Embedding 类
    embeddings = SimpleDashScopeEmbeddings(api_key=API_KEY, model="text-embedding-v2")
    
    # 4. 构建并保存
    if os.path.exists(VECTOR_STORE_PATH):
        import shutil
        shutil.rmtree(VECTOR_STORE_PATH)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("✅ 索引重建完成！请提交 vector_store 文件夹到 Git")

if __name__ == "__main__":
    rebuild()