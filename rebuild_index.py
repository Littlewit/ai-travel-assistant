import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from modelscope import snapshot_download # 引入魔搭下载工具

load_dotenv()

DATA_DIR = "./data"
VECTOR_STORE_PATH = "./vector_store"
# 【核心优化】：使用魔搭社区的镜像地址下载模型
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LOCAL_MODEL_DIR = "./models/bge-small-zh"

def rebuild():
    if not os.path.exists(DATA_DIR):
        print("❌ data 目录不存在")
        return

    # 1. 通过魔搭社区下载模型到本地
    if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        print(f"📥 正在从魔搭社区下载模型 {MODEL_NAME} ...")
        try:
            snapshot_download(MODEL_NAME, local_dir=LOCAL_MODEL_DIR)
            print("✅ 模型下载成功")
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            return
    else:
        print("✅ 检测到本地已存在模型文件")

    documents = []
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 已切分为 {len(chunks)} 个片段")

    # 2. 使用本地路径加载模型
    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_DIR,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists(VECTOR_STORE_PATH):
        import shutil
        shutil.rmtree(VECTOR_STORE_PATH)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("✅ 索引重建完成！请提交 vector_store 文件夹到 Git")

if __name__ == "__main__":
    rebuild()