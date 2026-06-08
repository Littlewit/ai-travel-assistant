import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 【修改】：将路径指向您实际下载的 ./dir 文件夹
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "dir") 

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
DATA_DIR = os.path.join(os.getcwd(), "data")

def create_vector_store():
    if not os.path.exists(DATA_DIR):
        print("❌ 数据目录不存在")
        return

    documents = []
    for ext in ["*.txt", "*.md"]:
        try:
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            loader = DirectoryLoader(DATA_DIR, glob=ext, loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            documents.extend(loader.load())
        except Exception as e:
            print(f"加载出错: {e}")

    if not documents: 
        print("⚠️ 未找到文档")
        return

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 使用本地模型生成向量
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH, model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    if os.path.exists(VECTOR_STORE_PATH):
        import shutil
        shutil.rmtree(VECTOR_STORE_PATH)
    os.makedirs(VECTOR_STORE_PATH)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ 使用本地模型重新构建索引完成，共 {len(chunks)} 个片段。")

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 未找到本地向量索引文件夹 vector_store/")
        return None
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"⚠️ 警告: 未找到本地模型文件夹 {LOCAL_MODEL_PATH}")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH, model_kwargs={'device': 'cpu'})
    
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
        print("✅ 成功加载本地向量知识库 (Local Model)")
except Exception as e:
    print(f"⚠️ 初始化检索器失败: {e}")

if __name__ == "__main__":
    if os.path.exists(LOCAL_MODEL_PATH):
        create_vector_store()
    else:
        print("⚠️ 请先确保模型已下载到 dir 文件夹")