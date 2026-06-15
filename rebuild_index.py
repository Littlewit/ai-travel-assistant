import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from dashscope import TextEmbedding

load_dotenv()

DATA_DIR = "./data"
VECTOR_STORE_PATH = "./vector_store"
DASHSCOPE_EMBEDDING_MODEL = TextEmbedding.Models.text_embedding_v3


class DashScopeEmbeddings(Embeddings):
    """DashScope 远程 Embedding 适配器，实现 LangChain Embeddings 接口"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            resp = TextEmbedding.call(
                model=DASHSCOPE_EMBEDDING_MODEL,
                input=text,
                text_type="document",
            )
            if resp.status_code == 200:
                embeddings.append(resp.output["embeddings"][0]["embedding"])
            else:
                raise RuntimeError(f"DashScope embedding 失败: {resp.message}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = TextEmbedding.call(
            model=DASHSCOPE_EMBEDDING_MODEL,
            input=text,
            text_type="query",
        )
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        else:
            raise RuntimeError(f"DashScope embedding 失败: {resp.message}")


def rebuild():
    if not os.path.exists(DATA_DIR):
        print("❌ data 目录不存在")
        return

    documents = []
    loader = DirectoryLoader(
        DATA_DIR, glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 已切分为 {len(chunks)} 个片段")

    # 使用 DashScope 远程 Embedding API
    embeddings = DashScopeEmbeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        import shutil
        shutil.rmtree(VECTOR_STORE_PATH)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("✅ 索引重建完成！请提交 vector_store 文件夹到 Git")


if __name__ == "__main__":
    rebuild()