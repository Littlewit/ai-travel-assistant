import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from dashscope import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")

# 【内存优化】：使用 DashScope 远程 Embedding API，省掉 PyTorch/Transformers 本地模型
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


def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 未找到本地向量索引文件夹 vector_store/")
        return None

    try:
        embeddings = DashScopeEmbeddings()
        return FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"⚠️ 加载向量库失败: {e}")
        print("💡 提示: 如果报 dimension mismatch，请运行 `uv run python rebuild_index.py` 重建索引（Embedding 模型已切换）")
        return None


retriever = None