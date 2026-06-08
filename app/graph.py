from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# 【核心修改】：使用 ChatOpenAI 兼容阿里云百炼接口
api_key_value = os.getenv("DASHSCOPE_API_KEY")
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=SecretStr(api_key_value) if api_key_value else None,
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    temperature=0.7,
    streaming=True,
    extra_body={"enable_thinking": False}
)