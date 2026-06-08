from typing import Annotated, Sequence, TypedDict, Literal, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI  # 引入 ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.rag_engine import retriever
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    history: str
    context: str

# 【核心修改】：使用 ChatOpenAI 兼容阿里云百炼接口
api_key_value = os.getenv("DASHSCOPE_API_KEY")
llm = ChatOpenAI(
    model="qwen-plus", # 【提速】：使用 qwen-plus 获得更快的响应速度
    api_key=SecretStr(api_key_value) if api_key_value else None,
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    temperature=0.7,
    streaming=True,
    extra_body={"enable_thinking": False}  # 保持关闭深度思考
)

def router_node(state: AgentState):
    """根据关键词简单判断意图"""
    last_msg_content = state["messages"][-1].content
    if isinstance(last_msg_content, str):
        last_msg = last_msg_content.lower()
    else:
        last_msg = str(last_msg_content).lower()
    keywords = ["旅游", "景点", "美食", "攻略", "酒店", "交通", "玩", "吃", "住", "行", "推荐", "路线"]
    if any(k in last_msg for k in keywords):
        return "rag_node"
    return "chat_node"

def rag_node(state: AgentState):
    """RAG 检索增强节点"""
    last_msg_content = state["messages"][-1].content
    question_str = last_msg_content if isinstance(last_msg_content, str) else str(last_msg_content)
    
    context = "暂无相关背景知识。"
    if retriever is not None:
        try:
            docs: List[Document] = retriever.invoke(question_str)
            if docs:  # 确保docs不是None或空列表
                context = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"RAG检索出错: {e}")
            context = "检索背景知识时出现错误。"
    
    prompt_template = """你是一个资深的智能旅游规划师。请根据以下【背景知识】和用户的【问题】，提供一份详细的旅行建议。

    【背景知识】：
    {context}

    【聊天历史】：
    {history}

    【用户问题】：
    {question}

    **请务必在回复中包含以下内容：**
    1. **最佳路线规划**：结合地理位置，给出一条不走回头路的顺畅游览顺序。
    2. **交通方式对比**：针对主要行程节点，列出打车、公交/地铁、骑行三种方式的优缺点。
    3. **避坑与建议**：基于背景知识，提醒用户需要注意的预约事项或天气情况。

    请使用清晰的 Markdown 格式展示。"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "history": state.get("history", ""), "question": state["messages"][-1].content})
    return {"messages": [response]}

def chat_node(state: AgentState):
    """纯闲聊节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("rag_node", rag_node)
workflow.add_node("chat_node", chat_node)

workflow.set_conditional_entry_point(
    router_node,
    path_map={"rag_node": "rag_node", "chat_node": "chat_node"}
)

workflow.add_edge("rag_node", END)
workflow.add_edge("chat_node", END)

travel_agent_graph = workflow.compile()