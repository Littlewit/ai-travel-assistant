from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from app.rag_engine import retriever
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    history: str
    context: str

llm = ChatTongyi(
    model="qwen-plus", 
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), 
    temperature=0.7,
    streaming=True # 开启流式
)

def router_node(state: AgentState):
    """根据关键词简单判断意图"""
    last_msg = state["messages"][-1].content.lower()
    keywords = ["旅游", "景点", "美食", "攻略", "酒店", "交通", "玩", "吃", "住", "行", "推荐"]
    if any(k in last_msg for k in keywords):
        return "rag_node"
    return "chat_node"

def rag_node(state: AgentState):
    """RAG 检索增强节点"""
    if retriever:
        docs = retriever.invoke(state["messages"][-1].content)
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        context = "暂无相关背景知识。"
    
    prompt_template = """你是一个专业的智能旅游助手。请结合以下信息回答：
    
    【背景知识】：
    {context}
    
    【用户问题】：
    {question}
    
    请给出详细、贴心且有条理的建议："""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    # 注意：在 graph 节点中直接 stream 比较难处理，通常我们在这里 invoke 拿到完整结果
    # 为了实现流式，我们在 main.py 中直接处理链式调用会更灵活。
    # 这里我们先返回一个标记，或者我们换一种思路：
    # 既然要流式，我们不在 graph 内部 invoke，而是让 graph 只负责路由，
    # 具体的 LLM 调用放在 main.py 的 stream 循环里？
    # 不，最标准的做法是：Graph 负责逻辑，Stream 负责输出。
    
    response = chain.invoke({"context": context, "question": state["messages"][-1].content})
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