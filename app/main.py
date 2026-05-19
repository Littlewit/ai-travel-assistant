from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import os, json, uuid, sqlite3
from app.graph import travel_agent_graph

# --- 数据库初始化 ---
DB_NAME = os.path.join(os.getcwd(), "chat_history.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def get_history(session_id, limit=6):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

# --- FastAPI 应用 ---
app = FastAPI(title="AI Travel Assistant Pro")

# 确保 static 目录存在并挂载
static_dir = os.path.join(os.getcwd(), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 补充：处理浏览器直接请求 /favicon.ico 的情况
@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import FileResponse
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return HTMLResponse("")  # 如果不存在则返回空

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

@app.get("/")
async def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Static files not found</h1>", status_code=404)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    save_message(session_id, "human", request.message)
    
    history = get_history(session_id)
    history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history[-6:]])
    
    from app.graph import retriever, llm, ChatPromptTemplate
    
    # 1. 简单的意图识别
    keywords = ["旅游", "景点", "美食", "攻略", "酒店", "交通", "玩", "吃", "住", "行", "推荐", "路线"]
    is_travel_query = any(k in request.message.lower() for k in keywords)
    
    if is_travel_query and retriever:
        docs = retriever.invoke(request.message)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 【核心优化】：增强版 Prompt，要求提供路线和交通对比
        prompt_template = """你是一个资深的智能旅游规划师。请根据以下【背景知识】和用户的【问题】，提供一份详细的旅行建议。

        【背景知识】：
        {context}

        【聊天历史】：
        {history}

        【用户问题】：
        {question}

        **请务必在回复中包含以下内容：**
        1. **最佳路线规划**：结合地理位置，给出一条不走回头路的顺畅游览顺序。
        2. **交通方式对比**：针对主要行程节点，列出打车、公交/地铁、骑行三种方式的优缺点（如：耗时、费用、舒适度、便利性）。
        3. **避坑与建议**：基于背景知识，提醒用户需要注意的预约事项或天气情况。

        请使用清晰的 Markdown 格式（如表格或列表）展示交通对比，方便用户挑选。"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        inputs = {"history": history_str, "context": context, "question": request.message}
    else:
        # 闲聊模式
        prompt_template = """你叫小智，你是一个友好的助手。
        【聊天历史】：{history}
        【用户问题】：{question}"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        inputs = {"history": history_str, "question": request.message}

    def event_generator():
        full_response = ""
        for chunk in chain.stream(inputs):
            if chunk.content:
                full_response += chunk.content
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        
        save_message(session_id, "ai", full_response)
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
