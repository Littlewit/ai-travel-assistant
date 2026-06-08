from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import os, json, uuid, sqlite3

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

# 【核心修改】：配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.getcwd(), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 全局检索器，初始为 None
retriever = None
model_loading_lock = False # 防止并发重复加载

def _background_load_model():
    """在后台线程中加载模型，不阻塞主进程"""
    global retriever, model_loading_lock
    if model_loading_lock or retriever is not None:
        return
    
    model_loading_lock = True
    try:
        print("⏳ [后台] 正在加载向量库和 Embedding 模型...")
        from app.rag_engine import load_vector_store
        vs = load_vector_store()
        if vs:
            retriever = vs.as_retriever(search_kwargs={"k": 3})
            print("✅ [后台] 向量库加载成功！")
        else:
            print("⚠️ [后台] 向量库加载返回 None")
    except Exception as e:
        print(f"❌ [后台] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model_loading_lock = False

# 启动时触发后台加载
import threading
threading.Thread(target=_background_load_model, daemon=True).start()

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.get("/")
async def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Static files not found</h1>", status_code=404)

@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import FileResponse
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return HTMLResponse("")

@app.get("/health")
async def health_check():
    return {"status": "ok", "retriever_loaded": retriever is not None}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    save_message(session_id, "human", request.message)
    
    # 【核心修改】：直接导入，模型已在后台线程加载
    try:
        from app.graph import llm
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        def error_generator():
            yield f"data: {json.dumps({'content': f'⚠️ 系统初始化失败: {str(e)}'})}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    history = get_history(session_id)
    history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history[-6:]])
    
    keywords = ["旅游", "景点", "美食", "攻略", "酒店", "交通", "玩", "吃", "住", "行", "推荐", "路线"]
    is_travel_query = any(k in request.message.lower() for k in keywords)
    
    context = ""
    if is_travel_query and retriever:
        try:
            # 【核心优化】：使用 concurrent.futures 实现跨平台超时控制
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            
            def do_retrieval():
                if retriever:
                    return retriever.invoke(request.message)
                return []

            with ThreadPoolExecutor() as executor:
                future = executor.submit(do_retrieval)
                docs = future.result(timeout=5)  # 5秒超时
                
            context = "\n\n".join([doc.page_content for doc in docs])
            print(f"✅ 检索成功，找到 {len(docs)} 条相关知识")
                
        except FuturesTimeoutError:
            print("⚠️ RAG 检索超时，切换为纯 LLM 模式")
        except Exception as e:
            import traceback
            print(f"⚠️ 检索出错: {type(e).__name__}: {str(e)}")
            print(traceback.format_exc()) # 打印完整堆栈

    if context:
        prompt_template = """你是一个资深的智能旅游规划师。请根据以下【背景知识】和用户的【问题】，提供一份详细的旅行建议。
        【背景知识】：{context}
        【聊天历史】：{history}
        【用户问题】：{question}
        **请务必在回复中包含以下内容：**
        1. **最佳路线规划**：结合地理位置，给出一条不走回头路的顺畅游览顺序。
        2. **交通方式对比**：针对主要行程节点，列出打车、公交/地铁、骑行三种方式的优缺点。
        3. **避坑与建议**：基于背景知识，提醒用户需要注意的预约事项或天气情况。
        请使用清晰的 Markdown 格式展示。"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        inputs = {"history": history_str, "context": context, "question": request.message}
    else:
        prompt_template = """你是一个友好的助手。
        【聊天历史】：{history}
        【用户问题】：{question}"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        inputs = {"history": history_str, "question": request.message}

    def event_generator():
        full_response = ""
        try:
            for chunk in chain.stream(inputs):
                if hasattr(chunk, 'content') and chunk.content:
                    content_str = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    full_response += content_str
                    # 【核心优化】：确保每个数据包都立即发送，防止缓冲导致连接超时
                    yield f"data: {json.dumps({'content': content_str})}\n\n"
        except Exception as e:
            # 如果流式生成中途出错，发送错误信息给前端
            yield f"data: {json.dumps({'content': f'\n⚠️ 生成中断: {str(e)}'})}\n\n"
        
        save_message(session_id, "ai", full_response)
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )

if __name__ == "__main__":
    import uvicorn
    # 强制使用 10000 端口（Render 默认期望的端口范围）
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

