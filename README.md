# 🌍 智能 AI 旅游助手 (Smart AI Travel Assistant)

基于 **LangChain**、**LangGraph**、**FAISS** 向量检索与 **FastAPI** 构建的全链路智能旅游咨询系统。本项目结合阿里云千问大模型（Qwen），通过 RAG（检索增强生成）技术解决大模型幻觉问题，为用户提供精准、个性化且具备多轮记忆的旅游行程规划服务。

## ✨ 核心功能

- **🤖 智能问答与规划**：支持景点推荐、美食攻略、出行避坑等多场景链式应答。
- **🧠 RAG 增强检索**：利用 FAISS 向量数据库存储旅游攻略，实现基于本地知识库的精准召回。
- **🔄 LangGraph 工作流编排**：实现意图识别路由，自动区分“闲聊”与“专业旅游咨询”。
- **💬 多轮对话记忆**：基于 SQLite 持久化存储聊天历史，支持上下文连续交互。
- **⚡ 流式实时响应**：前端采用 SSE (Server-Sent Events) 实现打字机效果的流式输出。
- **🎨 Markdown 渲染**：支持行程列表、加粗重点及交通对比表格的实时格式化显示。
- **🗺️ 专业路线建议**：自动生成最佳游览路线，并提供打车、公交、骑行等多种交通方式的优缺点对比。

## 🛠️ 技术栈

| 模块 | 技术选型 |
| :--- | :--- |
| **大模型编排** | LangChain, LangGraph |
| **向量数据库** | FAISS |
| **后端框架** | FastAPI, Uvicorn |
| **大模型基座** | 阿里云 DashScope (Qwen-Turbo) |
| **数据存储** | SQLite (聊天记录), Local File (向量索引) |
| **前端交互** | HTML5, CSS3, JavaScript, Marked.js |
| **依赖管理** | UV / Pip |

## 📂 项目结构

```text
langgraph_ai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口与流式接口
│   ├── graph.py             # LangGraph 智能体工作流定义
│   └── rag_engine.py        # RAG 检索引擎与 FAISS 管理
├── data/                    # 原始旅游知识数据目录 (.txt/.md)
├── vector_store/            # FAISS 向量索引存储目录
├── static/
│   └── index.html           # 前端交互页面
├── chat_history.db          # SQLite 聊天记录数据库
├── .env                     # 环境变量配置 (API Key)
├── pyproject.toml           # 项目依赖管理
└── README.md                # 项目说明文档
```
## 🚀 快速开始
### 1. 环境准备
确保已安装 Python 3.10+ 和 UV 包管理器。
### 2. 安装依赖
```bash
uv sync
# 或者使用 pip
pip install langchain langchain-community langgraph faiss-cpu fastapi uvicorn dashscope python-dotenv pydantic
```
### 3. 配置 API Key
在项目根目录创建 .env 文件，并填入您的阿里云 DashScope API Key：
```env
DASHSCOPE_API_KEY=your_api_key_here
```
### 4. 构建向量知识库
将旅游攻略文件（.txt 或 .md）放入 data/ 目录，然后运行以下命令进行向量化处理：
```bash
uv run python -m app.rag_engine
```
### 5. 启动服务
```bash
uv run uvicorn app.main:app --reload
```
服务启动后，访问浏览器：http://127.0.0.1:8000
## 💡 使用示例
1. 闲聊模式：输入“你好”，AI 将进行亲切互动。
2. 行程规划：输入“帮我规划一个长沙3天2晚的行程”，AI 将结合知识库给出包含路线、交通对比和避坑指南的详细方案。
3. 多轮追问：接着问“那附近有什么好吃的？”，AI 会结合之前的上下文继续推荐。
## 📝 待办事项 (TODO)
- 增加地图可视化功能，直观展示规划路线。
- 接入实时天气 API，提供动态出行建议。
- 支持 PDF 格式知识库文件的自动解析。
- 增加用户登录与多会话隔离管理。
## 📄 许可证
本项目仅供学习与交流使用。
```plaintext
您可以将上述内容保存为 `README.md`。这份文档清晰地展示了项目的技术深度和实用价值，非常适合作为开源项目或个人作品集的说明。
```