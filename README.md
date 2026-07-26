# Learning RAG API

一个基于 FastAPI 和 DashScope 的学习资料问答 API。
它会先从 docs 中检索相关资料，再基于检索到的资料生成回答。

## 功能

- 接收用户问题
- 使用 Embedding 计算问题与资料的语义相似度
- 选择最相关的学习资料
- 基于选中的资料调用 qwen-plus 生成回答
- 返回资料来源和相似度分数，便于查看检索结果
- 自动读取 `docs` 文件夹中的所有 `.txt` 学习资料

## 本地运行

1. 在项目根目录创建本地 `.env`，配置 `DASHSCOPE_API_KEY`。该文件不会上传到 GitHub。

2. 创建并激活虚拟环境，然后安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. 启动服务：

```powershell
uvicorn main:app --reload
```

启动后访问 `http://127.0.0.1:8000/docs` 测试 `/ask` 接口。

## 项目结构

```text
learning-rag-api/
├── main.py
├── docs/
│   ├── python.txt
│   ├── fastapi.txt
│   └── knowledge.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## 当前范围

这是一个学习型 RAG MVP，目前会自动读取 `docs` 文件夹中的短文本资料进行检索与问答。

后续可继续迭代：

- 自动读取 `docs` 文件夹中的多个文本文件
- 对较长资料进行分块后再检索
- 引入向量数据库，支持更多资料