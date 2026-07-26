# Learning RAG API

一个基于 FastAPI 和 DashScope 的学习资料问答 API。
它会先从 docs 中检索相关资料，再基于检索到的资料生成回答。

## 功能

- 接收用户问题
- 使用 Embedding 计算问题与资料的语义相似度
- 选择最相关的学习资料
- 基于选中的资料调用 qwen-plus 生成回答
- 返回资料来源和相似度分数，便于查看检索结果