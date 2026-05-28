# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 常用命令

```bash
# 激活环境
conda activate rag_py311

# 启动应用（Gradio 界面 http://localhost:7862）
python app.py

# 重建向量索引（首次使用或数据变更后执行）
python -c "from data_loader import load_all_data; from embedder import Embedder; from vector_store import VectorStore; vs=VectorStore(); e=Embedder(); vs.build_index(load_all_data(), e)"
```

## 架构概览

基于 RAG 的 AI 助教问答系统。三份 Excel 数据（Coze文档、课程视频、微信群答疑）→ 19,684 条文档 → ChromaDB 向量库 → LLM 生成回答（Ollama 本地或 API 在线）。

**数据流：** `data_loader.py`（解析 Excel，按 `||` 拆分问题）→ `embedder.py`（bge-small-zh-v1.5，CPU 推理）→ `vector_store.py`（ChromaDB 余弦相似度检索）→ `rag_engine.py`（RAG 管道编排）→ `llm_client.py`（Ollama 或 OpenAI 兼容 API）→ `app.py`（Gradio 聊天界面）

**关键约束：** RTX 5080（sm_120）与 PyTorch 2.5.1 不兼容，Embedding 强制使用 CPU。Ollama 需设置 `think: false` 避免 qwen3.5 模型返回空内容。

## 配置说明

- `config.py` — 路径、模型名称、RAG 参数等全局配置
- `model_configs.json` — 用户通过界面保存的 LLM 配置
- LLM 后端：`"ollama"`（本地）或 `"api"`（硅基流动/百炼/OpenAI/自定义）
- 数据分隔符：`||` 和 `\n||` 统一按 `||` 拆分
