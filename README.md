# AI助教 RAG 问答系统

基于 Coze知识库、课程视频、微信群答疑 三份 Excel 问答数据，构建本地 RAG（检索增强生成）问答系统，提供 Web 界面供用户提问。

## 技术架构

| 组件 | 选型 | 说明 |
|------|------|------|
| Embedding | bge-small-zh-v1.5 | 中文向量模型，512维，CPU推理 |
| 向量库 | ChromaDB | 轻量级，无需额外服务，持久化存储 |
| LLM | Ollama / 在线API | 支持本地 Ollama 和多种在线 API |
| Web界面 | Gradio | 内置聊天组件，快速搭建 |

## 环境依赖

- Python 3.11（conda 虚拟环境 `rag_py311`）
- PyTorch 2.5.1+cu121
- sentence-transformers 5.1.0
- chromadb 1.0.20
- gradio 6.15.1
- pandas、openpyxl、accelerate、bitsandbytes

模型文件位置：
- Embedding 模型：`C:/Users/xin/.cache/modelscope/AI-ModelScope/bge-small-zh-v1___5/`（91MB）
- 向量索引：`db/chroma_db/`（构建索引后自动生成）

## 启动方法

```bash
# 1. 激活环境
conda activate rag_py311

# 2. 启动应用
python app.py

# 3. 浏览器访问
http://localhost:7862
```

首次使用点击「构建索引」按钮，等待向量化完成（约2分钟）后即可提问。

## 项目结构

```
RAG/
├── app.py              # Gradio Web 界面入口
├── rag_engine.py       # RAG 核心管道（检索 + 生成）
├── data_loader.py      # Excel 数据解析与预处理
├── embedder.py         # Embedding 模型封装
├── vector_store.py     # ChromaDB 向量存储操作
├── llm_client.py       # LLM 客户端（Ollama / API）
├── config.py           # 全局配置
├── model_configs.json  # 用户保存的 LLM 配置
├── requirements.txt    # 依赖清单
├── AI助教相关数据/      # 原始 Excel 问答数据
└── db/chroma_db/       # 向量索引（gitignore）
```

## 遇到的问题与解决方案

### RTX 5080 与 PyTorch CUDA 不兼容

**现象**：`CUDA error: no kernel image is available for execution on the device`

**原因**：PyTorch 2.5.1 仅支持到 sm_90 架构，RTX 5080 是 sm_120。

**解决**：Embedding 模型强制使用 CPU 推理（`device="cpu"`）。

### HuggingFace 模型下载超时

**原因**：国内网络无法直接访问 HuggingFace，镜像与新版 huggingface_hub 不兼容。

**解决**：通过 ModelScope 下载 bge-small-zh-v1.5（91MB），替代 BGE-M3（2.3GB）。

### Ollama qwen3.5 返回空内容

**原因**：qwen3.5 默认启用思考模式，非流式 API 返回空内容。

**解决**：API 请求中添加 `"think": false` 参数。

### Windows GBK 编码错误

**解决**：设置环境变量 `PYTHONIOENCODING=utf-8`，或直接使用 conda 环境的 Python 路径。
