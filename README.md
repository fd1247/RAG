# AI助教 RAG 问答系统

基于 Coze知识库、课程视频、微信群答疑 三份 Excel 问答数据，构建本地 RAG（检索增强生成）问答系统，提供 Web 界面供用户提问。

## 技术架构

| 组件 | 选型 | 说明 |
|------|------|------|
| Embedding | bge-small-zh-v1.5 | 中文向量模型，512维，CPU推理 |
| 向量库 | ChromaDB | 轻量级，无需额外服务，持久化存储 |
| LLM | Ollama + qwen3.5:4b | 本地大模型，中文能力强 |
| Web界面 | Gradio | 内置聊天组件，快速搭建 |

## 项目结构

```
RAG/
├── app.py              # Gradio Web 界面入口
├── rag_engine.py       # RAG 核心管道（检索 + 生成）
├── data_loader.py      # Excel 数据解析与预处理
├── embedder.py         # Embedding 模型封装
├── vector_store.py     # ChromaDB 向量存储操作
├── llm_client.py       # LLM 客户端（Ollama API）
├── config.py           # 全局配置
├── requirements.txt    # 依赖清单
└── db/chroma_db/       # 向量索引持久化目录
```

## 执行过程

### 1. 环境准备

确认 conda 虚拟环境 `rag_py311` 中已安装 PyTorch 2.5.1、sentence-transformers、chromadb 等核心依赖。补装缺失的 gradio、openpyxl、pandas、accelerate、bitsandbytes。

### 2. 数据加载与预处理

解析三个 Excel 文件，共 3,240 行原始数据：

- **Coze知识库**：325 行，每行约 10 个相关问题，分隔符 `||`
- **课程视频**：1,167 行，每行约 5 个问题，分隔符 `\n||`
- **微信群答疑**：1,748 行，每行约 5-6 个问题，分隔符 `\n||`

统一分隔符处理后拆分为独立问题，最终生成 **19,684 条** 文档，每条包含问题文本、回答内容和来源元数据。

### 3. Embedding 模型加载

通过 ModelScope 下载 bge-small-zh-v1.5 模型（91MB），在 CPU 上完成 19,684 条文档的向量化，生成 512 维稠密向量。

### 4. 向量索引构建

使用 ChromaDB 持久化存储，按 500 条一批写入，支持基于 source 字段的元数据过滤（Coze/视频/微信）。

### 5. LLM 集成

通过 Ollama 本地运行的 qwen3.5:4b 模型，使用 OpenAI 兼容 API 调用。需关闭思考模式（`think: false`）以获得正常输出。

### 6. Web 界面

Gradio 实现聊天界面，支持知识源筛选、检索来源展示、索引构建与状态查看。

## 遇到的问题与解决方案

### 问题 1：RTX 5080 与 PyTorch CUDA 不兼容

**现象**：`CUDA error: no kernel image is available for execution on the device`

**原因**：conda 环境中的 PyTorch 2.5.1 仅支持到 sm_90 架构，而 RTX 5080 是 sm_120 架构。

**解决**：将 Embedding 模型强制使用 CPU 推理（`device="cpu"`）。bge-small-zh-v1.5 体积小，CPU 推理速度可接受。

### 问题 2：HuggingFace 模型下载超时

**现象**：从 HuggingFace 下载 BGE-M3（2.3GB）速度极慢，curl 直连超时。

**原因**：国内网络无法直接访问 HuggingFace，镜像 `hf-mirror.com` 与新版 huggingface_hub（0.34.4）不兼容。

**解决**：
1. 放弃 BGE-M3，改用更小的 bge-small-zh-v1.5（91MB）
2. 通过 ModelScope（国内模型托管平台）下载，速度稳定在 ~280KB/s

### 问题 3：Ollama qwen3.5 返回空内容

**现象**：通过 OpenAI 兼容 API 调用 Ollama 时，`response.choices[0].message.content` 为空。

**原因**：qwen3.5 默认启用思考模式（thinking mode），思考过程在流式输出中处理，非流式 API 返回空内容。

**解决**：在 API 请求中添加 `"think": false` 参数关闭思考模式。

### 问题 4：Gradio Chatbot 参数不兼容

**现象**：`TypeError: Chatbot.__init__() got an unexpected keyword argument 'type'`

**原因**：安装的 Gradio 版本不支持 `type="messages"` 参数（旧版本使用元组格式）。

**解决**：移除 `type="messages"` 参数，改用 `(user_msg, assistant_msg)` 元组格式管理对话历史。

### 问题 5：Windows GBK 编码错误

**现象**：`UnicodeEncodeError: 'gbk' codec can't encode character`

**原因**：Windows 终端默认 GBK 编码，conda run 输出中文时编码失败。

**解决**：设置环境变量 `PYTHONIOENCODING=utf-8`，或直接使用 conda 环境的 Python 可执行文件路径。
