from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "AI助教相关数据"
CHROMA_DB_PATH = BASE_DIR / "db" / "chroma_db"

# Data files
DATA_FILES = {
    "coze": DATA_DIR / "coze知识库" / "coze_questions_content.xlsx",
    "video": DATA_DIR / "课程视频知识库" / "video-all.xlsx",
    "wechat": DATA_DIR / "微信群问答知识库" / "AI解决方案专家-答疑汇总-all.xlsx",
}

# Embedding model
EMBEDDING_MODEL = "C:/Users/xin/.cache/modelscope/AI-ModelScope/bge-small-zh-v1___5"

# LLM settings
# Backend: "ollama" (local) or "api" (online)
# API providers: "siliconflow", "dashscope", "openai", "custom"
LLM_BACKEND = "ollama"
LLM_MODEL = "qwen3.5:4b"

# API settings (used when LLM_BACKEND="api")
API_PROVIDER = "siliconflow"  # "siliconflow" / "dashscope" / "openai" / "custom"
API_KEY = ""  # Set via environment variable LLM_API_KEY or fill here

# Pre-configured API providers
API_PROVIDERS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "custom": {
        "base_url": "",
        "model": "",
    },
}

# RAG settings
CHROMA_COLLECTION = "ai_teaching_assistant"
CHUNK_MAX_CHARS = 1000
RETRIEVAL_TOP_K = 5
CONTEXT_MAX_CHARS = 3000

# Source labels (Chinese)
SOURCE_LABELS = {
    "coze": "Coze知识库",
    "video": "课程视频",
    "wechat": "微信群答疑",
}
