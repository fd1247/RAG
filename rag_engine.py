from config import RETRIEVAL_TOP_K, CONTEXT_MAX_CHARS
from embedder import Embedder
from vector_store import VectorStore
from llm_client import LLMClient


SYSTEM_PROMPT = (
    "你是一个专业的AI助教，擅长回答关于Coze平台、AI应用开发及相关技术的问题。"
    "请根据提供的参考资料回答用户的问题。如果参考资料中没有足够信息，请明确说明。请用中文回答。"
)


def build_context(results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    context_parts = []
    total_chars = 0

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        source_label = meta.get("source_label", "未知")
        question = doc
        content = meta.get("content", "")
        similarity = 1 - dist

        block = f"[{i+1}] 来源: {source_label} | 相关度: {similarity:.2f}\n问题: {question}\n回答: {content}\n"

        if total_chars + len(block) > CONTEXT_MAX_CHARS:
            break
        context_parts.append(block)
        total_chars += len(block)

    return "\n".join(context_parts)


def build_source_list(results: dict) -> list[dict]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    sources = []
    for doc, meta, dist in zip(docs, metas, distances):
        sources.append({
            "question": doc,
            "content": meta.get("content", "")[:200],
            "source": meta.get("source_label", "未知"),
            "similarity": round(1 - dist, 3),
        })
    return sources


class RAGEngine:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.llm = None
        self.collection = None

    def build_index(self, documents: list[dict]):
        self.collection = self.vector_store.build_index(documents, self.embedder)

    def load_index(self):
        self.collection = self.vector_store.get_or_create_collection(self.embedder)
        count = self.collection.count()
        if count == 0:
            return False
        print(f"Loaded index with {count} documents")
        return True

    def query(self, user_question: str, source_filter: str = None, top_k: int = RETRIEVAL_TOP_K):
        if self.collection is None:
            self.collection = self.vector_store.get_or_create_collection(self.embedder)

        query_embedding = self.embedder.embed_query(user_question)
        results = self.vector_store.search(self.collection, query_embedding, top_k, source_filter)

        context = build_context(results)
        sources = build_source_list(results)

        prompt = f"{SYSTEM_PROMPT}\n\n【参考资料】\n{context}\n\n【用户问题】\n{user_question}"

        if self.llm is None:
            self.llm = LLMClient()

        answer = self.llm.generate(prompt)
        return answer, sources

    def get_stats(self) -> dict:
        return self.vector_store.get_stats()
