import chromadb
from config import CHROMA_DB_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    def get_or_create_collection(self, embedder):
        collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
        )
        return collection

    def build_index(self, documents: list[dict], embedder):
        collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
        )

        existing_count = collection.count()
        if existing_count > 0:
            print(f"Collection already has {existing_count} documents. Clearing...")
            self.client.delete_collection(CHROMA_COLLECTION)
            collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
            )

        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc["question"] for doc in documents]

        print(f"Embedding {len(texts)} documents...")
        embeddings = embedder.embed(texts)

        metadatas = [
            {
                "source": doc["source"],
                "source_label": doc["source_label"],
                "row_index": doc["row_index"],
                "question_index": doc["question_index"],
                "total_questions": doc["total_questions"],
                "content": doc["content"][:5000],
            }
            for doc in documents
        ]

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            print(f"  Indexed {end}/{len(ids)}")

        print(f"Index built: {collection.count()} documents")
        return collection

    def search(self, collection, query_embedding, top_k: int = 5, source_filter: str = None):
        where = {"source": source_filter} if source_filter else None
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_stats(self) -> dict:
        try:
            collection = self.client.get_collection(CHROMA_COLLECTION)
            count = collection.count()
            if count == 0:
                return {"total": 0}

            all_data = collection.get(include=["metadatas"])
            sources = {}
            for meta in all_data["metadatas"]:
                src = meta.get("source_label", "unknown")
                sources[src] = sources.get(src, 0) + 1
            return {"total": count, "by_source": sources}
        except Exception:
            return {"total": 0}
