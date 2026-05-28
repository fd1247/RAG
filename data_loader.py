import re
import pandas as pd
from config import DATA_FILES, SOURCE_LABELS


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_questions(questions_text: str) -> list[str]:
    text = clean_text(questions_text)
    if not text:
        return []
    text = text.replace("\n||", "||").replace("||\n", "||")
    questions = [q.strip() for q in text.split("||") if q.strip()]
    return questions


def load_excel(source_key: str) -> list[dict]:
    filepath = DATA_FILES[source_key]
    if not filepath.exists():
        print(f"Warning: {filepath} not found, skipping.")
        return []

    df = pd.read_excel(filepath)
    documents = []

    for row_idx, row in df.iterrows():
        raw_questions = row.get("questions", "")
        raw_content = row.get("content", "")

        content = clean_text(str(raw_content))
        if not content:
            continue

        questions = split_questions(str(raw_questions))
        if not questions:
            questions = [content[:50]]

        for q_idx, question in enumerate(questions):
            documents.append({
                "question": question,
                "content": content,
                "source": source_key,
                "source_label": SOURCE_LABELS.get(source_key, source_key),
                "row_index": int(row_idx),
                "question_index": q_idx,
                "total_questions": len(questions),
            })

    return documents


def load_all_data() -> list[dict]:
    all_docs = []
    for source_key in DATA_FILES:
        docs = load_excel(source_key)
        print(f"  {SOURCE_LABELS.get(source_key, source_key)}: {len(docs)} documents")
        all_docs.extend(docs)
    print(f"Total: {len(all_docs)} documents")
    return all_docs


if __name__ == "__main__":
    docs = load_all_data()
    if docs:
        print(f"\nSample document:")
        print(f"  Question: {docs[0]['question'][:100]}...")
        print(f"  Content:  {docs[0]['content'][:100]}...")
        print(f"  Source:   {docs[0]['source_label']}")
