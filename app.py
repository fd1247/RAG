import gradio as gr
from rag_engine import RAGEngine
from data_loader import load_all_data

engine = RAGEngine()

SOURCE_CHOICES = ["全部", "Coze知识库", "课程视频", "微信群答疑"]
SOURCE_FILTER_MAP = {
    "全部": None,
    "Coze知识库": "coze",
    "课程视频": "video",
    "微信群答疑": "wechat",
}


def build_index():
    docs = load_all_data()
    if not docs:
        return "No data loaded."
    engine.build_index(docs)
    stats = engine.get_stats()
    return f"Index built: {stats['total']} documents"


def chat(message, history, source_filter):
    source_key = SOURCE_FILTER_MAP.get(source_filter)
    answer, sources = engine.query(message, source_filter=source_key)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    source_display = ""
    for i, s in enumerate(sources):
        source_display += (
            f"**[{i+1}] {s['source']}** (相关度: {s['similarity']:.2f})\n"
            f"问题: {s['question'][:100]}\n"
            f"摘要: {s['content'][:150]}...\n\n"
        )

    return history, source_display, ""


def get_stats():
    stats = engine.get_stats()
    if stats["total"] == 0:
        return "Index is empty. Please build the index first."
    lines = [f"Total documents: {stats['total']}"]
    for src, count in stats.get("by_source", {}).items():
        lines.append(f"  - {src}: {count}")
    return "\n".join(lines)


with gr.Blocks(title="AI助教 RAG 问答系统") as demo:
    gr.Markdown("# AI助教 RAG 问答系统")
    gr.Markdown("基于 Coze知识库、课程视频、微信群答疑 的智能问答系统")

    with gr.Row():
        with gr.Column(scale=3):
            source_dropdown = gr.Dropdown(
                choices=SOURCE_CHOICES,
                value="全部",
                label="知识源筛选",
            )
            chatbot = gr.Chatbot(height=500, label="对话")
            msg_input = gr.Textbox(placeholder="请输入你的问题...", label="问题", lines=2)
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")

        with gr.Column(scale=2):
            gr.Markdown("### 检索来源")
            source_display = gr.Markdown(value="暂无检索结果")
            index_status = gr.Markdown(value="")
            with gr.Row():
                build_btn = gr.Button("构建索引")
                stats_btn = gr.Button("查看索引状态")

    msg_input.submit(
        chat,
        inputs=[msg_input, chatbot, source_dropdown],
        outputs=[chatbot, source_display, msg_input],
    )
    submit_btn.click(
        chat,
        inputs=[msg_input, chatbot, source_dropdown],
        outputs=[chatbot, source_display, msg_input],
    )
    clear_btn.click(lambda: ([], "暂无检索结果", ""), outputs=[chatbot, source_display, msg_input])
    build_btn.click(build_index, outputs=[index_status])
    stats_btn.click(get_stats, outputs=[index_status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
