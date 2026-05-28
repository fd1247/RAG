import json
from pathlib import Path
import gradio as gr
import config
from rag_engine import RAGEngine
from llm_client import LLMClient
from data_loader import load_all_data

engine = RAGEngine()

CONFIGS_FILE = Path(__file__).parent / "model_configs.json"

SOURCE_CHOICES = ["全部", "Coze知识库", "课程视频", "微信群答疑"]
SOURCE_FILTER_MAP = {
    "全部": None,
    "Coze知识库": "coze",
    "课程视频": "video",
    "微信群答疑": "wechat",
}
PROVIDER_CHOICES = ["siliconflow", "dashscope", "openai", "custom"]


def load_configs() -> dict:
    if CONFIGS_FILE.exists():
        return json.loads(CONFIGS_FILE.read_text(encoding="utf-8"))
    return {}


def save_configs(cfgs: dict):
    CONFIGS_FILE.write_text(json.dumps(cfgs, ensure_ascii=False, indent=2), encoding="utf-8")


def get_config_names():
    return list(load_configs().keys())


def refresh_config_list():
    names = get_config_names()
    return gr.update(choices=names, value=names[0] if names else None)


def show_add_form():
    return gr.update(visible=True)


def hide_add_form():
    return (
        gr.update(visible=False),
        "",
        "ollama",
        "qwen3.5:4b",
        "siliconflow",
        "",
        "",
        "",
    )


def update_form_visibility(backend, provider):
    show_api = backend == "api"
    return (
        gr.update(visible=show_api),
        gr.update(visible=show_api),
        gr.update(visible=not show_api),
        gr.update(visible=show_api),
        gr.update(visible=show_api),
    )


def add_config(name, backend, provider, api_key, ollama_model, custom_url, custom_model):
    if not name.strip():
        return "请输入配置名称", refresh_config_list(), gr.update(visible=True)
    cfgs = load_configs()
    cfg = {
        "backend": backend,
        "provider": provider,
        "api_key": api_key,
    }
    if backend == "ollama":
        cfg["ollama_model"] = ollama_model
    else:
        cfg["custom_url"] = custom_url
        cfg["custom_model"] = custom_model
    cfgs[name.strip()] = cfg
    save_configs(cfgs)
    return f"已保存: {name.strip()}", refresh_config_list(), gr.update(visible=False)


def delete_config(name):
    if not name:
        return "请选择要删除的配置", refresh_config_list()
    cfgs = load_configs()
    if name in cfgs:
        del cfgs[name]
        save_configs(cfgs)
        return f"已删除: {name}", refresh_config_list()
    return "配置不存在", refresh_config_list()


def select_config(name):
    if not name:
        return tuple([gr.update()] * 7)
    cfgs = load_configs()
    cfg = cfgs.get(name, {})
    backend = cfg.get("backend", "ollama")
    provider = cfg.get("provider", "siliconflow")
    api_key = cfg.get("api_key", "")
    ollama_model = cfg.get("ollama_model", "qwen3.5:4b")
    custom_url = cfg.get("custom_url", "")
    custom_model = cfg.get("custom_model", "")

    show_api = backend == "api"
    show_custom = provider == "custom" and show_api

    return (
        gr.update(value=backend),
        gr.update(value=ollama_model),
        gr.update(value=provider, visible=show_api),
        gr.update(value=api_key, visible=show_api),
        gr.update(value=custom_url, visible=show_custom),
        gr.update(value=custom_model, visible=show_custom),
        "",
    )


def apply_config(name):
    if not name:
        return "请先选择一个配置"
    cfgs = load_configs()
    cfg = cfgs.get(name)
    if not cfg:
        return "配置不存在"

    try:
        config.LLM_BACKEND = cfg["backend"]
        config.API_PROVIDER = cfg["provider"]
        config.API_KEY = cfg.get("api_key", "")

        if cfg["backend"] == "ollama":
            config.LLM_MODEL = cfg.get("ollama_model", "qwen3.5:4b")
        elif cfg["provider"] == "custom":
            base_url = cfg.get("custom_url", "")
            model = cfg.get("custom_model", "")
            if not base_url or not model:
                return "自定义配置缺少 Base URL 或模型名称"
            config.API_PROVIDERS["custom"]["base_url"] = base_url
            config.API_PROVIDERS["custom"]["model"] = model
            config.LLM_MODEL = model
        else:
            config.LLM_MODEL = config.API_PROVIDERS[cfg["provider"]]["model"]

        engine.llm = LLMClient(backend=cfg["backend"])
        return f"已应用: {name}"
    except Exception as e:
        return f"应用失败: {e}"


def build_index():
    docs = load_all_data()
    if not docs:
        return "No data loaded."
    engine.build_index(docs)
    stats = engine.get_stats()
    return f"Index built: {stats['total']} documents"


def chat(message, history, source_filter):
    try:
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
    except Exception as e:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"出错了: {e}"},
        ]
        return history, "", ""


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
            with gr.Accordion("模型配置", open=True):
                saved_config_dropdown = gr.Dropdown(
                    choices=get_config_names(),
                    label="已保存的配置",
                    interactive=True,
                )
                with gr.Row():
                    apply_saved_btn = gr.Button("应用", variant="primary", size="sm")
                    add_btn = gr.Button("添加配置", size="sm")
                    delete_btn = gr.Button("删除", variant="stop", size="sm")
                config_status = gr.Markdown(value="")

                with gr.Group(visible=False) as add_form:
                    gr.Markdown("**新配置**")
                    cfg_name = gr.Textbox(label="配置名称", placeholder="如: 硅基流动-Qwen7B")
                    cfg_backend = gr.Dropdown(choices=["ollama", "api"], value="ollama", label="后端")
                    cfg_ollama_model = gr.Textbox(label="Ollama 模型", value="qwen3.5:4b")
                    cfg_provider = gr.Dropdown(choices=PROVIDER_CHOICES, value="siliconflow", label="API 提供商", visible=False)
                    cfg_api_key = gr.Textbox(label="API Key", type="password", visible=False)
                    cfg_custom_url = gr.Textbox(label="Base URL（可选，覆盖默认）", placeholder="https://your-api.com/v1", visible=False)
                    cfg_custom_model = gr.Textbox(label="模型名称（可选，覆盖默认）", placeholder="model-name", visible=False)
                    with gr.Row():
                        save_btn = gr.Button("保存", variant="secondary", size="sm")
                        cancel_btn = gr.Button("取消", size="sm")

            with gr.Accordion("索引管理", open=False):
                index_status = gr.Markdown(value="")
                with gr.Row():
                    build_btn = gr.Button("构建索引")
                    stats_btn = gr.Button("查看索引状态")

            with gr.Accordion("检索来源", open=True):
                source_display = gr.Markdown(value="暂无检索结果")

    saved_config_dropdown.change(
        select_config,
        inputs=[saved_config_dropdown],
        outputs=[cfg_backend, cfg_ollama_model, cfg_provider, cfg_api_key, cfg_custom_url, cfg_custom_model, config_status],
    )
    add_btn.click(show_add_form, outputs=[add_form])
    cancel_btn.click(
        hide_add_form,
        outputs=[add_form, config_status, cfg_backend, cfg_ollama_model, cfg_provider, cfg_api_key, cfg_custom_url, cfg_custom_model],
    )
    apply_saved_btn.click(apply_config, inputs=[saved_config_dropdown], outputs=[config_status])
    delete_btn.click(delete_config, inputs=[saved_config_dropdown], outputs=[config_status, saved_config_dropdown])
    save_btn.click(
        add_config,
        inputs=[cfg_name, cfg_backend, cfg_provider, cfg_api_key, cfg_ollama_model, cfg_custom_url, cfg_custom_model],
        outputs=[config_status, saved_config_dropdown, add_form],
    )
    cfg_backend.change(
        update_form_visibility,
        inputs=[cfg_backend, cfg_provider],
        outputs=[cfg_provider, cfg_api_key, cfg_ollama_model, cfg_custom_url, cfg_custom_model],
    )
    cfg_provider.change(
        lambda p: (gr.update(), gr.update()),
        inputs=[cfg_provider],
        outputs=[cfg_custom_url, cfg_custom_model],
    )
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
    demo.launch(server_name="0.0.0.0", server_port=7862, theme=gr.themes.Soft())
