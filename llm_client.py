import os
from openai import OpenAI
from config import LLM_BACKEND, LLM_MODEL, API_PROVIDER, API_KEY, API_PROVIDERS


class LLMClient:
    def __init__(self, backend: str = LLM_BACKEND):
        self.backend = backend
        self._client = None

        if backend == "ollama":
            self._init_ollama()
        elif backend == "api":
            self._init_api()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'api'.")

    def _init_ollama(self):
        self._client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        print(f"Ollama LLM ready: {LLM_MODEL}")

    def _init_api(self):
        api_key = os.environ.get("LLM_API_KEY") or API_KEY
        if not api_key:
            raise ValueError(
                "API key not found. Set LLM_API_KEY environment variable or fill API_KEY in config.py"
            )

        provider = API_PROVIDERS.get(API_PROVIDER)
        if not provider:
            raise ValueError(f"Unknown API provider: {API_PROVIDER}")

        base_url = provider["base_url"]
        model = provider["model"]

        if not base_url:
            raise ValueError(f"Base URL not configured for provider: {API_PROVIDER}")

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        print(f"API LLM ready: {API_PROVIDER} / {model}")

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        model = LLM_MODEL if self.backend == "ollama" else self._model

        extra_body = {}
        if self.backend == "ollama":
            extra_body = {"think": False, "options": {"num_predict": max_new_tokens}}

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.7,
            extra_body=extra_body if extra_body else None,
        )
        content = response.choices[0].message.content or ""
        return content.strip()
