from openai import OpenAI
from config import LLM_BACKEND, LLM_MODEL


class LLMClient:
    def __init__(self, backend: str = LLM_BACKEND):
        self.backend = backend
        self._client = None

        if backend == "ollama":
            self._init_ollama()
        elif backend == "api":
            self._init_api()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_ollama(self):
        self._client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        print(f"Ollama LLM ready: {LLM_MODEL}")

    def _init_api(self):
        from config import API_BASE_URL, API_KEY
        self._client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"API LLM ready: {LLM_MODEL}")

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        response = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.7,
            extra_body={"think": False, "options": {"num_predict": max_new_tokens}},
        )
        content = response.choices[0].message.content or ""
        return content.strip()
