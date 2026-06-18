import json, requests
from models.base import ModelProvider

class OllamaProvider(ModelProvider):
    def __init__(self, config: dict):
        self.host = config.get("host", "http://localhost:11434")
        self.model = config.get("model_name", "qwen2.5:1.5b")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 4096)

    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"

    def _call(self, messages: list[dict], stream: bool = False):
        url = f"{self.host}/api/chat"
        body = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": stream,
        }
        return requests.post(url, json=body, timeout=120)

    def chat(self, messages: list[dict], **kwargs) -> str:
        r = self._call(messages, stream=False)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    def chat_stream(self, messages: list[dict], **kwargs):
        r = self._call(messages, stream=True)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done", False):
                    break


class OpenAIProvider(ModelProvider):
    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.3)

    @property
    def name(self) -> str:
        return f"OpenAI/{self.model}"

    def chat(self, messages: list[dict], **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        r = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return r.choices[0].message.content or ""

    def chat_stream(self, messages: list[dict], **kwargs):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        r = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in r:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta


def create_provider(config: dict) -> ModelProvider:
    prov_name = config.get("provider", "ollama")
    prov_cfg = config.get(prov_name, {})
    m = config.get("model_name", "")
    prov_cfg["model_name"] = m
    prov_cfg["temperature"] = config.get("temperature", 0.3)
    prov_cfg["max_tokens"] = config.get("max_tokens", 4096)

    if prov_name == "ollama":
        return OllamaProvider(prov_cfg)
    elif prov_name == "openai":
        return OpenAIProvider(prov_cfg)
    else:
        raise ValueError(f"Unknown provider: {prov_name}")
