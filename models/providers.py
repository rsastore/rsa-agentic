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
        self.last_tokens = {"input": data.get("prompt_eval_count",0), "output": data.get("eval_count",0)}
        return data.get("message", {}).get("content", "")

    def chat_stream(self, messages: list[dict], **kwargs):
        r = self._call(messages, stream=True)
        r.raise_for_status()
        self.last_tokens = {"input": 0, "output": 0}
        for line in r.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done", False):
                    self.last_tokens = {"input": data.get("prompt_eval_count",0), "output": data.get("eval_count",0)}
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
        self.last_tokens = {"input": r.usage.prompt_tokens if r.usage else 0, "output": r.usage.completion_tokens if r.usage else 0}
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
        self.last_tokens = {"input": 0, "output": 0}
        for chunk in r:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
            if chunk.usage:
                self.last_tokens = {"input": chunk.usage.prompt_tokens, "output": chunk.usage.completion_tokens}


def create_provider(config: dict) -> ModelProvider:
    prov_name = config.get("provider", "ollama")
    prov_cfg = config.get(prov_name, {})
    m = config.get("model_name", "")
    prov_cfg["model_name"] = m
    prov_cfg["model"] = prov_cfg.get("model", config.get("model_name") or "gpt-4o")
    prov_cfg["temperature"] = config.get("temperature", 0.3)
    prov_cfg["max_tokens"] = config.get("max_tokens", 4096)

    if prov_name == "ollama":
        return OllamaProvider(prov_cfg)
    elif prov_name == "openai":
        return OpenAIProvider(prov_cfg)
    elif prov_name == "anthropic":
        return AnthropicProvider(prov_cfg)
    elif prov_name == "google":
        return GoogleProvider(prov_cfg)
    else:
        # Unknown provider: try OpenAI-compatible (covers OpenRouter, DeepSeek, Groq, xAI, Together, etc.)
        prov_cfg["model"] = prov_cfg.get("model", config.get("model_name") or "gpt-4o")
        if prov_cfg.get("api_key") or prov_cfg.get("base_url"):
            return OpenAIProvider(prov_cfg)
        raise ValueError(f"Unknown provider: {prov_name}. Use /provider add <name> <base_url> <key> to register.")


class AnthropicProvider(ModelProvider):
    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 4096)

    @property
    def name(self) -> str:
        return f"Anthropic/{self.model}"

    def chat(self, messages, **kw) -> str:
        from anthropic import Anthropic
        cl = Anthropic(api_key=self.api_key)
        r = cl.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[m for m in messages if m["role"] != "system"],
            system="\n".join(m["content"] for m in messages if m["role"] == "system"),
        )
        return r.content[0].text if r.content else ""

    def chat_stream(self, messages, **kw):
        from anthropic import Anthropic
        cl = Anthropic(api_key=self.api_key)
        with cl.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[m for m in messages if m["role"] != "system"],
            system="\n".join(m["content"] for m in messages if m["role"] == "system"),
        ) as s:
            for text in s.text_stream:
                yield text


class GoogleProvider(ModelProvider):
    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.3)

    @property
    def name(self) -> str:
        return f"Google/{self.model}"

    def chat(self, messages, **kw) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        # Convert messages
        history = []
        for m in messages[:-1]:
            if m["role"] in ("user", "assistant"):
                history.append({"role": m["role"], "parts": [m["content"]]})
        prompt = messages[-1]["content"] if messages else ""
        chat = model.start_chat(history=history) if history else model
        r = chat.send_message(prompt)
        return r.text

    def chat_stream(self, messages, **kw):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        history = []
        for m in messages[:-1]:
            if m["role"] in ("user", "assistant"):
                history.append({"role": m["role"], "parts": [m["content"]]})
        prompt = messages[-1]["content"] if messages else ""
        chat = model.start_chat(history=history) if history else model
        r = chat.send_message(prompt, stream=True)
        for chunk in r:
            if chunk.text:
                yield chunk.text
