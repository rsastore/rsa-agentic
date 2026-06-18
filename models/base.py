
import abc
from typing import Any

class ModelProvider(abc.ABC):
    """Abstract LLM provider interface."""

    @abc.abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send chat messages, return response text."""
        ...

    @abc.abstractmethod
    def chat_stream(self, messages: list[dict], **kwargs):
        """Yield tokens as they arrive."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...


class MessageBuilder:
    """Build messages with Qwen2.5 chat template."""

    @staticmethod
    def build(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
