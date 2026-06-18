import json, re, os as os_mod
from pathlib import Path
from typing import Callable

from tools.builtin import get_tool, tool_descriptions, list_tools, BUILTIN_TOOLS

# ── System Prompt Builder ─────────────────────────────────────

def build_system_prompt(custom_prompt: str | None = None) -> str:
    tools_desc = tool_descriptions()

    base = f"""You are Neural, an autonomous AI agent.

You have access to these tools:

{tools_desc}

## How to Call a Tool
Respond with a JSON block:
```json
{{"tool": "tool_name", "args": {{"param": "value"}}}}
```

Then wait for the result and continue your reasoning.
When the task is complete, respond with a natural language answer.

## Rules
1. Plan your approach step by step.
2. Call one tool at a time. Read the output before deciding next step.
3. You are running on a Linux server with full shell access.
4. For system info: exec_shell with "uname -a", "df -h", etc.
5. Verify your results before reporting.
"""
    if custom_prompt:
        base += f"\n\n## Additional Instructions\n{custom_prompt}"
    return base


# ── Agent Session ─────────────────────────────────────────────

class AgentSession:
    """Maintains conversation state and runs agent loop."""

    def __init__(self, provider, config: dict):
        self.provider = provider
        self.config = config
        self.max_iters = config.get("max_tool_iters", 15)
        self.session_id = os_mod.urandom(4).hex()
        self._messages: list[dict] = []
        self.tool_callbacks: list[Callable] = []

    @property
    def messages(self) -> list[dict]:
        return self._messages

    def reset(self):
        self._messages = []
        self.session_id = os_mod.urandom(4).hex()

    def _extract_tool_call(self, text: str) -> dict | None:
        """Parse tool call JSON from model response. Handles nested braces."""
        # Try ```json ... ``` block first
        m = re.search(r'```(?:json)?\s*\n(\{.*?\})\n\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Try brace-counting approach
        idx = text.find('{"tool"')
        if idx == -1:
            idx = text.find('{"tool"')
        if idx == -1:
            return None

        depth = 0
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[idx:i+1])
                        if "tool" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    return None
        return None

    def run(self, user_input: str) -> str:
        """Process user input through agent loop. Returns final answer."""
        # Load system prompt
        sys_file = self.config.get("system_prompt_file", "system.md")
        sys_path = Path(os_mod.path.expanduser("~/neural")) / sys_file
        custom = sys_path.read_text() if sys_path.exists() else None
        sys_prompt = build_system_prompt(custom)

        # Initialize messages if first turn
        if not self._messages:
            self._messages.append({"role": "system", "content": sys_prompt})

        self._messages.append({"role": "user", "content": user_input})

        for step in range(self.max_iters):
            # Get model response
            raw = self.provider.chat(self._messages)

            # Check for tool call
            call = self._extract_tool_call(raw)

            if call is None:
                # No tool call — this is the final answer
                self._messages.append({"role": "assistant", "content": raw})
                return raw

            # Execute tool
            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool = get_tool(tool_name)
            if tool is None:
                output = f"Error: Unknown tool '{tool_name}'. Available: {', '.join(list_tools())}"
            else:
                output = tool(**tool_args)

            # Notify callbacks (for TUI to show tool calls)
            for cb in self.tool_callbacks:
                cb(tool_name, tool_args, output)

            # Add to conversation
            self._messages.append({"role": "assistant", "content": raw})
            self._messages.append({
                "role": "tool",
                "content": f"[{tool_name}] Result:\n{output[:3000]}",
            })

        self._messages.append({
            "role": "assistant",
            "content": "Max iterations reached. Task may be incomplete.",
        })
        return "Max iterations reached."


class SubAgent:
    """A lightweight sub-agent for parallel task execution."""

    def __init__(self, provider, config: dict, task: str, parent_messages: list[dict]):
        self.provider = provider
        self.config = config
        self.task = task
        self.parent_messages = parent_messages

    def run(self) -> dict:
        """Execute sub-task and return result."""
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": self.task},
        ]
        try:
            result = ""
            for _ in range(8):
                raw = self.provider.chat(messages)
                call = self._extract_tool_call(raw)
                if not call:
                    result = raw
                    break
                tool = get_tool(call.get("tool", ""))
                if tool:
                    args = call.get("args", {})
                    if not isinstance(args, dict):
                        args = {}
                    output = tool(**args)
                else:
                    output = "Unknown tool"
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "tool", "content": output[:2000]})
            return {"task": self.task, "result": result or "No result"}
        except Exception as e:
            return {"task": self.task, "result": f"Error: {e}"}

    @staticmethod
    def _extract_tool_call(text: str) -> dict | None:
        # Try ```json ... ``` block first
        m = re.search(r'```(?:json)?\s*\n(\{.*?\})\n\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try brace-counting approach
        idx = text.find('{"tool"')
        if idx == -1:
            idx = text.find('{"tool"')
        if idx == -1:
            return None
        depth = 0
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[idx:i+1])
                        if "tool" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    return None
        return None
