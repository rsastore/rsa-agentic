"""Session persistence for Neural agent."""

import json, os as os_mod
from pathlib import Path
from datetime import datetime
from typing import Optional

SESSION_DIR = Path(os_mod.path.expanduser("~/neural/sessions"))


def save_session(session_id: str, messages: list[dict],
                 tool_history: list[tuple], model: str) -> str:
    """Save session to disk. Returns file path."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SESSION_DIR / f"{ts}_{session_id}.json"

    data = {
        "session_id": session_id,
        "timestamp": ts,
        "model": model,
        "message_count": len(messages),
        "tool_calls": [
            {"tool": t[0], "args": t[1], "result_preview": str(t[2])[:200]}
            for t in tool_history
        ],
        "messages": messages,
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


def load_session(path: str) -> Optional[dict]:
    """Load a saved session."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return None


def list_sessions(limit: int = 10) -> list[Path]:
    """List recent sessions."""
    files = sorted(SESSION_DIR.glob("*.json"), reverse=True)
    return files[:limit]


def export_to_markdown(session_path: str) -> str:
    """Export session as markdown transcript."""
    data = load_session(session_path)
    if not data:
        return "Session not found."

    lines = [
        f"# Neural Session — {data['timestamp']}",
        f"**Model:** {data['model']}",
        f"**Messages:** {data['message_count']}",
        "",
    ]

    for msg in data.get("messages", []):
        role = msg["role"].upper()
        content = msg["content"]
        if role == "SYSTEM":
            continue
        lines.append(f"## {role}")
        lines.append("")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)
