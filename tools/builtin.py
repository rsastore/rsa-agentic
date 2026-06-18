import subprocess, json, os as os_mod
from pathlib import Path
from typing import Any

# ── Tool Registry ──────────────────────────────────────────────

class Tool:
    def __init__(self, name: str, fn, description: str, params: dict):
        self.name = name
        self.fn = fn
        self.description = description
        self.params = params

    def __call__(self, **kwargs) -> str:
        try:
            result = self.fn(**kwargs)
            if hasattr(result, "stdout"):
                out = result.stdout
                if result.returncode != 0:
                    out += f"\nSTDERR: {result.stderr}"
                return out
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.params,
        }


# ── Built-in Tools ────────────────────────────────────────────

def _exec_shell(cmd: str, cwd: str = "."):
    return subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=60,
    )

def _read_file(path: str):
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    return p.read_text(encoding="utf-8", errors="replace")

def _write_file(path: str, content: str):
    Path(path).write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"

def _list_dir(path: str = "."):
    items = os_mod.listdir(path)
    lines = []
    for name in sorted(items):
        full = os_mod.path.join(path, name)
        kind = "d" if os_mod.path.isdir(full) else "f"
        try:
            size = os_mod.path.getsize(full)
        except:
            size = 0
        lines.append(f"{kind} {size:>8} {name}")
    return "\n".join(lines) if lines else "(empty)"

def _grep_files(pattern: str, path: str = ".", include: str = ""):
    cmd = ["grep", "-rn", "--color=never", pattern, path]
    if include:
        cmd.extend(["--include", include])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout
    return "No matches found"

def _python_exec(code: str):
    r = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    out = r.stdout
    if r.returncode != 0:
        out += f"\nError: {r.stderr}"
    return out



def _fetch_url(url: str, max_chars: int = 5000):
    """Fetch a URL and return text content."""
    import urllib.request, re
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')
            # Strip HTML tags
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
            return text
    except Exception as e:
        return f"Error fetching URL: {e}"


def _edit_file(path: str, search: str, replace: str, mode: str = "preview"):
    """Edit a file with search/replace. Shows diff in preview mode."""
    import difflib
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    old_content = p.read_text(encoding="utf-8", errors="replace")
    idx = old_content.find(search)
    if idx == -1:
        return f"Error: search text not found in {path}"
    new_content = old_content[:idx] + replace + old_content[idx + len(search):]
    diff = list(difflib.unified_diff(
        old_content.splitlines(True), new_content.splitlines(True),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
    ))
    diff_text = "".join(diff[:40])  # Max 40 lines of diff
    if mode == "preview":
        return f"--- Diff for {path} ---\n{diff_text}\n--- End diff ({len(diff)} lines) ---\nCall with mode='apply' to apply."
    elif mode == "apply":
        p.write_text(new_content, encoding="utf-8")
        return f"Applied changes to {path} ({len(diff)} lines changed).\n{diff_text}"
    return f"Unknown mode: {mode}. Use 'preview' or 'apply'."


BUILTIN_TOOLS = [
    Tool("edit_file", _edit_file,
         "Edit a file with search/replace. Always call with mode='preview' first, then mode='apply'.",
         {"path": "file path", "search": "text to find", "replace": "replacement text",
          "mode": "'preview' (default) or 'apply'"}),

    Tool("web_fetch", _fetch_url,
         "Fetch a URL and extract readable text.",
         {"url": "URL to fetch", "max_chars": "max chars to return (optional)"}),

    Tool("exec_shell", _exec_shell,
         "Execute a shell command on the system.",
         {"cmd": "command to run", "cwd": "working directory (optional)"}),

    Tool("read_file", _read_file,
         "Read the contents of a file.",
         {"path": "path to file"}),

    Tool("write_file", _write_file,
         "Write text content to a file.",
         {"path": "path to file", "content": "content to write"}),

    Tool("list_dir", _list_dir,
         "List files and directories.",
         {"path": "directory path (optional, default .)"}),

    Tool("grep_files", _grep_files,
         "Search for a pattern in files.",
         {"pattern": "regex pattern", "path": "search path", "include": "file glob filter (optional)"}),

    Tool("python_exec", _python_exec,
         "Execute Python code and return output.",
         {"code": "Python code to run"}),
]

TOOL_MAP = {t.name: t for t in BUILTIN_TOOLS}


def get_tool(name: str) -> Tool | None:
    return TOOL_MAP.get(name)

def tool_descriptions() -> str:
    lines = []
    for t in BUILTIN_TOOLS:
        params = ", ".join(f"{k}: {v}" for k, v in t.params.items())
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines)

def list_tools() -> list[str]:
    return list(TOOL_MAP.keys())
