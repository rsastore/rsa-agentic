import subprocess, json, os as os_mod
from pathlib import Path
from tools.git_tools import _git_status, _git_diff, _git_commit, _git_log
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



# ── GitHub Tools ──

def _github_issue(repo: str, title: str, body: str = ""):
    """Create a GitHub issue."""
    import subprocess, os
    try:
        r = subprocess.run(["gh","issue","create","--repo",repo,"--title",title,"--body",body],
                          capture_output=True, text=True, timeout=30)
        if r.returncode == 0: return f"Issue created: {r.stdout.strip()}"
        return f"Error: {r.stderr[:200]}"
    except FileNotFoundError:
        return "GitHub CLI (gh) not installed. Install: apt install gh && gh auth login"
    except Exception as e:
        return f"Error: {e}"

def _github_pr(repo: str, title: str, body: str = "", base: str = "main", head: str = ""):
    """Create a GitHub PR."""
    import subprocess
    try:
        cmd = ["gh","pr","create","--repo",repo,"--title",title,"--body",body,"--base",base]
        if head: cmd += ["--head", head]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0: return f"PR created: {r.stdout.strip()}"
        return f"Error: {r.stderr[:200]}"
    except FileNotFoundError:
        return "GitHub CLI (gh) not installed."
    except Exception as e:
        return f"Error: {e}"

def _github_search(query: str):
    """Search GitHub repos."""
    import subprocess
    try:
        r = subprocess.run(["gh","search","repos",query,"--limit","5"], capture_output=True, text=True, timeout=15)
        return r.stdout[:2000] if r.stdout else r.stderr[:200]
    except Exception: return "GitHub CLI not installed."


# ── Fine-tuning Tool ──

def _fine_tune(dataset: str, model_name: str = "qwen2.5:1.5b", output: str = ""):
    "Generate a fine-tuning script using Unsloth."
    import os
    output = output or os.path.expanduser("~/rsa-agentic/models/ft")
    script = (
        "import json, os, sys\n"
        "try:\n"
        "    from unsloth import FastLanguageModel\n"
        "    import torch\n"
        "    from datasets import Dataset\n"
        "    from trl import SFTTrainer\n"
        "    from transformers import TrainingArguments\n"
        f"    data_path = \"{dataset}\"\n"
        "    with open(data_path) as f:\n"
        "        if data_path.endswith('.jsonl'):\n"
        "            data = [json.loads(l) for l in f]\n"
        "        else: data = json.load(f)\n"
        f"    model, tokenizer = FastLanguageModel.from_pretrained(\"{model_name}\", max_seq_length=2048, load_in_4bit=True)\n"
        "    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=['q_proj','k_proj','v_proj','o_proj'], lora_dropout=0, bias='none')\n"
        "    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=Dataset.from_list(data),\n"
        f"        args=TrainingArguments(output_dir=\"{output}\", per_device_train_batch_size=2, num_train_epochs=1, logging_steps=10, save_steps=100, learning_rate=2e-4))\n"
        "    trainer.train()\n"
        "    model.save_pretrained(output_dir)\n"
        "    print('Fine-tuning complete!')\n"
        "except ImportError:\n"
        "    print('Install: pip install unsloth transformers datasets trl')\n"
    )
    ft_path = os.path.expanduser("~/rsa-agentic/ft_script.py")
    with open(ft_path, 'w') as f:
        f.write(script)
    return f"Fine-tuning script: {ft_path}\nRun: python3 {ft_path} (requires GPU)"
SAFE_COMMANDS = ["ls", "cat", "grep", "df", "du", "ps", "top", "free", "who",
    "date", "echo", "pwd", "which", "head", "tail", "wc", "sort",
    "uname", "id", "uptime", "dmesg", "journalctl", "systemctl",
    "docker", "curl", "wget", "ping", "netstat", "ss", "ip", "python3", "git"]

DESTRUCTIVE_PATTERNS = ["rm -rf", "rm -fr", "mkfs", "dd if=", "> /dev/sd", "chmod -R 777", 
                     "wget -O /", "curl -o /", "mv /* ", ":(){ :|:& };:", "git push --force"]

def _exec_shell(cmd: str, cwd: str = "."):
    # Safety check
    cmd_lower = cmd.lower()
    # Check approved command list from config
    try:
        cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
        if os.path.exists(cfg_path):
            cfg = tomllib.load(open(cfg_path, "rb"))
            approved = cfg.get("tools", {}).get("approved", [])
            if approved:
                cmd_name = cmd_lower.split()[0] if cmd_lower.split() else ""
                if cmd_name not in approved and cmd_name not in ["/bin/" + c for c in approved]:
                    return f"[BLOCKED] Command '{cmd_name}' not in approved list. Allowed: {approved}"
    except Exception:
        pass
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern in cmd_lower:
            return f"[BLOCKED] Dangerous command detected: {pattern}. Use sandbox_exec or get explicit approval."
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60)
    out = r.stdout
    if r.returncode != 0:
        out += "\n[Exit: {}]".format(r.returncode)
        if r.stderr:
            out += "\nSTDERR: {}".format(r.stderr[:300])
    return out.strip()

PROJECT_ROOT = Path(__file__).parent.parent  # ~/rsa-agentic/

def _read_file(path: str):
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    # Only try PROJECT_ROOT fallback for RELATIVE paths
    if not path.startswith("/"):
        p2 = PROJECT_ROOT / path
        if p2.exists():
            return p2.read_text(encoding="utf-8", errors="replace")
        return f"File not found: {path} (tried: {p.resolve()} and {p2})"
    return f"File not found: {path}"

SYSTEM_PATHS = ["/etc/", "/boot/", "/usr/", "/bin/", "/sbin/", "/lib/", "/var/log/"]

def _write_file(path: str, content: str):
    # Safety: block overwriting system files
    resolved = str(Path(path).resolve())
    for sp in SYSTEM_PATHS:
        if resolved.startswith(sp):
            return f"[BLOCKED] Refusing to write to system path: {sp}"
    Path(path).write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"

def _list_dir(path: str = "."):
    p = Path(path)
    if p.exists():
        items = os_mod.listdir(path)
    elif not path.startswith("/"):
        p2 = PROJECT_ROOT / path
        if p2.exists():
            items = os_mod.listdir(str(p2))
        else:
            return f"Directory not found: {path} (tried: {p.resolve()} and {p2})"
    else:
        return f"Directory not found: {path}"
    lines = []
    for name in sorted(items):
        full = os_mod.path.join(path, name)
        kind = "d" if os_mod.path.isdir(full) else "f"
        try:
            size = os_mod.path.getsize(full)
        except Exception:
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


def _sandbox_exec(cmd: str, image: str = "neural-sandbox:latest"):
    """Execute a command inside a Docker sandbox container.
    Gracefully falls back to exec_shell with warning if Docker is not available.
    """
    import subprocess
    try:
        subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ("[WARNING] Docker not available. Falling back to exec_shell (unsafe).\n"
                "To fix: install Docker or use exec_shell directly.\n"
                "Run without sandbox: exec_shell(\"" + cmd + "\")")
    except Exception:
        return "Docker error. Use exec_shell directly."
    
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "-i", "--network", "none",
             "--memory", "512m", "--cpus", "1", image,
             "sh", "-c", cmd],
            capture_output=True, text=True, timeout=60,
        )
        out = r.stdout
        if r.returncode != 0:
            out += f"\n[Exit: {r.returncode}]"
            if r.stderr:
                out += f"\nSTDERR: {r.stderr[:500]}"
        return out
    except subprocess.TimeoutExpired:
        return "Error: Command timed out in sandbox."
    except Exception as e:
        return f"Sandbox error: {e}"


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



def _notify(title: str, body: str = ""):
    """Send desktop notification."""
    try:
        print(f"\033]9;{title};{body}\033\\", end="", flush=True)
        return f"Notification sent: {title}"
    except Exception:
        pass
    # Fallback: print to terminal
    return f"[NOTIFICATION] {title}: {body}"


BUILTIN_TOOLS = [
    Tool("edit_file", _edit_file,
         "Edit a file with search/replace. Always call with mode='preview' first, then mode='apply'.",
         {"path": "file path", "search": "text to find", "replace": "replacement text",
          "mode": "'preview' (default) or 'apply'"}),

    Tool("web_fetch", _fetch_url,
         "Fetch a URL and extract readable text.",
         {"url": "URL to fetch", "max_chars": "max chars to return (optional)"}),

    Tool("sandbox_exec", _sandbox_exec,
         "Execute a command in Docker sandbox (safe, isolated).",
         {"cmd": "command to run", "image": "Docker image (optional)"}),

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

    Tool("notify", _notify,
         "Send a desktop notification.",
         {"title": "title", "body": "body (optional)"}),

    Tool("git_status", _git_status,
         "Show git working tree status.",
         {"path": "repo path (default .)"}),

    Tool("git_diff", _git_diff,
         "Show staged/unstaged git changes.",
         {"path": "repo path (default .)"}),

    Tool("git_commit", _git_commit,
         "Stage all files and commit.",
         {"message": "commit message", "path": "repo path (default .)"}),

    Tool("github_issue", _github_issue,
         "Create a GitHub issue (requires gh CLI installed).",
         {"repo": "owner/repo", "title": "issue title", "body": "description (optional)"}),

    Tool("github_pr", _github_pr,
         "Create a GitHub pull request (requires gh CLI).",
         {"repo": "owner/repo", "title": "PR title", "body": "description", "base": "base branch", "head": "head branch (optional)"}),

    Tool("github_search", _github_search,
         "Search GitHub repos.",
         {"query": "search query"}),

    Tool("fine_tune", _fine_tune,
         "Generate a fine-tuning script for a model.",
         {"dataset": "path to dataset (JSONL or JSON)", "model_name": "base model name", "output": "output dir (optional)"}),

    Tool("git_log", _git_log,
         "Show recent git commits.",
         {"max_count": "commits (default 5)", "path": "repo path (default .)"}),
]

TOOL_MAP = {t.name: t for t in BUILTIN_TOOLS}
_PLUGIN_TOOLS_REGISTERED = False

def _ensure_plugins():
    global _PLUGIN_TOOLS_REGISTERED
    if _PLUGIN_TOOLS_REGISTERED:
        return
    _PLUGIN_TOOLS_REGISTERED = True
    try:
        from plugin_loader import discover
        discover()
    except Exception:
        pass

def get_tool(name: str) -> Tool | None:
    t = TOOL_MAP.get(name)
    if t:
        return t
    _ensure_plugins()
    try:
        from plugin_loader import get_tool as pg
        pt = pg(name)
        if pt:
            return Tool(pt.name, pt.fn, pt.desc, pt.params)
    except Exception:
        pass
    return None

def list_tools() -> list[str]:
    builtin = list(TOOL_MAP.keys())
    _ensure_plugins()
    try:
        from plugin_loader import list_tools as pl
        builtin.extend(pl())
    except Exception:
        pass
    return builtin

def tool_descriptions() -> str:
    lines = []
    lines.append("## Available Tools")
    lines.append('Format: {"tool": "name", "args": {"param": "val"}}')
    lines.append("")
    for t in BUILTIN_TOOLS:
        params = ", ".join(f"{k}: {v}" for k, v in t.params.items())
        lines.append(f"- {t.name}({params}): {t.description}")
    _ensure_plugins()
    try:
        from plugin_loader import descriptions as pd
        pds = pd()
        if pds:
            lines.append("")
            lines.append("### Plugin Tools")
            lines.append(pds)
    except Exception:
        pass
    return "\n".join(lines)

def plugin_tool_count() -> int:
    try:
        from plugin_loader import list_tools as pl
        return len(pl())
    except Exception:
        return 0
