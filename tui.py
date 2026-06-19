import sys, os as os_mod, json
from pathlib import Path
from datetime import datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich import box

from agent import AgentSession, SubAgent
from planner import PlannerAgent, Plan
from compact import compact_messages
from models.providers import create_provider
from tools.builtin import list_tools

# ── Console ────────────────────────────────────────────────────
console = Console()

# ── Brand ──────────────────────────────────────────────────────
BRAND = """
╔══════════════════════════════════════════╗
║       RSA Agentic v0.1              ║
║     Autonomous AI Agent — Local LLM      ║
╚══════════════════════════════════════════╝
"""

# ── Config Loader ──────────────────────────────────────────────
def load_config():
    import tomllib
    cfg_path = Path(os_mod.path.expanduser("~/rsa-agentic/config.toml"))
    if not cfg_path.exists():
        console.print("[red]Config not found. Run setup first.[/red]")
        return None
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)

def _check_ollama_status(host="http://localhost:11434", model=""):
    """Check if Ollama is running and model is available."""
    import requests
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            installed = [m["name"] for m in models]
            if model and model not in installed:
                return f"⚠️ Ollama OK, model '{model}' not pulled", "yellow"
            return f"✅ Ollama OK ({len(installed)} models)", "green"
        return f"⚠️ Ollama error: {r.status_code}", "yellow"
    except Exception:
        return "❌ Ollama not running (/install ollama)", "red"

def _write_toml(cfg, path):
    """Write TOML config without tomli_w dependency."""
    lines = []
    for section, values in cfg.items():
        lines.append(f"[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, dict):
                    lines.append(f"[{section}.{k}]")
                    for sk, sv in v.items():
                        if isinstance(sv, bool):
                            lines.append(f"{sk} = {'true' if sv else 'false'}")
                        elif isinstance(sv, (int, float)):
                            lines.append(f"{sk} = {sv}")
                        else:
                            lines.append(f'{sk} = "{sv}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f'{k} = "{v}"')
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))

def bootstrap_display(provider_name: str, model_name: str, tool_count: int):
    grid = Table.grid(padding=1)
    grid.add_column(style="cyan bold")
    grid.add_column()
    import requests as _req
    try:
        _r = _req.get("http://localhost:11434/api/tags", timeout=2)
        if _r.status_code == 200:
            models = _r.json().get("models", [])
            if models:
                # Show actual running model
                # If config model exists, use it; otherwise use first available
                cfg_model = model_name
                installed = [m["name"] for m in models]
                if cfg_model in installed:
                    actual = cfg_model
                else:
                    actual = installed[0]  # fallback to first available
                grid.add_row("● Model", actual)
                grid.add_row("● Provider", f"Ollama/{actual}")
            else:
                grid.add_row("● Model", "[dim]-[/dim]")
                grid.add_row("● Provider", "[dim]Ollama — no models pulled[/dim]")
        else:
            grid.add_row("● Model", "[dim]-[/dim]")
            grid.add_row("● Provider", "[dim]-[/dim]")
        status, color = "✅ Ollama running", "green"
    except Exception:
        grid.add_row("● Model", "[dim]-[/dim]")
        grid.add_row("● Provider", "[dim]-[/dim]")
        status, color = "❌ Ollama not running", "red"
    grid.add_row("● Tools", str(tool_count))
    grid.add_row("● Status", f"[{color}]{status}[/{color}]" )
    console.print(Panel(grid, title="[bold cyan]RSA Agentic[/bold cyan]", border_style="cyan"))

# ── TUI Core ───────────────────────────────────────────────────
class NeuralTUI:
    def __init__(self, config: dict):
        self.config = config
        self.provider = create_provider(config.get("model", {}))
        self.session = AgentSession(self.provider, config.get("agent", {}))
        self.checklist: list[dict] = []
        self.tool_history: list[tuple] = []
        self.status = "idle"

        # Session save dir
        self.sessions_dir = Path(os_mod.path.expanduser("~/rsa-agentic/sessions"))
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Register tool callback
        self.session.tool_callbacks.append(self._on_tool_call)

    def _on_tool_call(self, tool_name: str, args: dict, output: str):
        self.tool_history.append((tool_name, args, output[:200]))

    def _show_splash(self):
        console.clear()
        console.print(BRAND, style="cyan")
        bootstrap_display(
            self.provider.name,
            self.config.get("model", {}).get("model_name", "?"),
            len(list_tools()),
        )

    def _print_chat_bubble(self, role: str, content: str):
        if role == "user":
            console.print()
            console.print(Panel(
                content,
                title="[bold green]You[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            ))
        elif role == "assistant":
            console.print()
            try:
                md = Markdown(content)
                console.print(Panel(
                    md,
                    title="[bold cyan]Neural[/bold cyan]",
                    border_style="cyan",
                    box=box.ROUNDED,
                ))
            except Exception:
                console.print(Panel(
                    content,
                    title="[bold cyan]Neural[/bold cyan]",
                    border_style="cyan",
                    box=box.ROUNDED,
                ))
        elif role == "tool":
            # Show tool call inline (compact)
            console.print(f"  [dim]🔧 {content[:80]}{'...' if len(content) > 80 else ''}[/dim]")

    def _show_tool_call(self, name: str, args: dict, output: str):
        args_str = json.dumps(args)[:60]
        console.print(f"  [bright_black]⚡ {name}({args_str})[/bright_black]")
        if output.strip() and len(output) < 300:
            console.print(f"  [bright_black]  └─ {output.strip()[:80]}[/bright_black]")
        else:
            console.print(f"  [bright_black]  └─ ({len(output)} chars output)[/bright_black]")

    def _save_session(self, user_input: str, final_output: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.sessions_dir / f"session_{timestamp}.json"
        data = {
            "timestamp": timestamp,
            "model": self.config.get("model", {}).get("model_name", ""),
            "provider": self.provider.name,
            "input": user_input,
            "output": final_output,
            "tool_calls": [
                {"tool": t[0], "args": t[1], "result_preview": t[2]}
                for t in self.tool_history
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return str(path)

    def run(self):
        self._show_splash()
        console.print("[dim]Type your request. /help for commands. Ctrl+D or /exit to quit.[/dim]\n")

        style = Style.from_dict({
            "prompt": "ansicyan bold",
            "status": "ansibrightblack",
        })

        session = PromptSession(
            history=FileHistory(str(Path.home() / ".neural_history")),
            style=style,
        )

        while True:
            try:
                user_input = session.prompt(
                    [("class:prompt", " \u2502 ")],
                    style=style,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self._handle_exit()
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # Streaming + approval loop
            buf = []
            tool_cancelled = False
            self.tool_history.clear()
            console.print()

            try:
                # Animated spinner with status updates
                import threading as _th, time as _t2
                _done = [False]
                _status = ["⏳ Thinking..."]
                def _spinner():
                    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                    i = 0
                    while not _done[0]:
                        frame = frames[i % len(frames)]
                        msg = _status[-1]
                        console.print(f"[dim]{frame} {msg}[/dim]", end="\r")
                        i += 1
                        _t2.sleep(0.12)
                _spinner_th = _th.Thread(target=_spinner, daemon=True)
                _spinner_th.start()
                for event in self.session.run_stream(user_input):
                    if event["type"] == "status":
                        _status.append(event["content"])
                    elif event["type"] == "token":
                        _done[0] = True
                        console.print("\r", end="")  # just move to start
                        if not buf:
                            console.print("[dim]⚡ [/dim]", end="")
                        # Typewriter effect: word by word with delay
                        text = event["content"]
                        import time as _t
                        # Word-by-word with adaptive delay
                        words = text.split(" ")
                        for i, w in enumerate(words):
                            console.print(w, end="")
                            if i < len(words) - 1:
                                console.print(" ", end="")
                                # No delay after punctuation (paragraph ends)
                                if w and w[-1] in "!?.:;\n":
                                    _t.sleep(0.05)  # almost instant
                                else:
                                    _t.sleep(0.08)  # 80ms normal
                        buf.append(text)
                    elif event["type"] == "tool_call":
                        console.print(f"\n  [bright_black]⚡ {event['tool']}(...)[/bright_black]")
                    elif event["type"] == "approval_needed":
                        args_str = str(event.get("args", {}))[:100]
                        console.print(f"\n  [bold yellow]⚠️  Allow {event['tool']}?[/bold yellow]")
                        console.print(f"  [dim]{args_str}[/dim]")
                        console.print("  [y/N] ", end="")
                        ans = input().strip().lower()
                        if ans == "y":
                            console.print("  [green]Approved.[/green]")
                            self.session._pending_approved = True
                        else:
                            tool_cancelled = True
                            console.print("  [yellow]Skipped.[/yellow]")
                            self.session._pending_approved = False
                    elif event["type"] == "tool_result":
                        r = event["content"][:200]
                        console.print(f"  [bright_black]  └─ {r}[/bright_black]")
                    elif event["type"] == "final":
                        output = event["content"]
                        console.print()
                        self._print_chat_bubble("assistant", output)
                        saved = self._save_session(user_input, output)
                        console.print(f"  [dim]💾 {saved}[/dim]")

            except Exception as e:
                console.print(f"\n  [bold red]Error:[/bold red] {e}")

    def _handle_command(self, cmd: str):
        cmd = cmd.strip().lower()  # case-insensitive
        import os, sys
        import os as _os, sys as _sys

        # Simplify: map short commands
        if cmd.startswith("/model ") or cmd == "/model":
            # /model = install model, /model list = show popular
            parts = cmd.split(None, 1)
            sub = parts[1].strip() if len(parts) > 1 else ""
            if not sub or sub == "list":
                # Show popular models
                popular = [
                    ("qwen2.5:1.5b", "1.1 GB", "Best for HP 6GB RAM"),
                    ("llama3.2:3b", "2.0 GB", "Best English tool calling"),
                    ("gemma2:2b", "1.5 GB", "Ringan"),
                    ("mistral:7b", "4.2 GB", "Best quality"),
                ]
                console.print("[bold cyan]Popular models:[/bold cyan]")
                for i, (n, s, d) in enumerate(popular, 1):
                    console.print(f"  {i}. [cyan]{n:<20}[/cyan] {s:<8} [dim]{d}[/dim]")
                console.print("[dim]Usage: /model <name> (e.g. /model llama3.2:3b)[/dim]")
                return
            else:
                # Download model
                import subprocess as _sp
                console.print(f"[cyan]Downloading {sub}...[/cyan]")
                _sp.run(["ollama", "pull", sub], timeout=1800)
                # Switch to downloaded model immediately
                import tomllib, os as _os
                cfg_path = _os.path.expanduser("~/rsa-agentic/config.toml")
                cfg = tomllib.load(open(cfg_path, "rb"))
                cfg["model"]["model_name"] = sub
                cfg["model"]["provider"] = "ollama"
                # Write TOML manually (avoids tomli_w dependency)
                lines = []
                for section, values in cfg.items():
                    lines.append(f"[{section}]")
                    if isinstance(values, dict):
                        for k, v in values.items():
                            if isinstance(v, dict):
                                lines.append(f"[{section}.{k}]")
                                for sk, sv in v.items():
                                    lines.append(f'{sk} = "{sv}"')
                            elif isinstance(v, bool):
                                lines.append(f"{k} = {'true' if v else 'false'}")
                            elif isinstance(v, (int, float)):
                                lines.append(f"{k} = {v}")
                            else:
                                lines.append(f'{k} = "{v}"')
                    lines.append("")
                with open(cfg_path, "w") as f:
                    f.write("\n".join(lines))
                from models.providers import create_provider
                self.provider = create_provider(cfg)
                self.config = cfg
                console.print(f"[green]✅ Now using: {self.provider.name}[/green]")
        elif cmd == "/engine":
            engines = [
                ("ollama", "Default, easiest"),
                ("llama.cpp", "Native C++, faster on CPU"),
                ("vllm", "Fastest on GPU"),
            ]
            console.print("[bold cyan]Inference engines:[/bold cyan]")
            for i, (n, d) in enumerate(engines, 1):
                console.print(f"  {i}. [cyan]{n:<15}[/cyan] {d}")
            console.print("[dim]Usage: /engine ollama | /engine llama.cpp | /engine vllm[/dim]")
        elif cmd.startswith("/engine "):
            eng = cmd[8:].strip()
            import subprocess as _sp, os as _os
            if eng == "ollama":
                console.print("[cyan]Installing Ollama...[/cyan]")
                if _os.path.exists("/data/data/com.termux"):
                    _sp.run(["pkg", "install", "ollama", "-y"], timeout=120)
                else:
                    _sp.run(["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"], shell=True, timeout=120)
                console.print("[green]✅ Ollama installed! Run: ollama serve &[/green]")
            elif eng == "llama.cpp":
                console.print("[cyan]Installing llama.cpp...[/cyan]")
                _sp.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", "/tmp/llama.cpp"], timeout=60)
                console.print("[green]✅ Cloned! Build: cd /tmp/llama.cpp && make -j4[/green]")
            else:
                console.print(f"[yellow]Unknown engine: {eng}. Try: ollama, llama.cpp, vllm[/yellow]")
        
        # Simplified commands
        if cmd == "/engine":
            import shutil as _sh, subprocess as _sp, os as _os
            # Detect installed engines on the system
            detected = []
            if _sh.which("ollama"):
                try:
                    r = _sp.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
                    ver = r.stdout.strip() or r.stderr.strip() or ""
                    detected.append(f"ollama [green]✅ {ver.split()[-1]}[/green]")
                except:
                    detected.append("ollama [yellow]✅ installed (no version)[/yellow]")
            else:
                detected.append("ollama [red]❌ not installed[/red]")
            if _sh.which("llama-server") or _sh.which("llama-cli"):
                detected.append("llama.cpp [green]✅[/green]")
            else:
                detected.append("llama.cpp [dim]-[/dim]")
            try:
                import vllm
                detected.append("vllm [green]✅[/green]")
            except:
                detected.append("vllm [dim]-[/dim]")
            console.print("[bold cyan]Inference engines:[/bold cyan]")
            for e in detected:
                console.print(f"  ● {e}")
            console.print()
            console.print("[dim]Current: ollama (default)[/dim]")
            console.print("[dim]Usage: /engine ollama | /engine llama.cpp | /engine vllm[/dim]")
            return
        if cmd.startswith("/engine "):
            eng = cmd[8:].strip()
            if eng == "ollama":
                console.print("[cyan]Installing Ollama...[/cyan]")
                import os as _os, subprocess as _sp
                if _os.path.exists("/data/data/com.termux"):
                    _sp.run(["pkg", "install", "ollama", "-y"], timeout=120)
                else:
                    _sp.run(["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"], shell=True, timeout=120)
                console.print("[green]✅ Ollama installed! Run: ollama serve &[/green]")
            return
        if cmd.startswith("/model ") or cmd == "/model":
            parts = cmd.split(None, 1)
            sub = parts[1].strip() if len(parts) > 1 else ""
            if not sub or sub == "list":
                # Show REAL installed models from Ollama
                import requests as _req
                installed = []
                try:
                    _r = _req.get("http://localhost:11434/api/tags", timeout=3)
                    if _r.status_code == 200:
                        installed = [m["name"] for m in _r.json().get("models", [])]
                except Exception:
                    pass
                if installed:
                    console.print(f"[bold cyan]Installed models ({len(installed)}):[/bold cyan]")
                    for i, m in enumerate(installed, 1):
                        console.print(f"  {i}. [cyan]{m}[/cyan]")
                else:
                    console.print("[yellow]No models installed yet.[/yellow]")
                # Also show popular suggestions
                console.print("[dim]Popular (download with /model <name>):[/dim]")
                popular = ["qwen2.5:1.5b", "llama3.2:3b", "gemma2:2b"]
                console.print(f"  [dim]{', '.join(popular)}[/dim]")
                return
            else:
                import subprocess as _sp
                console.print(f"[cyan]Downloading {sub}...[/cyan]")
                _sp.run(["ollama", "pull", sub], timeout=1800)
            return
        if cmd == "/hf search":
            console.print("[yellow]Usage: /hf search <query> (e.g. /hf search qwen 3b)[/yellow]")
            return
        
        if cmd == "/exit":
            self._handle_exit()
            _sys.exit(0)
        elif cmd in ("/help", "/?"):
            help_text = """
**Quick commands:**
- `/model <name>` — Download & set model (e.g. /model llama3.2:3b)
- `/model list` — Show popular models
- `/engine` — List inference engines
- `/engine <name>` — Install engine (ollama, llama.cpp)
- `/hf search <query>` — Search HuggingFace models
- `/hf pull <model-id>` — Download from HuggingFace

**Session:**
- `/exit` — Quit
- `/reset` — Reset conversation
- `/save` — Save session
- `/session list` — List sessions
- `/session load <name>` — Load session

**Provider:**
- `/provider` — List providers & status
- `/provider set <name>` — Switch provider
- `/provider key <name> <key>` — Set API key
- `/persona <mode>` — Switch mode (coder, sysadmin, research)

**Tools & Memory:**
- `/tools` — List available tools
- `/memory` — Show memory
- `/remember <fact>` — Save fact
- `/knowledge` — Show learned knowledge
- `/forget` — Clear knowledge

**System:**
- `/status` — Session info
- `/check` — System status (Ollama, RAM)
- `/compact` — Compact context
- `/vectordb` — Rebuild vector index
- `/ft` — Fine-tune model
- `/plugins` — List plugins
- `/plan` — Planning mode

**Tasks:**
- `/checklist` — Show tasks
- `/checklist add <task>` — Add task
- `/checklist done <n>` — Mark done

**Dev:**
- `/project` — Show project info
- `/tree` — Show project tree
- `/explorer` — File explorer
- `/cost` — Token usage & cost
- `/reference <url>` — Analyze GitHub repo
- `/schedule` — Task scheduler
    """
            console.print(Markdown(help_text))
        elif cmd == "/clear":
            console.clear()
            self._show_splash()
        elif cmd == "/reset":
            self.session.reset()
            self.tool_history.clear()
            console.print("[yellow]Session reset.[/yellow]")
        elif cmd == "/status":
            t = Table(box=box.SIMPLE)
            t.add_column("Key", style="cyan")
            t.add_column("Value")
            t.add_row("Model", self.provider.name)
            t.add_row("Messages", str(len(self.session.messages)))
            t.add_row("Tool calls", str(len(self.tool_history)))
            t.add_row("Tools available", str(len(list_tools())))
            console.print(t)
        elif cmd == "/tools":
            for name in list_tools():
                console.print(f"  [green]●[/green] {name}")
        elif cmd == "/history":
            sessions = sorted(self.sessions_dir.glob("*.json"))
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions[-5:]:
                    size = s.stat().st_size
                    console.print(f"  [dim]{s.name} ({size} bytes)[/dim]")
        elif cmd.startswith("/checklist"):
            parts = cmd.split(maxsplit=2)
            action = parts[1] if len(parts) > 1 else "show"
            if action == "add" and len(parts) > 2:
                self.checklist.append({"task": parts[2], "done": False})
                console.print(f"[green]Added: {parts[2]}[/green]")
            elif action == "done" and len(parts) > 2:
                idx = int(parts[2]) - 1
                if 0 <= idx < len(self.checklist):
                    self.checklist[idx]["done"] = True
                    console.print(f"[green]OK {self.checklist[idx][chr(34)+chr(116)+chr(97)+chr(115)+chr(107)+chr(34)]}[/green]")
                else:
                    console.print("[red]Invalid index[/red]")
            elif action == "rm" and len(parts) > 2:
                idx = int(parts[2]) - 1
                if 0 <= idx < len(self.checklist):
                    removed = self.checklist.pop(idx)
                    console.print(f"[yellow]Removed: {removed[chr(34)+chr(116)+chr(97)+chr(115)+chr(107)+chr(34)]}[/yellow]")
                else:
                    console.print("[red]Invalid index[/red]")
            elif action == "clear":
                self.checklist.clear()
                console.print("[yellow]Checklist cleared[/yellow]")
            else:
                if not self.checklist:
                    console.print("[dim]Checklist is empty.[/dim]")
                else:
                    for i, item in enumerate(self.checklist, 1):
                        mark = "OK" if item["done"] else "  "
                        style = "green" if item["done"] else "yellow"
                        console.print(f"  [{style}]{i}. [{mark}] {item[chr(34)+chr(116)+chr(97)+chr(115)+chr(107)+chr(34)]}[/{style}]")
        elif cmd == "/compact":
            before = len(self.session._messages)
            self.session._messages = compact_messages(
                self.session._messages, self.provider
            )
            after = len(self.session.messages)
            console.print(f"[green]Compacted: {before} → {after} messages[/green]")
        elif cmd.startswith("/plan "):
            goal = cmd[6:].strip()
            if not goal:
                console.print("[yellow]Usage: /plan <goal>[/yellow]")
            else:
                console.print("[bold cyan]Neural Planner[/bold cyan]")
                console.print(f"[dim]Goal: {goal}[/dim]\n")
                planner = PlannerAgent(self.session)
                try:
                    for event in planner.run_with_plan(goal):
                        if event["type"] == "plan":
                            console.print("[bold]Plan:[/bold]")
                            for i, s in enumerate(event["steps"], 1):
                                desc = s.get("desc", "?")
                                console.print(f"  [cyan]{i}.[/cyan] {desc}")
                            console.print()
                        elif event["type"] == "step_start":
                            idx = event["index"] + 1
                            desc = event.get("desc", "")[:50]
                            console.print(f"\n[bold]Step {idx}:[/bold] {desc} [dim](attempt {event['attempt']})[/dim]")
                        elif event["type"] == "token":
                            sys.stdout.write(event["content"])
                            sys.stdout.flush()
                        elif event["type"] == "tool_call":
                            console.print(f"\\n  [bright_black]⚡ {event['tool']}(...)[/bright_black]")
                        elif event["type"] == "approval_needed":
                            console.print(f"\\n  [bold yellow]⚠️  Allow {event['tool']}?[/bold yellow] [y/N] ", end="")
                            ans = input().strip().lower()
                            self.session._pending_approved = (ans == "y")
                        elif event["type"] == "tool_result":
                            r = event["content"][:150]
                            console.print(f"  [bright_black]  └─ {r}[/bright_black]")
                        elif event["type"] == "step_done":
                            console.print(f"\n  [green]✓ Step {event['index'] + 1} complete[/green]")
                        elif event["type"] == "step_retry":
                            console.print(f"\n  [yellow]↻ Retry {event['retry']}[/yellow]")
                        elif event["type"] == "step_failed":
                            console.print(f"\n  [red]✗ Step {event['index'] + 1} failed[/red]")
                        elif event["type"] == "final":
                            console.print(f"\n[bold cyan]Result:[/bold cyan]\n{event['content']}")
                except Exception as e:
                    console.print(f"\\n[red]Planner error: {e}[/red]")
        elif cmd == "/plan":
            console.print("[bold cyan]Neural Planner[/bold cyan]")
            console.print("  [dim]Usage: /plan <goal>[/dim]")
            console.print("  [dim]Example: /plan cek disk usage dan laporan[/dim]")
        elif cmd == "/knowledge":
            try:
                from knowledge import knowledge_summary, get_facts, get_skills
                console.print(f"[bold cyan]{knowledge_summary()}[/bold cyan]")
                facts = get_facts()
                if facts:
                    console.print("\n[bold]Facts:[/bold]")
                    for f in facts[-5:]:
                        console.print(f"  [dim]{f['topic']}[/dim]: {f['content'][:80]}...")
                skills = get_skills()
                if skills:
                    console.print("\n[bold]Skills:[/bold]")
                    for s in skills[-3:]:
                        console.print(f"  [dim]{s['name']}[/dim]: {s['pattern'][:60]}...")
            except ImportError:
                console.print("[dim]Knowledge system not available.[/dim]")
        elif cmd == "/forget":
            import json, os
            for f in ["facts.json", "skills.json"]:
                p = os.path.expanduser(f"~/rsa-agentic/knowledge/{f}")
                with open(p, "w") as fh:
                    json.dump([], fh)
            console.print("[yellow]Knowledge cleared.[/yellow]")
        elif cmd.startswith("/reference "):
            url = cmd[11:].strip()
            if not url:
                console.print("[yellow]Usage: /reference <github-url>[/yellow]")
            else:
                console.print(f"[bold cyan]Analyzing:[/bold cyan] {url}")
                try:
                    from reference import analyze, report
                    data = analyze(url)
                    result = report(data)
                    console.print(result)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
        elif cmd.startswith("/persona"):
            parts = cmd.split(maxsplit=1)
            pname = parts[1].strip() if len(parts) > 1 else ""
            from context import PERSONAS
            if pname in PERSONAS:
                self.session.persona = pname
                pn = PERSONAS[pname]["name"]
                console.print(f"[green]Switched to mode: {pn}[/green]")
            elif not pname:
                cur = self.session.persona
                p_info = PERSONAS.get(cur, {"name": cur})
                console.print(f"[bold cyan]Current mode: {p_info['name']}[/bold cyan]")
                console.print("[dim]Available:[/dim]")
                for k, v in PERSONAS.items():
                    mark = " >" if k == cur else "  "
                    vname = v["name"]
                    console.print(f"  {mark} [cyan]{k}[/cyan]  {vname}")
            else:
                console.print(f"[yellow]Unknown mode: {pname}. Try: coder, sysadmin, research, default[/yellow]")
        elif cmd == "/context":
            from context import build_context_block
            ctx = build_context_block()
            console.print(f"[bold cyan]Terminal Context:[/bold cyan]")
            for line in ctx.split("\n")[1:]:
                console.print(f"  [dim]{line}[/dim]")
        elif cmd == "/nemotron":
            console.print("[bold cyan]Nemotron Quick Setup[/bold cyan]")
            console.print("Step 1: Searching for Nemotron datasets...")
            from hf_manager import search_hf_datasets
            results = search_hf_datasets("nemotron agentic", limit=3)
            nemotron_id = None
            for r in results:
                if "SFT-Agentic" in r["id"]:
                    nemotron_id = r["id"]
                    console.print(f"  Found: [cyan]{r['id']}[/cyan] ({r['downloads']/1000:.0f}k downloads)")
                    break
            if not nemotron_id:
                console.print("[yellow]Nemotron-SFT-Agentic not found. Trying first result...[/yellow]")
                if results:
                    nemotron_id = results[0]["id"]
            
            if nemotron_id:
                console.print(f"\nStep 2: Downloading {nemotron_id}...")
                from hf_manager import pull_dataset
                for msg in pull_dataset(nemotron_id):
                    console.print(f"  {msg}")
                console.print("\nStep 3: Learning from dataset...")
                from hf_manager import learn_from_dataset
                result = learn_from_dataset(nemotron_id, limit=100)
                if "error" in result:
                    console.print(f"[red]{result['error']}[/red]")
                else:
                    console.print(f"[green]Extracted {result['patterns']} patterns from {result['domains']} domains[/green]")
                console.print("\n[bold green]Nemotron ready! Neural now knows tool patterns from DeepSeek V3.2[/bold green]")
            else:
                console.print("[red]No Nemotron datasets found. Try: /dataset search nemotron[/red]")
        elif cmd == "/models":
            try:
                from hf_manager import list_installed
                models = list_installed()
                if not models:
                    console.print("[dim]No models installed.[/dim]")
                    console.print("Try: /hf search <query>")
                else:
                    console.print("[bold cyan]Installed Models:[/bold cyan]")
                    for m in models:
                        backend_icon = {"ollama": "O", "gguf": "G"}.get(m["backend"], "?")
                        console.print(f"  [{backend_icon}] {m['name']:<30} {m['size']:>8}")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        elif cmd.startswith("/hf search "):
            query = cmd[11:].strip()
            if not query:
                console.print("[yellow]Usage: /hf search <query> (e.g. /hf search qwen 3b)[/yellow]")
            else:
                console.print(f"[bold]Searching HuggingFace for:[/bold] {query}")
                try:
                    from hf_manager import search_hf
                    results = search_hf(query, limit=8)
                    if not results:
                        console.print("[dim]No results found.[/dim]")
                    elif "error" in results[0]:
                        console.print(f"[red]API error: {results[0]['error']}[/red]")
                    else:
                        for i, r in enumerate(results, 1):
                            downloads = r.get("downloads", 0)
                            dl_str = f"{downloads/1000:.0f}k" if downloads > 1000 else str(downloads)
                            console.print(f"  {i}. [cyan]{r['id']}[/cyan] [dim]{dl_str} downloads[/dim]")
                        console.print("\nUse: /hf pull <number> to install")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
        elif cmd.startswith("/hf pull "):
            query = cmd[9:].strip()
            if not query:
                console.print("[yellow]Usage: /hf pull <model-id> or number from search[/yellow]")
            else:
                console.print(f"[bold]Installing:[/bold] {query}")
                try:
                    from hf_manager import pull_model
                    for msg in pull_model(query):
                        console.print(msg)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
        elif cmd.startswith("/model "):
            model_name = cmd[7:].strip()
            # Check if current provider is API (not ollama)
            if self.provider and "ollama" not in self.provider.name.lower():
                # Switch model on current API provider
                import os as _os, tomllib
                cfg_path = _os.path.expanduser("~/rsa-agentic/config.toml")
                cfg = tomllib.load(open(cfg_path, "rb"))
                pname = cfg.get("model", {}).get("provider", "")
                if pname and pname in cfg.get("model", {}):
                    if isinstance(cfg["model"][pname], dict):
                        cfg["model"][pname]["model"] = model_name
                    cfg["model"]["model_name"] = model_name
                    from tools.builtin import _write_toml
                    _write_toml(cfg, cfg_path)
                    from models.providers import create_provider
                    self.provider = create_provider(cfg)
                    self.config = cfg
                    console.print(f"[green]✅ Switched to {self.provider.name}[/green]")
                    return

            if not model_name:
                console.print("[yellow]Usage: /model <name> (e.g. /model qwen2.5:1.5b)[/yellow]")
            else:
                try:
                    from hf_manager import use_model
                    result = use_model(model_name)
                    console.print(f"[green]{result}[/green]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
        elif cmd == "/dataset list":
            try:
                from hf_manager import list_datasets
                ds = list_datasets()
                if not ds:
                    console.print("[dim]No datasets downloaded.[/dim]")
                    console.print("Try: /dataset pull nvidia/Nemotron-SFT-Agentic-v2")
                else:
                    console.print("[bold cyan]Downloaded Datasets:[/bold cyan]")
                    for name, info in ds.items():
                        console.print(f"  [green]●[/green] {name:<50} {info['samples']:>6} samples")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        elif cmd.startswith("/dataset pull "):
            name = cmd[14:].strip()
            if not name:
                console.print("[yellow]Usage: /dataset pull <hf-dataset-name>[/yellow]")
                console.print("[dim]Example: /dataset pull nvidia/Nemotron-SFT-Agentic-v2[/dim]")
            else:
                console.print(f"[bold]Downloading dataset:[/bold] {name}")
                from hf_manager import pull_dataset
                for msg in pull_dataset(name):
                    console.print(f"  {msg}")
        elif cmd.startswith("/dataset learn "):
            name = cmd[15:].strip()
            if not name:
                console.print("[yellow]Usage: /dataset learn <name>[/yellow]")
            else:
                console.print(f"[bold]Learning from:[/bold] {name}")
                from hf_manager import learn_from_dataset
                result = learn_from_dataset(name, limit=100)
                if "error" in result:
                    console.print(f"[red]{result['error']}[/red]")
                else:
                    console.print(f"[green]Extracted {result['patterns']} patterns from {result['domains']} domains[/green]")
                    console.print("[green]Injected into Neural knowledge![/green]")
        elif cmd.startswith("/dataset search "):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 3:
                console.print("[yellow]Search HF: /dataset search <query> (e.g. /dataset search nemotron)[/yellow]")
                console.print("[yellow]Search local: /dataset search <dataset> <query> (e.g. /dataset search nemotron art dealer)[/yellow]")
            elif len(parts) == 2:
                # Only 2 parts: search HF catalog
                query = parts[1]
                console.print(f"[bold]Searching HuggingFace datasets for:[/bold] {query}")
                from hf_manager import search_hf_datasets
                results = search_hf_datasets(query, limit=8)
                if not results:
                    console.print("[dim]No results[/dim]")
                elif "error" in results[0]:
                    console.print(f"[red]{results[0]['error']}[/red]")
                else:
                    for i, r in enumerate(results, 1):
                        icon = "📄" if r.get("has_jsonl") else "  "
                        console.print(f"  {i}. {icon} [cyan]{r['id']}[/cyan] [dim]{r['downloads']/1000:.0f}k downloads[/dim]")
                    console.print("\nUse: /dataset pull <number> or /dataset pull <full-name>")
            else:
                # 3 parts: search within downloaded dataset
                name = parts[1]
                query = parts[2]
                console.print(f"[bold]Searching {name} for:[/bold] {query}")
                from hf_manager import search_dataset
                result = search_dataset(name, query, limit=3)
                if "error" in result:
                    console.print(f"[red]{result['error']}[/red]")
                elif result["total"] == 0:
                    console.print("[dim]No matches[/dim]")
                else:
                    for r in result["results"]:
                        console.print(f"  [cyan][{r['domain']}][/cyan] {r['snippet'][:100]}...")
        elif cmd == "/ft":
            console.print("[yellow]Usage: /ft <dataset-name>[/yellow]")
            console.print("[dim]Available datasets: /dataset list[/dim]")
        elif cmd.startswith("/ft "):
            dset = cmd[4:].strip()
            from tools.builtin import _fine_tune
            console.print(f"[cyan]Fine-tuning with: {dset}...[/cyan]")
            result = _fine_tune(dset)
            console.print(f"[green]{result}[/green]")
        elif cmd == "/dataset":
            console.print("[bold cyan]Dataset Manager[/bold cyan]")
            console.print("  /dataset list")
            console.print("  /dataset pull <name>")
            console.print("  /dataset learn <name>")
            console.print("  /dataset search <name> <query>")
        elif cmd == "/model":
            import os as _os
            cfg_path = _os.path.expanduser("~/rsa-agentic/config.toml")
            if _os.path.exists(cfg_path):
                import tomllib
                cfg = tomllib.load(open(cfg_path, "rb"))
                current = cfg.get("model", {}).get("model_name", "unknown")
                console.print(f"[cyan]Current model: {current}[/cyan]")
            console.print("[yellow]Usage: /model <name> (e.g. /model llama3.2:3b)[/yellow]")
        elif cmd.startswith("/model "):
            model_name = cmd[7:].strip()
            # Check if current provider is API (not ollama)
            if self.provider and "ollama" not in self.provider.name.lower():
                # Switch model on current API provider
                import os as _os, tomllib
                cfg_path = _os.path.expanduser("~/rsa-agentic/config.toml")
                cfg = tomllib.load(open(cfg_path, "rb"))
                pname = cfg.get("model", {}).get("provider", "")
                if pname and pname in cfg.get("model", {}):
                    if isinstance(cfg["model"][pname], dict):
                        cfg["model"][pname]["model"] = model_name
                    cfg["model"]["model_name"] = model_name
                    from tools.builtin import _write_toml
                    _write_toml(cfg, cfg_path)
                    from models.providers import create_provider
                    self.provider = create_provider(cfg)
                    self.config = cfg
                    console.print(f"[green]✅ Switched to {self.provider.name}[/green]")
                    return

            if not model_name:
                console.print("[yellow]Usage: /model <name> (e.g. /model llama3.2:3b)[/yellow]")
            else:
                import tomllib, os as _os
                cfg_path = _os.path.expanduser("~/rsa-agentic/config.toml")
                if _os.path.exists(cfg_path):
                    cfg = tomllib.load(open(cfg_path, "rb"))
                    cfg["model"]["model_name"] = model_name
                    with open(cfg_path, "w") as f:
                        _write_toml(cfg, cfg_path)
                    console.print(f"[green]✅ Model set to {model_name}. Restart to apply.[/green]")
                else:
                    console.print("[red]Config not found at ~/rsa-agentic/config.toml[/red]")
        elif cmd == "/audit":
            import os as _os, sys as _sys, importlib
            console.print("[bold cyan]🔍 Self-Audit[/bold cyan]")
            console.print("Scanning own codebase for issues...\n")
            
            # 1. Check for bare except:
            issues = []
            code_files = ["agent.py", "neural.py", "tui.py", "tools/builtin.py", 
                         "db.py", "memory.py", "knowledge.py", "planner.py", 
                         "grammar.py", "compact.py", "server.py", "models/providers.py"]
            
            for f in code_files:
                fp = f"{_os.path.dirname(_os.path.abspath(__file__))}/{f}"
                if not _os.path.exists(fp):
                    continue
                content = open(fp).read()
                import re
                bare = re.findall(r'^\s*except:\s*$', content, re.MULTILINE)
                if bare:
                    issues.append(("❌ Bare except", f, f"{len(bare)} found"))
                if "TODO" in content or "FIXME" in content:
                    todos = re.findall(r'(?:TODO|FIXME)[^\n]*', content)
                    for t in todos[:3]:
                        issues.append(("📝 TODO", f, t.strip()))
                    
            if issues:
                console.print(f"[yellow]Found {len(issues)} issues:[/yellow]")
                for typ, file, msg in issues[:15]:
                    console.print(f"  {typ:<15} {file:<20} {msg[:40]}")
            else:
                console.print("[green]✅ No issues found![/green]")
            
            # 2. Git status
            console.print("\n[bold]Git status:[/bold]", end=" ")
            import subprocess as _sp
            r = _sp.run(["git", "status", "--short"], capture_output=True, text=True, timeout=5, 
                       cwd=_os.path.dirname(_os.path.abspath(__file__)))
            if r.stdout.strip():
                console.print(f"[yellow]{r.stdout.count(chr(10))} uncommitted[/yellow]")
                for line in r.stdout.strip().split(chr(10))[:5]:
                    console.print(f"  [dim]{line[:60]}[/dim]")
            else:
                console.print("[green]clean[/green]")
            
            # 3. File sizes
            console.print("\n[bold]Codebase size:[/bold]", end=" ")
            total = 0
            for f in code_files:
                fp = f"{_os.path.dirname(_os.path.abspath(__file__))}/{f}"
                if _os.path.exists(fp):
                    total += _os.path.getsize(fp)
            console.print(f"{total/1024:.0f} KB across {len(code_files)} files")
            console.print("\n[dim]Usage: /audit fix to auto-fix common issues[/dim]")
        
        elif cmd == "/audit fix":
            console.print("[yellow]Auto-fix requires API/7B+ model. Run manually or use /provider set openai[/yellow]")
        
        elif cmd == "/setup":
            import os as _os
            # Create symlink in Termux bin dir
            target = _os.path.expanduser("~/rsa-agentic/rsa")
            if _os.path.exists("/data/data/com.termux"):
                link = "/data/data/com.termux/files/usr/bin/rsa"
                try:
                    if _os.path.exists(link):
                        _os.remove(link)
                    _os.symlink(target, link)
                    console.print(f"[green]✅ Symlink created: rsa -> {target}[/green]")
                    console.print("[green]✅ Now you can type 'rsa' from anywhere![/green]")
                except Exception as e:
                    console.print(f"[yellow]Could not create symlink: {e}[/yellow]")
                    console.print("[yellow]Try: pkg install termux-exec[/yellow]")
            else:
                # Linux: add to .bashrc
                rc = _os.path.expanduser("~/.bashrc")
                line = 'export PATH=$PATH:~/rsa-agentic'
                if line not in open(rc).read():
                    with open(rc, "a") as f:
                        f.write(f"\n{line}\n")
                    console.print(f"[green]✅ Added ~/rsa-agentic to PATH in {rc}[/green]")
                console.print("[green]✅ Now type 'rsa' or restart terminal[/green]")
        elif cmd == "/install":
            console.print("[yellow]Usage: /install ollama | /install model <name> | /install llama.cpp | /install engines[/yellow]")
        elif cmd.startswith("/install "):
            # Auto-install Ollama (Termux / Linux)
            import subprocess, sys, os as _os
            parts = cmd.split(None, 2)
            action = parts[1] if len(parts) > 1 else ""
            model_name = parts[2] if len(parts) > 2 else ""
            if not action:
                console.print("[yellow]Usage: /install ollama | /install model <name> | /install llama.cpp | /install vllm[/yellow]")
            elif action == "ollama":
                console.print("[cyan]Installing Ollama...[/cyan]")
                if _os.path.exists("/data/data/com.termux"):
                    r = subprocess.run(["pkg", "install", "ollama", "-y"], capture_output=True, text=True, timeout=120)
                else:
                    r = subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"], capture_output=True, text=True, timeout=120, shell=True)
                if r.returncode == 0:
                    console.print("[green]✅ Ollama installed! Starting...[/green]")
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    console.print(f"[red]Install failed: {r.stderr[:200]}[/red]")
            elif action == "llama.cpp":
                console.print("[cyan]Installing llama.cpp (native C++ inference)...[/cyan]")
                if _os.path.exists("/data/data/com.termux"):
                    r = subprocess.run(["pkg", "install", "llama.cpp", "-y"], capture_output=True, text=True, timeout=120)
                else:
                    r = subprocess.run(["bash", "-c", "git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make -j4"], capture_output=True, text=True, timeout=300)
                if r.returncode == 0:
                    console.print("[green]✅ llama.cpp installed![/green]")
                    console.print("[dim]Run: cd ~/llama.cpp && ./server -m model.gguf[/dim]")
                else:
                    console.print(f"[red]Install failed: {r.stderr[:200]}[/red]")
            elif action == "vllm":
                console.print("[cyan]Installing vLLM (fast inference server)...[/cyan]")
                r = subprocess.run(["pip", "install", "vllm", "flask"], capture_output=True, text=True, timeout=300)
                if r.returncode == 0:
                    console.print("[green]✅ vLLM installed![/green]")
                    console.print("[dim]Run: python -m vllm.entrypoints.openai.api_server --model model-name[/dim]")
                else:
                    console.print(f"[red]Install failed: {r.stderr[:200]}[/red]")
            elif action == "engines":
                console.print("[bold cyan]Available inference engines:[/bold cyan]")
                console.print("  [cyan]ollama[/cyan]     — Default, easiest (recommended)")
                console.print("  [cyan]llama.cpp[/cyan]  — Native C++, faster on CPU")
                console.print("  [cyan]vllm[/cyan]       — Fastest on GPU")
                console.print()
                console.print("[dim]Usage: /install <engine> (e.g. /install ollama)[/dim]")
            elif action == "model" and model_name:
                console.print(f"[cyan]Downloading {model_name}...[/cyan]")
                # Try Ollama first
                r = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True, timeout=1800)
                if r.returncode == 0:
                    console.print(f"[green]✅ {model_name} downloaded![/green]")
                else:
                    # Fallback to HuggingFace
                    console.print(f"[yellow]Not found on Ollama. Searching HuggingFace...[/yellow]")
                    try:
                        import requests
                        hf_r = requests.get(
                            f"https://huggingface.co/api/models?search={model_name}&sort=downloads&direction=-1&limit=5",
                            timeout=10
                        )
                        if hf_r.status_code == 200:
                            results = hf_r.json()
                            if results:
                                console.print("[bold cyan]Found on HuggingFace:[/bold cyan]")
                                for i, m in enumerate(results[:5], 1):
                                    console.print(f"  {i}. [cyan]{m['modelId']}[/cyan] [dim]{m.get('downloads',0):,} downloads[/dim]")
                                console.print()
                                console.print("[dim]Use: /hf pull <model-id> to download[/dim]")
                            else:
                                console.print("[red]Model not found anywhere.[/red]")
                        else:
                            console.print(f"[red]Failed: {r.stderr[:200]}[/red]")
                    except Exception as e:
                        console.print(f"[red]Error searching: {e}[/red]")
            elif action == "model" and not model_name:
                popular = [
                    ("qwen2.5:1.5b", "1.1 GB", "Best for HP 6GB RAM"),
                    ("qwen2.5:3b", "2.0 GB", "Good balance"),
                    ("llama3.2:3b", "2.0 GB", "Good English tool calling"),
                    ("gemma2:2b", "1.5 GB", "Ringan, cukup akurat"),
                    ("mistral:7b", "4.2 GB", "Butuh RAM lebih"),
                    ("phi3:3.8b", "2.3 GB", "Microsoft, lumayan"),
                ]
                console.print("[bold cyan]Popular models:[/bold cyan]")
                for i, (name, size, note) in enumerate(popular, 1):
                    console.print(f"  {i}. [cyan]{name:<20}[/cyan] {size:<8} [dim]{note}[/dim]")
                console.print()
                console.print("[dim]Type: /install model <name> (e.g. /install model qwen2.5:1.5b)[/dim]")
            else:
                console.print("[yellow]Usage: /install ollama | /install model <name> | /install llama.cpp | /install engines[/yellow]")
        elif cmd == "/check":
            # Quick system check
            import requests as _req, os as _os
            console.print("[bold cyan]System Check[/bold cyan]")
            # Ollama check
            try:
                _r = _req.get("http://localhost:11434/api/tags", timeout=2)
                if _r.status_code == 200:
                    models = _r.json().get("models", [])
                    names = [m["name"] for m in models]
                    console.print(f"  [green]✅ Ollama running[/green] ({len(models)} models: {', '.join(names[:3])})")
                else:
                    console.print("  [red]❌ Ollama not responding[/red]")
            except:
                console.print("  [red]❌ Ollama not running[/red]")
            # RAM check
            import subprocess
            r = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5)
            console.print(f"  [cyan]💾 RAM:[/cyan] {r.stdout[:80].strip()}")
        elif cmd == "/provider":
            import os, tomllib
            cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            current = cfg.get("model", {}).get("provider", "ollama")
            model_name = cfg.get("model", {}).get("model_name", "?")
            console.print(f"[bold cyan]Current provider: {current}[/bold cyan]")
            # Real Ollama status
            if current == "ollama":
                status, color = _check_ollama_status(model=model_name)
                console.print(f"  Status: [{color}]{status}[/{color}]")
            console.print("")
            console.print("[bold]Available:[/bold]")
            providers = [k for k in cfg.get("model", {}).keys() if k not in ("provider","model_name","temperature","max_tokens","ctx_size")]
            for pv in providers:
                pcfg = cfg.get("model", {}).get(pv, {})
                if isinstance(pcfg, dict):
                    has_key = bool(pcfg.get("api_key",""))
                    key_status = "✅ key set" if has_key else "❌ no key"
                    host = pcfg.get("host") or pcfg.get("base_url","")
                    console.print(f"  [cyan]{pv:<10}[/cyan] {host:<30} {key_status}")
            console.print("")
            console.print("[dim]Usage:[/dim]")
            console.print("  /provider set openai             Switch provider")
            console.print("  /provider key openai sk-xxx      Set API key")
            console.print("  /provider add myapi url key      Add custom provider")
        elif cmd.startswith("/provider set "):
            rest = cmd[14:].strip()
            parts = rest.split(None, 1)
            pname = parts[0]
            model_override = parts[1] if len(parts) > 1 else None
            import os, tomllib
            cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            cfg["model"]["provider"] = pname
            # Allow model override: /provider set <name> <model>
            if model_override:
                if pname in cfg["model"] and isinstance(cfg["model"][pname], dict):
                    cfg["model"][pname]["model"] = model_override
                else:
                    cfg["model"][pname] = {"model": model_override, "api_key": ""}
                cfg["model"]["model_name"] = model_override
            if pname == "openai" and not cfg["model"].get("openai",{}).get("api_key"):
                console.print("[yellow]Warning: OpenAI key not set. Set with: /provider key openai <key>[/yellow]")
            with open(cfg_path, "w") as f:
                _write_toml(cfg, cfg_path)
            console.print(f"[green]Switched to provider: {pname}[/green]")
            # Apply immediately - recreate provider & reload session
            from models.providers import create_provider
            self.provider = create_provider(cfg)
            self.config = cfg
            console.print(f"[dim]Provider active: {self.provider.name}[/dim]")
        elif cmd.startswith("/provider key "):
            rest = cmd[14:].strip()  # remove "/provider key "
            parts = rest.split(None, 1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /provider key <provider> <api_key>[/yellow]")
            else:
                pname = parts[0]
                key = parts[1]
                import os, tomllib
                cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
                with open(cfg_path, "rb") as f:
                    cfg = tomllib.load(f)
                if pname not in cfg.get("model", {}):
                    cfg["model"][pname] = {"api_key": ""}
                cfg["model"][pname]["api_key"] = key
                with open(cfg_path, "w") as f:
                    _write_toml(cfg, cfg_path)
                console.print(f"[green]API key set for {pname}[/green]")
        elif cmd.startswith("/provider add "):
            parts = cmd.split(maxsplit=3)
            if len(parts) < 3:
                console.print("[yellow]Usage: /provider add <name> <base_url> [api_key][/yellow]")
            else:
                pname = parts[1]
                url = parts[2]
                key = parts[3] if len(parts) > 3 else ""
                import os, tomllib
                cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
                with open(cfg_path, "rb") as f:
                    cfg = tomllib.load(f)
                cfg["model"][pname] = {"api_key": key, "base_url": url, "model": "gpt-4o"}
                with open(cfg_path, "w") as f:
                    _write_toml(cfg, cfg_path)
                console.print(f"[green]Added provider: {pname}[/green]")
                console.print("[dim]To use it: /provider set {pname}[/dim]")
        elif cmd == "/project":
            import subprocess, json, os
            cwd = os.getcwd()
            console.print(f"[bold cyan]Project: {os.path.basename(cwd)}[/bold cyan]")
            console.print(f"  [dim]Path: {cwd}[/dim]")
            # Auto-detect project type
            detectors = {
                "package.json": ("Node.js", lambda d: json.loads(open(f"{d}/package.json").read()).get("name","?")),
                "Cargo.toml": ("Rust", lambda d: [l for l in open(f"{d}/Cargo.toml") if l.startswith("name")][0].split("=")[-1].strip().strip('"')),
                "pyproject.toml": ("Python", lambda d: "pyproject"),
                "go.mod": ("Go", lambda d: open(f"{d}/go.mod").readline().split()[-1]),
                "Gemfile": ("Ruby", lambda d: "ruby"),
                "composer.json": ("PHP", lambda d: json.loads(open(f"{d}/composer.json").read()).get("name","?")),
                "build.gradle": ("Java/Kotlin", lambda d: "gradle"),
                "CMakeLists.txt": ("C/C++", lambda d: "cmake"),
                "Makefile": ("C/C++", lambda d: "make"),
                "Dockerfile": ("Docker", lambda d: "docker"),
            }
            found = False
            for fname, (lang, parser) in detectors.items():
                fp = f"{cwd}/{fname}"
                if os.path.exists(fp):
                    try:
                        name = parser(cwd)
                        console.print(f"  [green]●[/green] {lang:<15} {name}")
                    except Exception:
                        console.print(f"  [green]●[/green] {lang:<15} {fname}")
                    found = True
            # Git info
            try:
                r = subprocess.run(["git","remote","-v"], capture_output=True, text=True, timeout=5, cwd=cwd)
                if r.stdout:
                    for line in r.stdout.split("\n")[0:1]:
                        parts = line.split()
                        if len(parts) >= 2:
                            console.print(f"  [cyan]●[/cyan] Git remote:  {parts[1]}")
            except: pass
            # Dependencies count
            for pf, label in [("package.json", "npm"), ("Cargo.lock", "cargo"), ("requirements.txt", "pip")]:
                if os.path.exists(f"{cwd}/{pf}"):
                    console.print(f"  [dim]●[/dim] {label} project detected")
            if not found:
                console.print("  [yellow]Unknown project type[/yellow]")
        elif cmd == "/tree":
            import os
            cmd_str = cmd.strip()
            depth = 2
            parts = cmd_str.split()
            if len(parts) > 1:
                try: depth = int(parts[1])
                except: depth = 2
            import subprocess
            cwd = os.getcwd()
            try:
                r = subprocess.run(["find",cwd,"-maxdepth",str(depth),"-not","-path","*/node_modules/*","-not","-path","*/.git/*","-not","-path","*/__pycache__/*","-not","-path","*/vendor/*","|","sort"], capture_output=True, text=True, timeout=10, shell=True)
                lines = r.stdout.strip().split("\n")[:50]
                console.print(f"[bold cyan]Project Tree[/bold cyan] (depth={depth}, max 50)")
                for line in lines:
                    if line.strip():
                        rel = line.replace(cwd, ".").strip("/")
                        depth_level = rel.count("/")
                        indent = "  " * depth_level
                        name = rel.split("/")[-1]
                        icon = "📁" if os.path.isdir(line) else "📄"
                        console.print(f"  {indent}{icon} {name}")
                if len(lines) >= 50:
                    console.print(f"  [dim]... (truncated at 50)[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        elif cmd.startswith("/cost history"):
            from db import get_cost_summary
            rows = get_cost_summary(10)
            console.print("[bold cyan]Cost History[/bold cyan]")
            if not rows:
                console.print("  [dim]No cost data yet[/dim]")
            for r in rows:
                console.print(f"  {r['session_id'][:8]}  {r['provider']:<15}  {r['inp']+r['out']:>6} tok  ${r['cost']:.6f}")
        elif cmd == "/cost":
            console.print(f"[bold cyan]Session Cost[/bold cyan]")
            try:
                t = self.session.total_tokens
                provider_name = getattr(self.provider, "name", "?")
                inp = t.get("input", 0)
                out = t.get("output", 0)
                console.print(f"  Provider:  {provider_name}")
                console.print(f"  Input:     {inp:,} tokens")
                console.print(f"  Output:    {out:,} tokens")
                console.print(f"  Total:     {inp+out:,} tokens")
                # Show timing if available
                try:
                    t = getattr(self.provider, "last_tokens", {})
                    ttft = t.get("ttft", 0)
                    elapsed = t.get("elapsed", 0)
                    if ttft:
                        console.print(f"  TTFT:      {ttft:.2f}s")
                    if elapsed:
                        console.print(f"  Elapsed:   {elapsed:.2f}s")
                except: pass
                console.print(f"  Messages:  {len(self.session.messages)}")
                try:
                    from db import log_cost
                    log_cost(self.session.session_id, provider_name, inp, out, 0.0)
                except Exception: pass
            except Exception:
                console.print(f"  [dim]No session data[/dim]")
        elif cmd.startswith("/schedule add"):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 3:
                console.print("[yellow]Usage: /schedule add <name> <goal>[/yellow]")
            else:
                from db import add_task
                add_task(parts[1], "manual", parts[2])
                console.print(f"[green]Scheduled: {parts[1]}[/green]")
        elif cmd.startswith("/schedule run"):
            from db import get_due_tasks
            tasks = get_due_tasks()
            if not tasks:
                console.print("[dim]No tasks[/dim]")
            else:
                for t in tasks:
                    console.print(f"  [cyan]●[/cyan] {t['name']}: running...")
                    self.session.run(t["goal"])
                    console.print(f"  [green]  OK[/green]")
        elif cmd == "/schedule":
            from db import get_due_tasks
            tasks = get_due_tasks()
            console.print("[bold cyan]Scheduled Tasks[/bold cyan]")
            if not tasks:
                console.print("  [dim]None. Add: /schedule add backup Backup data[/dim]")
            for t in tasks:
                console.print(f"  {t['name']:<20} {t['goal'][:60]}")
        elif cmd == "/explorer":
            import subprocess, os
            cwd = os.getcwd()
            console.print("[bold cyan]File Explorer[/bold cyan]")
            console.print(f"  [dim]CWD: {cwd}[/dim]")
            console.print("  [green]ls <path>[/green] | [green]cat <file>[/green] | [green]up[/green] | [green]q[/green]")
            cur = cwd
            while True:
                try:
                    ans = console.input(f"  [cyan]{cur}> [/cyan]").strip()
                    if ans == "q": break
                    if ans == "up": cur = os.path.dirname(cur); continue
                    if ans.startswith("ls "):
                        ap = os.path.join(cur, ans[3:].strip())
                        if os.path.isdir(ap):
                            for item in os.listdir(ap)[:25]:
                                icon = "📁" if os.path.isdir(os.path.join(ap,item)) else "📄"
                                console.print(f"    {icon} {item}")
                            cur = ap
                        else: console.print("  [red]Not a dir[/red]")
                    elif ans.startswith("cat "):
                        ap = os.path.join(cur, ans[4:].strip())
                        if os.path.isfile(ap):
                            console.print(open(ap).read()[:1500])
                        else: console.print("  [red]Not a file[/red]")
                except KeyboardInterrupt:
                    break
        elif cmd == "/quant":
            try:
                from quant import recommend_quantization, suggest_model, get_available_ram
                ram = get_available_ram()
                console.print(f"[bold cyan]Hardware Analysis[/bold cyan]")
                console.print(f"  RAM available: {ram:.1f} GB")
                for size in ["1.5b", "3b", "7b"]:
                    rec = recommend_quantization(size)
                    if "error" not in rec:
                        console.print(f"  {size:<6} → {rec['recommended']:>2} ({rec['model_size_gb']:.1f} GB)")
                console.print(f"\n[bold]Best model:[/bold] {suggest_model()}")
                console.print("[dim]Tip: /hf search qwen 3b to find bigger models[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        elif cmd == "/vectordb":
            try:
                # Ensure embedding model is available
                import requests as _req, subprocess as _sp
                _check = _req.get("http://localhost:11434/api/tags", timeout=3)
                if _check.status_code == 200:
                    installed = [m["name"] for m in _check.json().get("models", [])]
                    if "nomic-embed-text:latest" not in installed:
                        console.print("[yellow]Downloading nomic-embed-text for embeddings...[/yellow]")
                        _sp.run(["ollama", "pull", "nomic-embed-text"], timeout=300)
                        console.print("[green]✅ Embedding model ready[/green]")
                from vectordb import add_from_knowledge, search
                console.print("[bold cyan]Vector Database[/bold cyan]")
                console.print("Rebuilding vector index from knowledge...")
                n = add_from_knowledge("knowledge")
                console.print(f"[green]Indexed {n} documents with embeddings[/green]")
                console.print("[dim]Using nomic-embed-text via Ollama[/dim]")
                console.print("[dim]Total: 768-d vectors[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        elif cmd == "/memory":
            try:
                from memory import working_context, episodic_context, preference_context, add_fact, load_working
                console.print("[bold cyan]Memory Status[/bold cyan]")
                wc = working_context()
                if wc:
                    lines = wc.split("\n")
                    if len(lines) > 1:
                        console.print(f"[bold]Working Memory ({len(lines)-1} facts)[/bold]")
                        for l in lines[1:]:
                            console.print(f"  {l}")
                ec = episodic_context()
                if ec:
                    lines = ec.split("\n")
                    console.print(f"[bold]Past Sessions ({len(lines)-1})[/bold]")
                    for l in lines[1:]:
                        console.print(f"  [dim]{l}[/dim]")
                pc = preference_context()
                if pc:
                    console.print("[bold]Preferences[/bold]")
                    lines = pc.split("\n")
                    for l in lines[1:]:
                        console.print(f"  {l}")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        elif cmd.startswith("/remember "):
            fact = cmd[10:].strip()
            if fact:
                from memory import add_fact
                add_fact(fact)
                console.print(f"[green]Remembered: {fact}[/green]")
        elif cmd.startswith("/pref "):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /pref key=value (e.g. /pref shell=bash)[/yellow]")
            else:
                kv = parts[1].split("=", 1)
                if len(kv) == 2:
                    from memory import save_pref
                    save_pref(kv[0].strip(), kv[1].strip())
                    console.print(f"[green]Preference saved: {kv[0].strip()} = {kv[1].strip()}[/green]")
                else:
                    console.print("[yellow]Format: key=value[/yellow]")
        elif cmd.startswith("/ft "):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 2:
                console.print("[yellow]Usage: /ft <dataset> [base_model]")
                console.print("  /ft dataset.jsonl qwen3:0.6b")
                console.print("  /ft nemotron-tool_calling.jsonl qwen2.5:1.5b")
            else:
                ds = parts[1]
                model = parts[2] if len(parts) > 2 else "qwen3:0.6b"
                console.print(f"[bold cyan]Fine-Tuning[/bold cyan]")
                from fine_tune import run, detect_gpu
                gpu = detect_gpu()
                if not gpu["available"]:
                    console.print("[yellow]Warning: No GPU detected. Training will be SLOW on CPU.[/yellow]")
                result = run(ds, model)
                console.print(result)
        elif cmd == "/plugins":
            try:
                from plugin_loader import list_loaded, list_tools as plt
                loaded = list_loaded()
                if loaded:
                    console.print("[bold]Loaded plugins:[/bold]")
                    for p in loaded:
                        console.print(f"  [green]●[/green] {p}")
                    ptools = plt()
                    if ptools:
                        console.print("[bold]Plugin tools:[/bold]")
                        for t in ptools:
                            console.print(f"  [cyan]  └ {t}[/cyan]")
                else:
                    console.print("[dim]No plugins loaded.[/dim]")
            except ImportError:
                console.print("[dim]Plugin system not available.[/dim]")
        elif cmd.startswith("/session"):
            parts = cmd.split(maxsplit=1)
            sub = parts[1].strip() if len(parts) > 1 else ""
            if not sub or sub == "list":
                sessions = sorted(self.sessions_dir.glob("*.json"), reverse=True)
                if not sessions:
                    console.print("[dim]No saved sessions.[/dim]")
                else:
                    console.print("[bold]Saved sessions:[/bold]")
                    for i, s in enumerate(sessions[:10], 1):
                        size = s.stat().st_size
                        ts = s.stem.replace("session_", "").replace("_", " ")[:16]
                        console.print(f"  {i}. [cyan]{s.name}[/cyan] ({ts})")
            elif sub.startswith("load "):
                name = sub[5:].strip()
                path = self.sessions_dir / name
                if not path.exists():
                    path = self.sessions_dir / f"{name}.json"
                if not path.exists():
                    console.print(f"[red]Session not found: {name}[/red]")
                else:
                    import json
                    data = json.loads(path.read_text())
                    if "messages" in data:
                        self.session._messages = data["messages"]
                        console.print(f"[green]Loaded {len(data['messages'])} messages from {name}[/green]")
                    else:
                        console.print("[red]Invalid session file[/red]")
            elif sub.startswith("rm "):
                name = sub[3:].strip()
                path = self.sessions_dir / name
                if not path.exists():
                    path = self.sessions_dir / f"{name}.json"
                if path.exists():
                    path.unlink()
                    console.print(f"[green]Deleted: {name}[/green]")
                else:
                    console.print(f"[red]Not found: {name}[/red]")
            else:
                console.print("[yellow]Usage: /session [list|load <name>|rm <name>][/yellow]")
        elif cmd == "/save":
            path = self.sessions_dir / f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            data = {
                "messages": self.session.messages,
                "tool_history": self.tool_history,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            console.print(f"[green]✓ Saved: {path}[/green]")
        else:
            # Tool shortcut commands - bypass model, run directly
            parts = cmd.split(None, 1)
            tool_name = parts[0][1:] if parts else ""  # remove /
            tool_args = parts[1] if len(parts) > 1 else ""
            
            from tools.builtin import get_tool, list_tools
            if tool_name in list_tools():
                tool = get_tool(tool_name)
                if tool:
                    console.print(f"  [dim]⚡ /{tool_name} {tool_args}[/dim]")
                    try:
                        # Convert args: first param name = the main arg
                        params = list(tool.params.keys())
                        kwargs = {}
                        if tool_args and params:
                            kwargs[params[0]] = tool_args
                        result = tool(**kwargs)
                        console.print(f"  [green]{str(result)[:500]}[/green]")
                    except Exception as e:
                        console.print(f"  [red]Error: {e}[/red]")
                else:
                    console.print(f"[red]Unknown command: {cmd}. Try /help[/red]")
            else:
                console.print(f"[red]Unknown command: {cmd}. Try /help[/red]")

    def _handle_exit(self):
        console.print("\n[dim]Neural session ended.[/dim]")
