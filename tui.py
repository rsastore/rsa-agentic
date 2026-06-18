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
║           Neural v0.1                    ║
║     Autonomous AI Agent — Local LLM      ║
╚══════════════════════════════════════════╝
"""

# ── Config Loader ──────────────────────────────────────────────
def load_config():
    import tomllib
    cfg_path = Path(os_mod.path.expanduser("~/neural/config.toml"))
    if not cfg_path.exists():
        console.print("[red]Config not found. Run setup first.[/red]")
        return None
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)

def bootstrap_display(provider_name: str, model_name: str, tool_count: int):
    grid = Table.grid(padding=1)
    grid.add_column(style="cyan bold")
    grid.add_column()
    grid.add_row("● Model", f"{model_name}")
    grid.add_row("● Provider", provider_name)
    grid.add_row("● Tools", str(tool_count))
    grid.add_row("● Status", "✅ Ollama connected")
    console.print(Panel(grid, title="[bold green]Neural[/bold green]", border_style="cyan"))

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
        self.sessions_dir = Path(os_mod.path.expanduser("~/neural/sessions"))
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
            except:
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
                for event in self.session.run_stream(user_input):
                    if event["type"] == "token":
                        sys.stdout.write(event["content"])
                        sys.stdout.flush()
                        buf.append(event["content"])
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
        cmd = cmd.strip().lower()

        if cmd == "/exit":
            self._handle_exit()
            sys.exit(0)
        elif cmd in ("/help", "/?"):
            help_text = """
**Commands:**
- `/exit` — Quit Neural
- `/help` — Show this help
- `/clear` — Clear screen
- `/reset` — Reset conversation
- `/status` — Show session info
- `/tools` — List available tools
- `/compact` — Compact long context (summarize old messages)
- `/plan` — Show planning mode
- `/plugins` — List loaded plugins & tools
- `/checklist` — Show tasks
- `/checklist add <task>` — Add task
- `/checklist done <n>` — Mark done
- `/checklist rm <n>` — Remove task
- `/checklist clear` — Clear all
- `/history` — Show recent sessions
- `/session list` — List all sessions
- `/session load <name>` — Load a session
- `/session rm <name>` — Delete a session
- `/save` — Force save session
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
            before = len(self.session.messages)
            self.session.messages = compact_messages(
                self.session.messages, self.provider
            )
            after = len(self.session.messages)
            console.print(f"[green]Compacted: {before} → {after} messages[/green]")
        elif cmd.startswith("/plan "):
            goal = cmd[6:].strip()
            if not goal:
                console.print("[yellow]Usage: /plan <goal>[/yellow]")
            else:
                console.print("[bold cyan]Neural Planner[/bold cyan]")
                console.print(f"[dim]Goal: {goal}[/dim]\\n")
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
                            console.print(f"\\n[bold]Step {idx}:[/bold] {desc} [dim](attempt {event['attempt']})[/dim]")
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
                            console.print(f"\\n  [green]✓ Step {event['index'] + 1} complete[/green]")
                        elif event["type"] == "step_retry":
                            console.print(f"\\n  [yellow]↻ Retry {event['retry']}[/yellow]")
                        elif event["type"] == "step_failed":
                            console.print(f"\\n  [red]✗ Step {event['index'] + 1} failed[/red]")
                        elif event["type"] == "final":
                            console.print(f"\\n[bold cyan]Result:[/bold cyan]\\n{event['content']}")
                except Exception as e:
                    console.print(f"\\n[red]Planner error: {e}[/red]")
        elif cmd == "/plan":
            console.print("[bold cyan]Neural Planner[/bold cyan]")
            console.print("  [dim]Usage: /plan <goal>[/dim]")
            console.print("  [dim]Example: /plan cek disk usage dan laporan[/dim]")
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
            console.print(f"[red]Unknown command: {cmd}. Try /help[/red]")

    def _handle_exit(self):
        console.print("\n[dim]Neural session ended.[/dim]")
