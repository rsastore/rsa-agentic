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
                    HTML("<style prompt>┃ </style>"),
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

            # Show thinking
            sys.stdout.write("  ⚡ ")
            sys.stdout.flush()

            try:
                output = self.session.run(user_input)
                self.tool_history.clear()

                # Print result in chat bubble
                self._print_chat_bubble("assistant", output)

                # Save session
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
- `/history` — Show recent sessions
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
                    console.print(f"  [dim]{s.name}[/dim]")
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
