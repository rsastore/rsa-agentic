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
- `/knowledge` — Show what Neural has learned
- `/forget` — Clear all learned knowledge
- `/persona` — Show current mode
- `/provider` — List providers & API key status
- `/provider set <name>` — Switch provider
- `/provider key <name> <key>` — Set API key
- `/provider add <name> <url> [key]` — Add custom provider
- `/persona <mode>` — Switch mode (coder, sysadmin, research, default)
- `/reference <url>` — Analyze a GitHub repo and compare with Neural
- `/context` — Show terminal context
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
                p = os.path.expanduser(f"~/neural/knowledge/{f}")
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
        elif cmd == "/dataset":
            console.print("[bold cyan]Dataset Manager[/bold cyan]")
            console.print("  /dataset list")
            console.print("  /dataset pull <name>")
            console.print("  /dataset learn <name>")
            console.print("  /dataset search <name> <query>")
        elif cmd == "/provider":
            import tomllib
            cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            current = cfg.get("model", {}).get("provider", "ollama")
            console.print(f"[bold cyan]Current provider: {current}[/bold cyan]")
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
            pname = cmd[14:].strip()
            import tomllib, tomli_w
            cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            cfg["model"]["provider"] = pname
            if pname == "openai" and not cfg["model"].get("openai",{}).get("api_key"):
                console.print("[yellow]Warning: OpenAI key not set. Set with: /provider key openai <key>[/yellow]")
            with open(cfg_path, "w") as f:
                tomli_w.dump(cfg, f)
            console.print(f"[green]Switched to provider: {pname}[/green]")
            console.print("[dim]Restart Neural for changes to take effect.[/dim]")
        elif cmd.startswith("/provider key "):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 3:
                console.print("[yellow]Usage: /provider key <provider> <api_key>[/yellow]")
            else:
                pname = parts[1]
                key = parts[2]
                import tomllib, tomli_w
                cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
                with open(cfg_path, "rb") as f:
                    cfg = tomllib.load(f)
                if pname not in cfg.get("model", {}):
                    cfg["model"][pname] = {"api_key": ""}
                cfg["model"][pname]["api_key"] = key
                with open(cfg_path, "w") as f:
                    tomli_w.dump(cfg, f)
                console.print(f"[green]API key set for {pname}[/green]")
        elif cmd.startswith("/provider add "):
            parts = cmd.split(maxsplit=3)
            if len(parts) < 3:
                console.print("[yellow]Usage: /provider add <name> <base_url> [api_key][/yellow]")
            else:
                pname = parts[1]
                url = parts[2]
                key = parts[3] if len(parts) > 3 else ""
                import tomllib, tomli_w
                cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
                with open(cfg_path, "rb") as f:
                    cfg = tomllib.load(f)
                cfg["model"][pname] = {"api_key": key, "base_url": url, "model": "gpt-4o"}
                with open(cfg_path, "w") as f:
                    tomli_w.dump(cfg, f)
                console.print(f"[green]Added provider: {pname}[/green]")
                console.print("[dim]To use it: /provider set {pname}[/dim]")
        elif cmd == "/project":
            import subprocess, json
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
                    except:
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
        elif cmd == "/cost":
            console.print(f"[bold cyan]Session Cost[/bold cyan]")
            try:
                t = self.session.total_tokens
                provider_name = getattr(self.provider, "name", "?")
                console.print(f"  Provider:  {provider_name}")
                inp = t.get("input", 0)
                out = t.get("output", 0)
                console.print(f"  Input:     {inp:,} tokens")
                console.print(f"  Output:    {out:,} tokens")
                console.print(f"  Total:     {inp+out:,} tokens")
                # Cost estimates per 1M tokens
                rates = {
                    "gpt-4o": (2.50, 10.00),
                    "gpt-4o-mini": (0.15, 0.60),
                    "claude": (3.00, 15.00),
                    "gemini": (0.10, 0.40),
                    "deepseek": (0.14, 0.28),
                    "qwen": (0.0, 0.0),
                    "llama": (0.0, 0.0),
                }
                pname = provider_name.lower()
                est = 0.0
                for key, (r_in, r_out) in rates.items():
                    if key in pname:
                        est = (inp/1000000*r_in + out/1000000*r_out)
                        break
                if est > 0:
                    console.print(f"  Est cost:  ${est:.6f}")
                else:
                    console.print(f"  Cost:      [green]local (free)[/green]")
                # Messages count
                console.print(f"  Messages:  {len(self.session.messages)}")
            except Exception as e:
                console.print(f"  [dim]No session data[/dim]")
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
