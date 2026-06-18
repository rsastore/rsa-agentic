#!/usr/bin/env python3
"""Neural — Autonomous AI Agent TUI.

Usage:
    python neural.py                          # Start TUI
    python neural.py --cli "list files"       # One-shot CLI mode
    python neural.py --version                # Show version
"""

import sys, os, json, argparse
from pathlib import Path

# Python 3.10 compatibility
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pip install tomli

# Auto-cd to project root (works on Colab, VPS, etc.)
_PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(str(_PROJECT_ROOT))

def main():
    # Check first-run: warn if no providers configured
    import tomllib
    cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
    try:
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
        provider = cfg.get("model", {}).get("provider", "ollama")
        if provider == "openai":
            key = cfg.get("model", {}).get("openai", {}).get("api_key", "")
            if not key:
                print("\033[33m⚠️  OpenAI configured but no API key set!\033[0m")
                print("\033[33m   Set it: /provider key openai sk-xxxxxxxx\033[0m\n")
    except:
        pass

    # Safety: warn if running as root
    if os.geteuid() == 0:
        print("\033[33m⚠️  Running as root — dangerous operations will NOT auto-approve!\033[0m")
        print("\033[33m   Consider creating a non-root user for regular use.\033[0m\n")
    parser = argparse.ArgumentParser(description="RSA Agentic — Autonomous AI Agent")
    parser.add_argument("--cli", "-c", type=str, help="Run one-shot command (no TUI)")
    parser.add_argument("--model", "-m", type=str, help="Override model name")
    parser.add_argument("--version", "-v", action="store_true", help="Show version")
    args = parser.parse_args()

    if args.version:
        print("Neural v0.1.0")
        return

    # Load config
    cfg_path = Path.home() / "neural" / "config.toml"
    if not cfg_path.exists():
        print("Config not found. Run from ~/rsa-agentic/ directory.")
        return

    import tomllib
    config = tomllib.loads(cfg_path.read_text())

    # Override model
    if args.model:
        config["model"]["model_name"] = args.model

    if args.cli:
        run_cli(config, args.cli)
    else:
        run_tui(config)

def run_tui(config):
    from tui import NeuralTUI
    app = NeuralTUI(config)
    app.run()

def run_cli(config: dict, prompt: str):
    from agent import AgentSession
    from models.providers import create_provider
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    provider = create_provider(config.get("model", {}))
    session = AgentSession(provider, config.get("agent", {}))
    console.print(f"[cyan]Neural[/cyan] — {provider.name}")
    console.print(f"[dim]Input: {prompt}[/dim]\n")

    result = session.run(prompt)
    console.print(Markdown(result))

if __name__ == "__main__":
    if "--server" in __import__("sys").argv:
        from server import run_server
        import tomllib
        from models.providers import create_provider
        cfg_path = os.path.expanduser("~/rsa-agentic/config.toml")
        cfg = tomllib.load(open(cfg_path,"rb"))
        provider = create_provider(cfg.get("model",{}))
        run_server(provider=provider, config=cfg.get("agent",{}))
    else:
        main()
