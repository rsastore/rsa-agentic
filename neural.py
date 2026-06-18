#!/usr/bin/env python3
"""Neural — Autonomous AI Agent TUI.

Usage:
    python neural.py                          # Start TUI
    python neural.py --cli "list files"       # One-shot CLI mode
    python neural.py --version                # Show version
"""

import sys, os, json, argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Neural — Autonomous AI Agent")
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
        print("Config not found. Run from ~/neural/ directory.")
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
        cfg = tomllib.load(open(os.path.expanduser("~/neural/config.toml"),"rb"))
        provider = create_provider(cfg.get("model",{}))
        run_server(provider=provider, config=cfg.get("agent",{}))
    else:
        main()
