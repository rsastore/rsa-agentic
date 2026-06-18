<div align="center">
  <h1>Neural</h1>
  <p><strong>Autonomous AI Agent — Terminal Native</strong></p>
  <p>
    <a href="#"><img src="https://img.shields.io/badge/python-3.11+-blue?logo=python" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/github/actions/workflow/status/neural/neural/ci.yml?branch=main" alt="CI"></a>
    <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  </p>
  Agent loop • Tool calling • Planner • MCP plugins • Docker sandbox • Multi-provider
</div>

## Features

| Feature | Status |
|---------|--------|
| Agent loop (plan->act->observe) | ✅ |
| Tool calling (shell, file, git, python) | ✅ |
| Sub-agents (parallel execution) | ✅ |
| Multi-provider (Ollama, OpenAI) | ✅ |
| Streaming output | ✅ |
| Edit file with diff preview | ✅ |
| Approval gates | ✅ |
| MCP plugin system | ✅ |
| Docker sandbox (safe execution) | ✅ |
| Planner engine (goal+steps+retry) | ✅ |
| Web browsing | ✅ |
| Context compaction | ✅ |
| Session persistence | ✅ |
| Android (Termux) | ✅ |

## Quick Install

```bash
git clone https://github.com/rsastore/rsa-agentic.git ~/neural
cd ~/neural
pip install prompt_toolkit rich requests
ollama pull qwen2.5:1.5b
./neural.py
```

## Android (Termux)

```bash
pkg install git cmake python ninja build-essential openblas
pip install prompt_toolkit rich requests
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENBLAS=ON ..
make -j4 llama-cli
mkdir -p ~/storage/models
cd ~/storage/models
wget -O qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
git clone https://github.com/rsastore/rsa-agentic.git ~/neural
cd ~/neural && python3 neural.py
```

## Docker Sandbox

Run exec_shell commands safely inside a container:

```bash
docker build -t neural-sandbox -f Dockerfile.sandbox .
neural --sandbox
```

## MCP Plugins

Connect any MCP server:

```json
// plugins/mcp_servers.json
{
  "sqlite": {
    "cmd": "uvx",
    "args": ["mcp-server-sqlite", "--db-path", "/tmp/test.db"]
  }
}
```

## Commands

/help /clear /reset /status /tools /plugins /plan plan /checklist /session /compact /save /exit

## Architecture

```
neural/
|- neural.py       Entry point
|- tui.py          Terminal UI
|- agent.py        Agent loop + sub-agents
|- planner.py      Planner engine (goal + steps + retry)
|- compact.py      Context compaction
|- sessions.py     Session persistence
|- config.toml     Configuration
|- models/         Multi-provider abstraction
|- tools/          Built-in tools
|- plugins/        MCP plugins
|- sessions/       Saved conversations
```

