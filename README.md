# Neural — Autonomous AI Agent

> Terminal-based AI agent with tool calling, multi-provider, sub-agents, and MCP plugin system.
> Inspired by DeepSeek TUI, OpenHands, Continue, Claude Code, GPT Codex, and Goose.

## Quick Start (VPS)

```bash
cd ~/neural
python3 neural.py           # TUI mode
python3 neural.py -cli "check disk"   # CLI mode
```

## Deploy to Android (Termux)

### Requirements
- Realme 13+ (6GB+ RAM)
- Termux from F-Droid
- 2GB free storage

### Setup

```bash
# 1. Install
pkg install git cmake python ninja build-essential openblas
pip install prompt_toolkit rich requests

# 2. Build llama.cpp
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENBLAS=ON -DLLAMA_NATIVE=ON ..
make -j4 llama-cli

# 3. Download model
mkdir -p ~/storage/models
cd ~/storage/models
wget -O qwen2.5-1.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 4. Clone + run
git clone <your-repo> ~/neural
cd ~/neural
python3 neural.py
```

### Performance (Realme 13+)

| Model | Speed | RAM |
|-------|-------|-----|
| Qwen2.5-1.5B Q4 | 8-15 tok/s | 2-3GB |
| Llama-3.2-1B Q4 | 12-20 tok/s | 1.5-2GB |

## Architecture

```
neural/
├── neural.py          # Entry point (CLI / TUI)
├── tui.py             # Terminal UI (prompt_toolkit + rich)
├── agent.py           # Agent loop + sub-agents
├── sessions.py        # Session persistence
├── config.toml        # Configuration
├── system.md          # System prompt
├── models/
│   ├── base.py        # Abstract provider
│   └── providers.py   # Ollama, OpenAI, llama.cpp
├── tools/
│   └── builtin.py     # Shell, file, git, python tools
├── sessions/          # Saved conversations
└── plugins/           # MCP plugins (coming soon)
```

## Commands

| Command | Action |
|---------|--------|
| /help | Show help |
| /clear | Clear screen |
| /reset | Reset conversation |
| /status | Session info |
| /tools | List tools |
| /history | Recent sessions |
| /save | Save session |
| /exit | Quit |
