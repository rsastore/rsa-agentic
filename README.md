# RSA Agentic

Autonomous AI Agent — Terminal Native. Local LLM or Cloud API.

> Agent loop • Planner • RAG • MCP plugins • Docker sandbox • Multi-provider

---

## Quick Start

```bash
git clone https://github.com/rsastore/rsa-agentic.git
cd rsa-agentic

# Update (get latest features + bug fixes)
git pull

# That's it! Just type:
rsa

# Local (Ollama)
pip install prompt_toolkit rich requests
ollama pull qwen2.5:1.5b
python3 neural.py

# Cloud API
/provider key openai sk-xxxxx
/provider set openai
python3 neural.py
```

## macOS Setup

```bash
# Install dependencies
brew install python3 ollama gh
pip3 install prompt_toolkit rich requests

# Pull a model
ollama pull qwen2.5:1.5b

# Clone & run
git clone https://github.com/rsastore/rsa-agentic.git
cd rsa-agentic
python3 neural.py
```

> Note: On macOS, Neural auto-detects `.zsh_history` and works with
> Homebrew-installed Ollama, Docker Desktop, and GitHub CLI.


## Platform Support

| Platform | Status | Install |
|----------|--------|---------|
| **Linux** (Debian/Ubuntu) | ✅ Primary | `install.sh` |
| **macOS** | ✅ | `brew install python3 ollama` |
| **Windows** | ✅ (beta) | `install.ps1` |
| **Android** (Termux) | ✅ | Manual setup |

### Windows
```powershell
# PowerShell (Admin)
pip install prompt_toolkit rich requests
git clone https://github.com/rsastore/rsa-agentic.git
cd rsa-agentic
python neural.py
```

> Note: exec_shell uses cmd.exe on Windows.
> Install Ollama for Windows from https://ollama.com/download/windows

### Android (Termux)
```bash
pkg install git python
pip install prompt_toolkit rich requests
git clone https://github.com/rsastore/rsa-agentic.git
cd rsa-agentic && python neural.py
```

No root required. Runs in Termux with local models via llama.cpp.

## Features

| Area | Feature | Description |
|------|---------|-------------|
| **Core** | Agent Loop | Plan → Act → Observe cycle with tool calling |
| | Planner | Break goals into steps, execute, retry on failure |
| | Sub-Agents | Parallel execution of independent tasks |
| | Streaming | Real-time token-by-token output |
| **Models** | Ollama (local) | Default, works out of box |
| | OpenAI | GPT-4o, GPT-4o-mini |
| | Anthropic | Claude Sonnet 4, Haiku 3 |
| | Google | Gemini Flash, Pro |
| | Auto-compat | OpenRouter, DeepSeek, Groq, xAI, Together |
| | Model Manager | /hf search, /hf pull, /model switch |
| **Tools (18)** | Shell | exec_shell, sandbox_exec (Docker) |
| | File | read_file, write_file, edit_file (diff), list_dir |
| | Search | grep_files, web_fetch, web_search |
| | Git | git_status, git_diff, git_commit, git_log |
| | GitHub | github_issue, github_pr, github_search |
| | Code | python_exec |
| | Utility | notify (desktop notification) |
| | AI | fine_tune (auto-generate training script) |
| **Safety** | Approval Gates | Confirm before dangerous operations |
| | Destructive Block | rm -rf /, system file writes auto-blocked |
| | Root Warning | Warning when running as root |
| | Sandbox | Docker-based isolated execution |
| **Knowledge** | RAG | BM25 search across learned facts + skills |
| | Self-Learning | Auto-extract knowledge from interactions |
| | Dataset Manager | /dataset pull, /dataset learn (Nemotron, etc) |
| | Vector DB | Chroma integration for proper embeddings |
| **Planning** | Planner Engine | Goal → Steps → Execute → Retry → Complete |
| | Failure Memory | Remember what failed and why |
| | Auto-Reference | Auto-analyze GitHub repos for solutions |
| | Personas | coder, sysadmin, research modes |
| **Terminal** | Context Awareness | Auto-detect CWD, git branch, OS |
| | File Explorer | /explorer interactive browsing |
| | Project Detect | /project auto-detect project type |
| | Cost Tracking | /cost token usage + $ estimates |
| | SQLite Storage | Sessions, knowledge, cost history in DB |
| | Session Manager | Save, load, export conversations |
| | Context Compaction | /compact summarize long context |
| **Integration** | MCP Plugins | Model Context Protocol support |
| | Plugin System | Load custom tools from plugins/ |
| | REST API | Server mode: POST /chat, /plan, /status |
| | GitHub Tools | Create issues, PRs, search repos |
| | Scheduled Tasks | /schedule add, /schedule run |
| | Reference Analyzer | /reference clone + analyze any GitHub repo |
## Commands Reference

### Basic
| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear screen |
| `/reset` | Reset conversation |
| `/status` | Session info + tools count |
| `/tools` | List available tools |
| `/exit` | Quit |

### Model & Provider
| Command | Description |
|---------|-------------|
| `/provider` | List providers and API key status |
| `/provider set <name>` | Switch provider (ollama, openai, anthropic, google) |
| `/provider key <name> <key>` | Set API key for a provider |
| `/provider add <name> <url> <key>` | Add custom OpenAI-compatible provider |
| `/models` | List installed models |
| `/model <name>` | Switch to a model |
| `/hf search <query>` | Search HuggingFace for GGUF models |
| `/hf pull <id>` | Download model from HuggingFace |

### Knowledge & Learning
| Command | Description |
|---------|-------------|
| `/knowledge` | Show learned facts and skills |
| `/forget` | Clear all learned knowledge |
| `/nemotron` | One-click download + learn from Nemotron dataset |
| `/vectordb` | Rebuild vector index with embeddings |
| `/dataset list` | List downloaded datasets |
| `/dataset pull <name>` | Download a HuggingFace dataset |
| `/dataset learn <name>` | Extract patterns from dataset |
| `/dataset search <query>` | Find datasets on HuggingFace |

### Planning & Execution
| Command | Description |
|---------|-------------|
| `/plan <goal>` | Create plan, execute steps with retry |
| `/checklist` | Show task list |
| `/checklist add <task>` | Add task manually |
| `/checklist done <n>` | Mark task complete |

### Project & Files
| Command | Description |
|---------|-------------|
| `/project` | Auto-detect project type, deps, git remote |
| `/tree [depth]` | Show project file tree |
| `/explorer` | Interactive file browser (ls, cat, up, q) |

### Agent Control
| Command | Description |
|---------|-------------|
| `/persona` | Show current mode |
| `/persona <mode>` | Switch mode (coder, sysadmin, research, default) |
| `/context` | Show terminal context (CWD, git, host) |
| `/compact` | Summarize and compress long context |
| `/memory` | Show working/episodic/preference memory |
| `/remember <fact>` | Add fact to working memory |
| `/pref key=value` | Save user preference |

### Cost & Schedule
| Command | Description |
|---------|-------------|
| `/cost` | Current session token usage |
| `/cost history` | Cost log across sessions |
| `/schedule` | List scheduled tasks |
| `/schedule add <name> <goal>` | Add a scheduled task |
| `/schedule run` | Execute all pending tasks |
| `/cost` | Session token & cost estimate |
| `/cost history` | Cost log across sessions |
| `/quant` | Auto-detect RAM & recommend best model |

### Integration
| Command | Description |
|---------|-------------|
| `/plugins` | List loaded plugin tools |
| `/reference <url>` | Analyze any GitHub repo |
| `/session list` | List saved sessions |
| `/session load <name>` | Load a session |
| `/save` | Force save current session |
| `/ft <dataset> [model]` | Generate fine-tuning script |

### Server Mode
| Command | Description |
|---------|-------------|
| `--server` | Start REST API server on port 8765 |
| `--cli <query>` | One-shot CLI mode (non-interactive) |
| `/vectordb` | Rebuild vector index with embeddings |
## Architecture

```
rsa-agentic/
├── neural.py              Entry point (CLI / TUI / Server)
├── tui.py                 Terminal UI (prompt_toolkit + rich)
├── agent.py               Agent loop + tool calling + sub-agents
├── planner.py             Planner engine (goal → steps → retry)
├── knowledge.py           RAG + self-learning (BM25)
├── context.py             Terminal context + personas
├── compact.py             Context compaction
├── sessions.py            Session persistence
├── db.py                  SQLite storage
├── server.py              REST API server
├── reference.py           GitHub repo analyzer
├── hf_manager.py          HuggingFace model + dataset manager
├── vectordb.py            Chroma vector DB adapter
├── plugin_loader.py       Plugin discovery
├── config.toml            Configuration
├── system.md              System prompt template
├── SDK.md                 Plugin development guide
│
├── models/
│   ├── base.py            Abstract provider
│   └── providers.py       Ollama, OpenAI, Anthropic, Google
│
├── tools/
│   ├── builtin.py         18 built-in tools
│   └── git_tools.py       Git automation
│
├── plugins/               Custom tools (MCP + Python)
├── knowledge/             Facts + skills (learned data)
├── datasets/              Downloaded HF datasets
└── sessions/              Saved conversations
```
## Examples

### System Admin
```
/persona sysadmin
/plan check disk, ram, and generate report
→ Neural runs df -h, free -h, top
→ Writes report to report.md
→ Done
```

### Coding
```
/persona coder
/project
→ Auto-detects Node.js / Python / Rust project
→ Shows deps, git remote

/plan "add login feature"
→ Neural reads existing code
→ Creates new files
→ Commits with git_commit
```

### Research
```
/persona research
/reference https://github.com/rsastore/rsa-agentic
→ Clones, analyzes structure, compares features
→ Injects findings into knowledge

/nemotron
→ Auto-downloads 12k agent examples from DeepSeek V3.2
→ Extracts tool patterns → Neural learns
```

### API Integration
```
# Terminal
/provider key openai sk-xxxxx
/provider set openai
→ Now using GPT-4o

# From another app (Server mode)
neural --server &
curl http://localhost:8765/chat -d '{"message":"check disk"}'
```

> **⚠️ Security:** Never commit `config.toml` to GitHub if it contains API keys.
> Use `/provider key <name> <key>` to set keys instead of editing config.toml directly.
> Add `config.toml` to your local `.gitignore` if needed.
