from __future__ import annotations
"""Memory system: working, episodic, preference memory for local LLMs."""
import json, os, time
from pathlib import Path

MEM_DIR = Path(__file__).parent / "memory"
MEM_DIR.mkdir(parents=True, exist_ok=True)

# ── Working Memory (mid-conversation scratchpad) ──

WORKING_FILE = MEM_DIR / "working.json"

def init_working():
    """Initialize or reset working memory for a new session."""
    data = {
        "key_facts": [],
        "current_task": "",
        "user_preferences": {},
        "session_start": time.time(),
    }
    WORKING_FILE.write_text(json.dumps(data, indent=2))
    return data

def load_working() -> dict:
    if WORKING_FILE.exists():
        try: return json.loads(WORKING_FILE.read_text())
        except: pass
    return init_working()

def update_working(key: str, value):
    data = load_working()
    data[key] = value
    WORKING_FILE.write_text(json.dumps(data, indent=2))

def add_fact(fact: str):
    """Add a key fact to working memory (max 10)."""
    data = load_working()
    if "key_facts" not in data:
        data["key_facts"] = []
    if fact not in data["key_facts"]:
        data["key_facts"].append(fact)
        data["key_facts"] = data["key_facts"][-10:]  # keep last 10
    WORKING_FILE.write_text(json.dumps(data, indent=2))

def working_context() -> str:
    """Build working memory context block for system prompt."""
    data = load_working()
    lines = ["## Working Memory"]
    kf = data.get("key_facts", [])
    if kf:
        lines.append("Key facts from this conversation:")
        for f in kf:
            lines.append(f"  - {f}")
    task = data.get("current_task", "")
    if task:
        lines.append(f"Current task: {task}")
    return "\n".join(lines)

# ── Episodic Memory (remember past sessions) ──

EPISODIC_FILE = MEM_DIR / "episodic.json"

def load_episodic() -> list:
    if EPISODIC_FILE.exists():
        try: return json.loads(EPISODIC_FILE.read_text())
        except: pass
    return []

def save_episodic(episodes: list):
    EPISODIC_FILE.write_text(json.dumps(episodes, indent=2, default=str))

def summarize_session(messages: list, summary: str):
    """Save a summary of a completed session."""
    eps = load_episodic()
    eps.append({
        "time": time.time(),
        "summary": summary[:300],
        "message_count": len(messages),
    })
    eps = eps[-20:]  # keep last 20 sessions
    save_episodic(eps)
    # Also inject into knowledge
    try:
        from knowledge import add_fact
        add_fact("Session Summary", summary[:200], source="episodic_memory")
    except: pass

def episodic_context() -> str:
    """Build episodic memory context for system prompt."""
    eps = load_episodic()
    if not eps: return ""
    lines = ["## Past Sessions"]
    for e in eps[-3:]:  # last 3 sessions
        lines.append(f"  - {e['summary'][:100]}")
    return "\n".join(lines)

# ── Preference Memory ──

PREF_FILE = MEM_DIR / "preferences.json"

def load_prefs() -> dict:
    if PREF_FILE.exists():
        try: return json.loads(PREF_FILE.read_text())
        except: pass
    return {}

def save_pref(key: str, value: str):
    prefs = load_prefs()
    prefs[key] = {"value": value, "updated": time.time()}
    PREF_FILE.write_text(json.dumps(prefs, indent=2, default=str))

def get_pref(key: str) -> str | None:
    prefs = load_prefs()
    d = prefs.get(key)
    return d["value"] if d else None

def preference_context() -> str:
    prefs = load_prefs()
    if not prefs: return ""
    lines = ["## User Preferences"]
    for k, v in prefs.items():
        lines.append(f"  {k}: {v['value']}")
    return "\n".join(lines)
