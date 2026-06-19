
"""SQLite storage — replace JSON files for sessions, knowledge, cost tracking."""
import sqlite3, json, time, os
from pathlib import Path

DB_PATH = Path(os.path.expanduser("~/rsa-agentic/data.db"))

def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db

def init():
    """Create tables if not exist."""
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            model TEXT,
            created_at REAL,
            message_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            export TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            tool_calls TEXT,
            created_at REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT UNIQUE,
            content TEXT,
            source TEXT,
            created_at REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            pattern TEXT,
            tool TEXT,
            args TEXT,
            created_at REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS cost_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            provider TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost REAL,
            created_at REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            cron_expr TEXT,
            goal TEXT,
            enabled INTEGER DEFAULT 1,
            last_run REAL,
            created_at REAL DEFAULT (strftime('%s','now'))
        );
    """)
    db.commit()
    db.close()

# ── Session Operations ──

def save_session(session_id, model, messages):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO sessions (id, model, created_at, message_count) VALUES (?,?,?,?)",
               (session_id, model, time.time(), len(messages)))
    db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    for m in messages:
        db.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                   (session_id, m.get("role"), m.get("content",""), time.time()))
    db.commit(); db.close()
    return session_id

def load_session(session_id):
    db = get_db()
    s = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not s: return None
    msgs = db.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    db.close()
    return {"id": s["id"], "model": s["model"], "messages": [dict(m) for m in msgs]}

def list_sessions(limit=10):
    db = get_db()
    rows = db.execute("SELECT id, model, message_count, created_at FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ── Knowledge Operations ──

def add_fact(topic, content, source=""):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO knowledge (topic, content, source) VALUES (?,?,?)",
               (topic, content, source))
    db.commit(); db.close()

def add_skill(name, pattern, tool, args_tmpl):
    db = get_db()
    db.execute("INSERT INTO skills (name, pattern, tool, args) VALUES (?,?,?,?)",
               (name, pattern, tool, json.dumps(args_tmpl)))
    db.commit(); db.close()

def get_all_knowledge():
    db = get_db()
    facts = db.execute("SELECT topic, content FROM knowledge ORDER BY created_at DESC").fetchall()
    skills = db.execute("SELECT name, pattern, tool FROM skills ORDER BY created_at DESC").fetchall()
    db.close()
    return [dict(f) for f in facts], [dict(s) for s in skills]

# ── Cost Log ──

def log_cost(session_id, provider, inp, out, cost):
    db = get_db()
    db.execute("INSERT INTO cost_log (session_id, provider, input_tokens, output_tokens, cost) VALUES (?,?,?,?,?)",
               (session_id, provider, inp, out, cost))
    db.execute("UPDATE sessions SET input_tokens=input_tokens+?, output_tokens=output_tokens+?, cost=cost+? WHERE id=?",
               (inp, out, cost, session_id))
    db.commit(); db.close()

def get_cost_summary(limit=10):
    db = get_db()
    rows = db.execute("""
        SELECT session_id, provider, SUM(input_tokens) as inp, SUM(output_tokens) as out, SUM(cost) as cost
        FROM cost_log GROUP BY session_id ORDER BY MAX(created_at) DESC LIMIT ?
    """, (limit,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ── Scheduled Tasks ──

def add_task(name, cron_expr, goal):
    db = get_db()
    db.execute("INSERT INTO scheduled_tasks (name, cron_expr, goal) VALUES (?,?,?)",
               (name, cron_expr, goal))
    db.commit(); db.close()

def get_due_tasks():
    """Get tasks that are due to run (simplified: just return all enabled)."""
    db = get_db()
    rows = db.execute("SELECT * FROM scheduled_tasks WHERE enabled=1").fetchall()
    db.close()
    return [dict(r) for r in rows]

# Init on import
init()
