"""RAG + Self-Learning engine for Neural."""
import json, os, re, time, math
from pathlib import Path
from collections import Counter

KDIR = Path(__file__).parent / "knowledge"
KDIR.mkdir(parents=True, exist_ok=True)

# ── Bundled metadata & dataset ─────────────────────────────
_BUNDLED = None

def _load_bundled():
    global _BUNDLED
    if _BUNDLED is not None:
        return _BUNDLED
    import json
    ds_dir = Path(__file__).parent / "datasets"
    meta_file = ds_dir / "metadata.json"
    ds_file = ds_dir / "tool_calling.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else []
    data = json.loads(ds_file.read_text()) if ds_file.exists() else []
    _BUNDLED = {"metadata": meta, "dataset": data}
    return _BUNDLED

def get_bundled_stats():
    b = _load_bundled()
    return f"Bundled: {len(b['metadata'])} metadata + {len(b['dataset'])} dataset samples"

def search_bundled(query, k=5):
    """Search bundled metadata for matching intents."""
    b = _load_bundled()
    if not b["metadata"]:
        return []
    q = query.lower()
    from knowledge import BM25
    docs = [m["intent"] + " " + " ".join(m.get("tags", [])) for m in b["metadata"]]
    bm = BM25(docs)
    results = bm.search(q, k)
    if not results or not results[0]:
        return []
    matched = []
    for score, doc in results:
        idx = docs.index(doc)
        matched.append((score, b["metadata"][idx]))
    return matched

import threading as _threading
_lock = _threading.Lock()

def _load(n):
    p = KDIR / n
    with _lock:
        return json.loads(p.read_text()) if p.exists() else []

def _save(n, d):
    with _lock:
        (KDIR / n).write_text(json.dumps(d, indent=2, default=str))

class BM25:
    def __init__(self, docs):
        self.docs = docs; self.N = len(docs)
        self.k1 = 1.5; self.b = 0.75; self.avgdl = 1
        self.idf = {}; self._build()
    def _tok(self, t):
        return re.findall(r"\w+", t.lower())
    def _build(self):
        if not self.docs: return
        self.avgdl = sum(len(self._tok(d)) for d in self.docs) / self.N
        df = Counter()
        for d in self.docs:
            for t in set(self._tok(d)): df[t] += 1
        for t, n in df.items():
            self.idf[t] = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
    def score(self, q, i):
        qt = self._tok(q); dt = self._tok(self.docs[i])
        tf = Counter(dt); dl = len(dt); s = 0
        for t in qt:
            if t in self.idf:
                f = tf.get(t, 0)
                s += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s
    def search(self, q, k=5):
        sc = [(self.score(q,i),self.docs[i]) for i in range(self.N)]
        sc.sort(key=lambda x: -x[0])
        return [s for s in sc if s[0] > 0][:k]

def get_facts(): return _load("facts.json")
def get_skills(): return _load("skills.json")
def get_stats(): return _load("stats.json")
def add_fact(topic, content, source=""):
    facts = get_facts()
    for f in facts:
        if f["topic"].lower() == topic.lower():
            f["content"] = content
            f["updated"] = time.time()
            _save("facts.json", facts)
            return f"Updated: {topic}"
    facts.append({"topic": topic, "content": content,
                  "source": source, "created": time.time()})
    _save("facts.json", facts)
    return f"Learned: {topic}"

def add_skill(name, pattern, tool, args_tmpl):
    skills = get_skills()
    for s in skills:
        if s["name"] == name and s["tool"] == tool: return  # dedup
    skills.append({"name": name, "pattern": pattern, "tool": tool,
                   "args_tmpl": args_tmpl, "created": time.time(), "ok": 1})
    _save("skills.json", skills)

def search_knowledge(query, k=5):
    facts = get_facts()
    skills = get_skills()
    docs = [f"{f['topic']}: {f['content']}" for f in facts]
    docs += [f"{s['name']}: {s['pattern']}" for s in skills]
    if not docs:
        return ""
    
    # BM25 search
    bm = BM25(docs)
    bm_results = bm.search(query, k)
    
    # Vector search (if available)
    vector_results = []
    try:
        from vectordb import search as vec_search
        vector_results = vec_search(query, k)
    except Exception:
        pass
    
    # Hybrid: combine both results
    seen = set()
    combined = []
    
    for sc, doc in bm_results:
        if doc not in seen:
            combined.append((sc, doc, "bm25"))
            seen.add(doc)
    
    for sim, doc, meta in vector_results:
        if doc not in seen:
            combined.append((sim, doc, "vector"))
            seen.add(doc)
    
    # Also search bundled metadata
    try:
        for sc, m in search_bundled(query, k):
            doc = f"[{m['tool']}] {m['intent']}"
            if doc not in seen:
                combined.append((sc, doc, "bundle"))
                seen.add(doc)
    except:
        pass
    
    combined.sort(key=lambda x: -x[0])
    
    lines = ["## Knowledge Context"]
    for score, doc, method in combined[:k]:
        tag = "🔤" if method == "bm25" else "🧠"
        lines.append(f"[{tag}{score:.2f}] {doc}")
    return "\n".join(lines)

# ── Learning quality filters ──────────────────────────────────
_MIN_TERM_LEN = 8
_MAX_FACTS_PER_TURN = 2
_SKIP_TERMS = {"Error", "Result", "True", "False", "None", "Step", "User",
               "System", "Please", "Note", "Warning", "Important", "Example",
               "Output", "Input", "Thank", "Here", "This", "That"}
_TRIVIAL_TOOLS = {"notify"}

def _is_meaningful_term(term: str) -> bool:
    if len(term) < _MIN_TERM_LEN: return False
    if term in _SKIP_TERMS: return False
    words = term.split()
    if len(words) == 1 and len(term) < 12: return False
    return True

def learn_from_interaction(usr, out, tool_history, success=True):
    if not success or not out or len(out) < 50: return
    # Extract multi-word Title Case phrases (e.g. "Docker Container")
    terms = re.findall(r"\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})+)\b", out)
    saved = 0
    for t in terms:
        if saved >= _MAX_FACTS_PER_TURN: break
        if _is_meaningful_term(t):
            add_fact(t, out[:150], source=usr[:50])
            saved += 1
    # Save skills for non-trivial tools
    for tn, ta, res in tool_history[-3:]:
        if tn not in _TRIVIAL_TOOLS and "error" not in res.lower()[:50] and len(res) > 20:
            add_skill(f"use_{tn}", usr[:80], tn, ta)
    s = get_stats()
    s["total_sessions"] = s.get("total_sessions", 0) + 1
    s["total_tools"] = s.get("total_tools", 0) + len(tool_history)
    _save("stats.json", s)

def knowledge_summary():
    facts = get_facts()
    skills = get_skills()
    stats = get_stats()
    txt = f"Facts: {len(facts)} | Skills: {len(skills)} | "
    txt += f"Sessions: {stats.get('total_sessions', 0)}"
    return txt
