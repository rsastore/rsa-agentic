"""Grammar-guided JSON: auto-fix tool calling from local LLMs."""
import re, json

COMMON_FIXES = [
    (r':\s*:', ':'),
    (r',\s*\}', '}'),
]

def extract_json(text):
    m = re.search(r'```(?:json)?\s*\n?(\{.*?\})\n?\s*```', text, re.DOTALL)
    if m: return m.group(1)
    start = text.find('{"tool"')
    if start < 0: start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            if depth == 0: return text[start:i+1]
    return ""

def auto_fix(text):
    for pat, repl in COMMON_FIXES:
        text = re.sub(pat, repl, text)
    return text

def parse_tool_call(text):
    js = extract_json(text)
    if not js: return None
    try:
        d = json.loads(js)
        if "tool" in d: return d
    except: pass
    try:
        d = json.loads(auto_fix(js))
        if "tool" in d: return d
    except: pass
    tm = re.search(r'"tool"\s*:\s*"([^"]+)"', js)
    am = re.search(r'"args"\s*:\s*(\{[^}]+\})', js)
    if tm:
        args = {}
        if am:
            try: args = json.loads(am.group(1))
            except: pass
        return {"tool": tm.group(1), "args": args}
    return None
