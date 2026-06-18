import importlib.util, sys, os, json
from pathlib import Path

PLUGIN_DIR = Path(os.path.expanduser("~/neural/plugins"))

class PluginTool:
    def __init__(self, n, fn, d, p):
        self.name=n; self.fn=fn; self.desc=d; self.params=p
    def __call__(self,**kw):
        try: return str(self.fn(**kw))
        except Exception as e: return f"Plugin error: {e}"

_reg={};_ld=[]
def register(t): _reg[t.name]=t
def get_tool(n): return _reg.get(n)
def list_tools(): return list(_reg.keys())
def list_loaded(): return _ld.copy()

def descriptions():
    lines = []
    for n, t in _reg.items():
        p = ", ".join(f"{k}: {v}" for k,v in t.params.items())
        lines.append(f"- {n}({p}): {t.desc}")
    return "\\n".join(lines)

def discover():
    _ld.clear()
    for f in sorted(PLUGIN_DIR.glob("*.py")):
        if f.name=="__init__.py": continue
        try:
            spec=importlib.util.spec_from_file_location(f"np_{f.stem}",f)
            if spec and spec.loader:
                m=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                if hasattr(m,"register"): m.register(register); _ld.append(f.stem)
        except: pass
