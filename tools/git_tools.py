"""Git automation tools for Neural."""
import subprocess

def _git_status(path: str = "."):
    try:
        r = subprocess.run(["git","status","--short"], capture_output=True, text=True, timeout=10, cwd=path)
        return r.stdout if r.stdout else "Clean working tree"
    except: return "Not a git repo"

def _git_diff(path: str = "."):
    try:
        r = subprocess.run(["git","diff","--stat"], capture_output=True, text=True, timeout=10, cwd=path)
        staged = subprocess.run(["git","diff","--cached","--stat"], capture_output=True, text=True, timeout=10, cwd=path)
        out = ""
        if staged.stdout: out += f"Staged:\n{staged.stdout}\n"
        if r.stdout: out += f"Unstaged:\n{r.stdout}"
        return out if out else "No changes"
    except: return "Not a git repo"

def _git_commit(message: str, path: str = "."):
    try:
        subprocess.run(["git","add","-A"], capture_output=True, text=True, timeout=10, cwd=path)
        r = subprocess.run(["git","commit","-m",message], capture_output=True, text=True, timeout=10, cwd=path)
        if r.returncode == 0: return r.stdout
        return r.stderr or "Nothing to commit"
    except Exception as e: return f"Git error: {e}"

def _git_log(max_count: int = 5, path: str = "."):
    try:
        r = subprocess.run(["git","log","--oneline",f"-{max_count}"], capture_output=True, text=True, timeout=10, cwd=path)
        return r.stdout or "No commits"
    except: return "Not a git repo"

TOOLS = [
    ("git_status", _git_status, "Show working tree status", {"path": "repo path (optional)"}),
    ("git_diff", _git_diff, "Show staged/unstaged changes", {"path": "repo path (optional)"}),
    ("git_commit", _git_commit, "Stage all and commit", {"message": "commit message", "path": "repo path (optional)"}),
    ("git_log", _git_log, "Show recent commits", {"max_count": "number of commits (default 5)", "path": "repo path (optional)"}),
]
