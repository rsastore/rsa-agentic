"""Dataset Manager — download, index, search, learn from HF datasets."""
import os, json, urllib.request, re
from pathlib import Path

DATA_DIR = Path(os.path.expanduser("~/rsa-agentic/datasets"))
INDEX_DIR = Path(os.path.expanduser("~/rsa-agentic/datasets/.index"))

def _api_url(dataset, file_path):
    return f"https://huggingface.co/datasets/{dataset}/resolve/main/{file_path}"

def list_datasets():
    """List downloaded datasets."""
    if not DATA_DIR.exists():
        return []
    datasets = {}
    for d in DATA_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            files = list(d.glob("*.jsonl"))
            total_lines = sum(1 for f in files for _ in open(f) if f.stat().st_size > 0) if files else 0
            datasets[d.name] = {"files": len(files), "samples": total_lines}
    return datasets

def pull_dataset(name, files=None):
    """Download a HuggingFace dataset. Returns list of status messages."""
    msgs = []
    safe_name = name.replace("/", "_")
    dest = DATA_DIR / safe_name
    dest.mkdir(parents=True, exist_ok=True)
    
    # Get file list from API
    try:
        url = f"https://huggingface.co/api/datasets/{name}"
        req = urllib.request.Request(url, headers={"User-Agent": "Neural"})
        with urllib.request.urlopen(req, timeout=15) as r:
            meta = json.loads(r.read())
        siblings = meta.get("siblings", [])
        jsonl_files = [s.get("rfilename","") for s in siblings if s.get("rfilename","").endswith(".jsonl")]
        if not jsonl_files:
            return ["No JSONL files found in dataset"]
        msgs.append(f"Found {len(jsonl_files)} JSONL files")
    except Exception as e:
        return [f"API error: {e}"]

    # Download each JSONL file
    total_samples = 0
    for jf in jsonl_files[:10]:  # Max 10 files
        dest_file = dest / jf.replace("/", "_")
        if dest_file.exists():
            count = sum(1 for _ in open(dest_file))
            total_samples += count
            msgs.append(f"  Already: {jf} ({count} samples)")
            continue
        dl_url = _api_url(name, jf)
        msgs.append(f"  Downloading: {jf}...")
        try:
            req = urllib.request.Request(dl_url, headers={"User-Agent": "Neural"})
            with urllib.request.urlopen(req, timeout=120) as r:
                count = 0
                with open(dest_file, "w") as fout:
                    for line in r:
                        fout.write(line.decode())
                        count += 1
                total_samples += count
                msgs.append(f"  Downloaded: {jf} ({count} samples)")
        except Exception as e:
            msgs.append(f"  Error: {e}")
    msgs.append(f"\nDone: {total_samples} total samples")
    return msgs

def search_dataset(name, query, limit=5):
    safe_name = name.replace("/", "_")
    data_dir = DATA_DIR / safe_name
    if not data_dir.exists():
        return {"error": f"Dataset not downloaded"}
    results = []
    q = query.lower()
    for jf in data_dir.glob("*.jsonl"):
        for line in open(jf):
            try:
                entry = json.loads(line)
                for m in entry.get("messages", []):
                    c = m.get("content", "")
                    if isinstance(c, str) and q in c.lower():
                        results.append({"domain":entry.get("domain","?"), "snippet":c[:200]})
                        break
                if len(results) >= limit: break
            except: pass
        if len(results) >= limit: break
    return {"results": results, "total": len(results)}

def learn_from_dataset(name, limit=50):
    safe_name = name.replace("/", "_")
    data_dir = DATA_DIR / safe_name
    if not data_dir.exists():
        return {"error": "Not downloaded"}
    from knowledge import add_skill, add_fact
    patterns = 0; domains = set()
    for jf in data_dir.glob("*.jsonl"):
        for i, line in enumerate(open(jf)):
            if i >= limit: break
            try:
                e = json.loads(line)
                d = e.get("domain", "?")
                domains.add(d)
                tools = [t.get("function",{}).get("name","?") for t in e.get("tools",[])[:3]]
                first = ""
                for m in e.get("messages",[]):
                    if m.get("role")=="user":
                        first = m.get("content","")[:80]
                        break
                if tools and first:
                    add_skill(f"nemotron_{d[:15]}_{i}", f"[{d}] {first[:50]} -> {tools[0]}", tools[0], {})
                    patterns += 1
            except: pass
    add_fact("Nemotron", f"{patterns} patterns from {len(domains)} domains")
    return {"patterns": patterns, "domains": len(domains)}


# ── Model Management ──

MODELS_DIR = __import__('pathlib').Path(os.path.expanduser('~/rsa-agentic/models_data'))

def list_installed():
    models = []
    try:
        r = __import__('subprocess').run(['ollama','list'], capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().split('\n')[1:]:
            if line.strip():
                parts = line.split()
                if parts:
                    models.append({'name': parts[0], 'size': parts[2] if len(parts)>2 else '?','backend':'ollama'})
    except:
        pass
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob('*.gguf'):
            size = f.stat().st_size / 1024 / 1024
            models.append({'name': f.stem, 'size': f'{size:.0f}MB', 'backend':'gguf'})
    return models

def search_hf(query, limit=10):
    import urllib.parse
    url = f'https://huggingface.co/api/models?search={urllib.parse.quote(query)}&sort=downloads&direction=-1&limit={limit}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'RSA-Agentic'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        return [{'id': m.get('modelId', m.get('id','')), 'downloads': m.get('downloads',0)} for m in data]
    except Exception as e:
        return [{'error': str(e)}]

def pull_model(model_id):
    msgs = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    short = model_id.split('/')[-1].replace('-Instruct-GGUF','').replace('-GGUF','')
    try:
        msgs.append(f'Pulling {short}...')
        r = __import__('subprocess').run(['ollama','pull',short], capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            msgs.append(f'Done: {short}')
            return msgs
    except:
        pass
    msgs.append(f'Try: ollama pull {short}')
    return msgs

def use_model(model_name):
    import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
    fp = os.path.expanduser('~/rsa-agentic/config.toml')
    with open(fp, 'rb') as f:
        cfg = tomllib.load(f)
    cfg['model']['model_name'] = model_name
    import tomli_w
    with open(fp, 'w') as f:
        tomli_w.dump(cfg, f)
    return f'Switched to {model_name}'


def search_hf_datasets(query, limit=10):
    """Search HuggingFace datasets."""
    import urllib.parse
    url = f"https://huggingface.co/api/datasets?search={urllib.parse.quote(query)}&sort=downloads&direction=-1&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"RSA-Agentic"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        results = []
        for d in data:
            did = d.get("id", d.get("dataset", "?"))
            downloads = d.get("downloads", 0)
            tags = d.get("tags", [])
            # Check if has JSONL files
            siblings = d.get("siblings", [])
            has_jsonl = any(s.get("rfilename","").endswith(".jsonl") for s in siblings)
            results.append({
                "id": did,
                "downloads": downloads,
                "has_jsonl": has_jsonl,
                "tags": [t for t in tags[:3] if t],
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]

