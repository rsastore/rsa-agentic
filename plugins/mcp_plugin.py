import subprocess, json, os, sys, threading
from pathlib import Path

MCP_SERVERS = {}
MCP_TOOLS = {}

def _read_message(proc):
    """Read a JSON-RPC message from MCP server (Content-Length framing)."""
    header = b""
    while True:
        c = proc.stdout.read(1)
        if not c:
            return None
        header += c
        if header.endswith(b"\r\n\r\n"):
            break
    try:
        clen = int(header.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        return json.loads(proc.stdout.read(clen).decode())
    except:
        return None

def _send_message(proc, msg):
    data = json.dumps(msg).encode()
    proc.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
    proc.stdin.write(data)
    proc.stdin.flush()

def connect_server(name, cmd, args=None):
    """Connect to an MCP server via stdio."""
    if args is None:
        args = []
    try:
        proc = subprocess.Popen([cmd] + list(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _send_message(proc, {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})
        resp = _read_message(proc)
        if resp and "error" not in resp:
            _send_message(proc, {"jsonrpc":"2.0","id":2,"method":"tools/list"})
            tools_resp = _read_message(proc)
            if tools_resp and "result" in tools_resp:
                tools = tools_resp["result"].get("tools", [])
                for t in tools:
                    mcp_name = t.get("name", "unknown")
                    tname = f"mcp_{name}_{mcp_name}"
                    MCP_TOOLS[tname] = {"server": name, "proc": proc, "mcp_name": mcp_name, "desc": t.get("description",""), "params": t.get("inputSchema",{}).get("properties",{})}
                MCP_SERVERS[name] = proc
                return {"status":"ok","tools":len(tools)}
        proc.terminate()
        return {"status":"error","msg":"handshake failed"}
    except Exception as e:
        return {"status":"error","msg":str(e)}

def _call_mcp(tool_name, **kwargs):
    info = MCP_TOOLS.get(tool_name)
    if not info:
        return f"Error: MCP tool {tool_name} not found"
    proc = info["proc"]
    mcp_name = info["mcp_name"]
    _send_message(proc, {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":mcp_name,"arguments":kwargs}})
    resp = _read_message(proc)
    if resp and "result" in resp:
        content = resp["result"].get("content", [])
        texts = [c["text"] for c in content if isinstance(c, dict) and c.get("type")=="text"]
        return "\n".join(texts) if texts else json.dumps(resp["result"])
    return "MCP error"

MCP_CONFIG = Path(os.path.expanduser("~/neural/plugins/mcp_servers.json"))
if MCP_CONFIG.exists():
    try:
        servers = json.loads(MCP_CONFIG.read_text())
        for name, cfg in servers.items():
            connect_server(name, cfg.get("cmd",""), cfg.get("args",[]))
    except: pass

def register(reg):
    from plugin_loader import PluginTool
    for tname, info in MCP_TOOLS.items():
        srv_name = info.get("server", "unknown")
        desc = info.get("desc", "") or f"MCP tool from {srv_name}"
        reg(PluginTool(tname, lambda **kw: _call_mcp(tname, **kw), desc, info["params"]))
    # Also register connect command as tool
    reg(PluginTool("mcp_connect", lambda srv,cmd,args="": json.dumps(connect_server(srv,cmd,args.split())), "Connect to an MCP server", {"srv":"server name","cmd":"command","args":"arguments (space-separated)"}))