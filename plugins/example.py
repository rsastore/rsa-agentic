def register(reg):
    from plugin_loader import PluginTool
    reg(PluginTool("hello_plugin", lambda **kw: f"Hello {kw.get('name','?')} from plugin!", "Test plugin tool", {"name": "string"}))
    reg(PluginTool("calc", lambda **kw: str(eval(kw.get('expr','0'))), "Evaluate math expression", {"expr": "expression"}))
