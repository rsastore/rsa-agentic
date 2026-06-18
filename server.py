"""Agent Server — REST API for Neural."""
import os, json, asyncio

_app = None
_session = None

def _get_session(provider, config):
    global _session
    if _session is None and provider:
        from agent import AgentSession
        _session = AgentSession(provider, config or {})
    return _session

def create_app(provider, config=None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    app = FastAPI(title="Neural API")
    _get_session(provider, config)

    @app.get("/status")
    async def status():
        from tools.builtin import list_tools
        from knowledge import knowledge_summary
        return {"status":"ok","tools":list_tools(),"knowledge":knowledge_summary()}

    @app.post("/chat")
    async def chat(body: dict):
        msg = body.get("message","")
        if not msg:
            raise HTTPException(400,"message required")
        async def gen():
            for ev in _session.run_stream(msg):
                yield json.dumps(ev)+"\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/chat/sync")
    async def chat_sync(body: dict):
        msg = body.get("message","")
        if not msg:
            raise HTTPException(400,"message required")
        return {"response": _session.run(msg)}

    @app.post("/plan")
    async def plan(body: dict):
        from planner import PlannerAgent
        p = PlannerAgent(_session)
        results = list(p.run_with_plan(body.get("goal","")))
        return {"results": results}

    @app.get("/knowledge")
    async def knowledge():
        from knowledge import get_facts, get_skills, knowledge_summary
        return {"summary":knowledge_summary(),"facts":get_facts()[-10:],"skills":get_skills()[-10:]}

    @app.post("/reset")
    async def reset():
        if _session: _session.reset()
        return {"status":"reset"}

    return app

def run_server(host="0.0.0.0", port=8765, provider=None, config=None):
    import uvicorn
    app = create_app(provider, config)
    print(f"Neural API @ http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
