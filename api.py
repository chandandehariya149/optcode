"""
api.py
-------
Day 7: Wraps our compression pipeline as a real web API.

Instead of running a script, this starts a live server that
any app can send data to and get back a compressed result.
This is the actual "product" shape - a hosted service, not a script.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from pipeline import run_pipeline

app = FastAPI(title="Token Harness API", description="Compresses LLM context to save tokens")


class Message(BaseModel):
    role: str
    content: str


class CompressRequest(BaseModel):
    system_prompt: str
    conversation: List[Message]
    tool_result: Optional[Dict[str, Any]] = None
    tool_keep_keys: Optional[List[str]] = []
    keep_recent: Optional[int] = 4


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is alive."""
    return {"status": "ok", "message": "Token Harness API is running"}


@app.post("/compress")
def compress(request: CompressRequest):
    """
    Main endpoint: takes a system prompt, conversation, and optional
    tool result, runs them through the full compression pipeline,
    and returns before/after token stats plus the compressed content.
    """
    conversation_dicts = [m.dict() for m in request.conversation]
    tool_result = request.tool_result or {}
    tool_keep_keys = request.tool_keep_keys or []

    result = run_pipeline(
        system_prompt=request.system_prompt,
        conversation=conversation_dicts,
        tool_result=tool_result,
        tool_keep_keys=tool_keep_keys,
        keep_recent=request.keep_recent,
    )

    return result