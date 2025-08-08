"""Compatibility shim providing `AsyncOllamaService` used by tests.

Minimal in-memory async service exposing `create_session` and a `run_model`
that shells out via asyncio (and will be mocked in tests).
"""
from __future__ import annotations

import asyncio
from typing import Dict, Optional


# Simple registry without using the `global` statement
_registry: Dict[str, Optional["AsyncOllamaService"]] = {"current": None}


def register_current_service(service: "AsyncOllamaService") -> None:
    _registry["current"] = service


def current_service() -> Optional["AsyncOllamaService"]:
    return _registry["current"]


class AsyncOllamaService:
    def __init__(self) -> None:
        self.active_sessions: Dict[str, Dict[str, str]] = {}
        # Register this instance as the current service for test shims
        register_current_service(self)

    async def run_model(self, model_name: str, prompt: str) -> str:
        """Run a model via subprocess. In tests, this is mocked."""
        proc = await asyncio.create_subprocess_exec(
            "ollama", "run", model_name, prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()

    async def create_session(self, session_id: str, model_name: str) -> bool:
        if session_id in self.active_sessions:
            return False
        self.active_sessions[session_id] = {"model": model_name}
        return True


 
