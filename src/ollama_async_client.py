"""Test-focused compatibility client.

Implements the minimal surface used by tests without requiring a running gRPC
server. `run_model` shells out via asyncio (and is patched in tests).
`create_session` updates the in-memory AsyncOllamaService instance.
"""
from __future__ import annotations

import asyncio


class AsyncOllamaClient:
    def __init__(self, host: str = "localhost", port: int = 50052) -> None:
        self.host = host
        self.port = port

    async def run_model(self, model_name: str, prompt: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "run", model_name, prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()

    async def create_session(self, session_id: str, model_name: str) -> bool:
        # Import here to avoid circular import at module load
        from .ollama_async_server import current_service

        svc = current_service()
        if svc is None:
            # No service registered; behave like a client-only success
            return True
        if session_id in svc.active_sessions:
            return False
        svc.active_sessions[session_id] = {"model": model_name}
        return True

    async def close(self) -> None:
        # No persistent resources in this shim
        return None
