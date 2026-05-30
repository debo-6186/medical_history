# medical_history

## Running the server

From the project root:

```
uv run uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Binds to `127.0.0.1` only — remote access goes through `tailscale serve` (see `TECH_SPEC.md`). Requires a local Ollama instance for chat and the `/api/health` check.
