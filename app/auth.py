"""Shared-PIN authentication: login, in-memory sessions, throttling.

Sessions live in process memory only — a backend restart simply requires the
user to re-enter the PIN. This is a second layer behind the Tailscale network
boundary, so that is acceptable.
"""
import hmac
import secrets
import time

from fastapi import Header, HTTPException

from core.config import APP_PIN, SESSION_TTL_HOURS

_SESSIONS: dict[str, float] = {}   # token -> expiry epoch seconds
_FAILED: list[float] = []          # timestamps of recent failed logins

_LOCKOUT_THRESHOLD = 5
_LOCKOUT_WINDOW = 60.0
_FAIL_DELAY = 0.5


def _purge_failed(now: float) -> None:
    cutoff = now - _LOCKOUT_WINDOW
    while _FAILED and _FAILED[0] < cutoff:
        _FAILED.pop(0)


def _purge_sessions(now: float) -> None:
    for token in [t for t, exp in _SESSIONS.items() if exp < now]:
        _SESSIONS.pop(token, None)


def login(pin: str) -> tuple[str, int]:
    """Validate the PIN and return (session_token, ttl_seconds)."""
    now = time.time()
    _purge_failed(now)
    if len(_FAILED) >= _LOCKOUT_THRESHOLD:
        raise HTTPException(429, 'Too many attempts, wait 60s')

    if not APP_PIN or not hmac.compare_digest(pin, APP_PIN):
        _FAILED.append(now)
        time.sleep(_FAIL_DELAY)
        raise HTTPException(401, 'Invalid PIN')

    _FAILED.clear()
    token = secrets.token_urlsafe(32)
    ttl = SESSION_TTL_HOURS * 3600
    _SESSIONS[token] = now + ttl
    return token, ttl


def require_auth(authorization: str = Header(default='')) -> str:
    """FastAPI dependency: validate the bearer token, return it, or raise 401."""
    now = time.time()
    _purge_sessions(now)
    if not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Missing or invalid token')
    token = authorization[len('Bearer '):]
    expiry = _SESSIONS.get(token)
    if expiry is None or expiry < now:
        raise HTTPException(401, 'Missing or invalid token')
    return token
