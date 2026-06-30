"""Pydantic request/response models for the API."""
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    pin: str


class LoginResponse(BaseModel):
    token: str
    expires_in: int


class HistoryRequest(BaseModel):
    text: str = Field(min_length=1)


class HistoryResponse(BaseModel):
    doc_id: str
    chunks: int


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


class ReminderConfirmRequest(BaseModel):
    """Edits the user may apply before a pending reminder is pushed to Google.

    `start` is an ISO-8601 datetime; all fields are optional — omitted ones keep
    the drafted value. `timezone` is an IANA name (e.g. 'Asia/Kolkata').
    """
    title: str | None = None
    proposed_text: str | None = None
    start: str | None = None
    rrule: str | None = None
    timezone: str = 'UTC'
