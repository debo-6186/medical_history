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
