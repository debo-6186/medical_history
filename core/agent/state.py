"""Shared state for the LangGraph agent turn.

A single chat turn flows through the graph as this dict. Nodes return partial
updates that LangGraph merges in (plain overwrite — no custom reducers). The
graph's job is to populate `messages` + `sources` + `user_msg`; the server
streams the answer from `messages` and stores `user_msg` in the conversation
history.
"""
from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Inputs
    question: str
    history: list[dict]

    # Router output
    intent: str  # records | general | mixed | schedule | clarify

    # Retrieval working set (records / mixed branches)
    vector_text: str
    keywords: list[str]
    is_value_query: bool
    date_range: Optional[tuple[int, int]]
    hits: list[tuple]
    retrieval_attempts: int

    # Outputs the server consumes
    messages: list[dict]
    sources: list[dict]
    user_msg: str

    # Scheduling branch: the drafted pending reminder, surfaced to the client as
    # a `proposal` SSE event for the user to review and confirm.
    proposal: dict
