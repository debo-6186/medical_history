"""Graph node functions + the prompts for the general / mixed branches.

Each node takes the AgentState and returns a partial-state dict that LangGraph
merges. Nodes never stream — they only assemble `messages`; the server streams
the final answer via core.qa.stream_answer. All retrieval reuses core.qa
verbatim so the records branch behaves exactly like today.
"""
import json
from datetime import date

import ollama

from core.agent.router import classify_intent
from core.agent.state import AgentState
from core.config import CHAT_MODEL, CHAT_NUM_CTX, MODEL_KEEP_ALIVE
from core import reminders, vector_store
from core.calendar import build_rrule
from core.qa import (
    build_messages,
    expand_query,
    format_context,
    retrieve,
    sources_from_hits,
)

# A records retrieval is capped at this many attempts (the reformulate cycle)
# to bound laptop latency. A hit count below WEAK_HIT_THRESHOLD with a non-empty
# store triggers one reformulated retry.
MAX_RETRIEVAL_ATTEMPTS = 2
WEAK_HIT_THRESHOLD = 2

GENERAL_SYSTEM_PROMPT = (
    'You are a medical information assistant. The user is asking a GENERAL '
    'medical question, not about their own records. Answer from established '
    'general medical knowledge.\n'
    '- Be accurate, clear and concise. Use Markdown: short paragraphs, bullet '
    'lists, and **bold** for key terms.\n'
    '- You do NOT have access to the user\'s personal medical records in this '
    'answer; do not invent personal values or claim to have read their '
    'reports.\n'
    '- If the question needs a clinician\'s judgement or is outside general '
    'medical knowledge, say so plainly.\n'
    '- End with a one-line disclaimer that this is educational information, not '
    'medical advice.'
)

MIXED_SYSTEM_PROMPT = (
    'You are a medical assistant. The user wants you to interpret THEIR OWN '
    'result using general medical knowledge.\n'
    'You are given record excerpts, each labelled [file: <filename> | type: '
    '<report/prescription/history>].\n'
    '- Take the personal value(s) ONLY from the provided excerpts — quote them '
    'with units and any reference range, and refer to the source by filename. '
    'If the value is not in the excerpts, say "Not found in the available '
    'records." and do not invent it.\n'
    '- You MAY use general medical knowledge to interpret what the value means '
    '(e.g. whether it is typically high or low and why), but keep the personal '
    'facts (from the records) clearly separate from the general explanation.\n'
    '- Format in Markdown: put the key number(s) in **bold**; when reporting '
    'several values use a table with the columns Date | Result | Reference '
    'range.\n'
    '- End with a one-line disclaimer that this is educational information, not '
    'medical advice, and that significant concerns should be discussed with a '
    'doctor.'
)

REFORMULATE_PROMPT = (
    'A search over a patient\'s medical records returned little or nothing for '
    'the question below. Produce a BROADER set of search terms: add synonyms, '
    'the full forms of abbreviations, related lab / test / panel names, and '
    'common spelling variants. Reply with ONLY a JSON object of the form '
    '{"terms": ["...", "..."]}. Do not answer the question or invent values.'
)

SCHEDULE_EXTRACT_PROMPT = (
    'The user wants to set a reminder. Extract its details and reply with ONLY '
    'a JSON object:\n'
    '  {"title": "<short reminder text>",\n'
    '   "datetime": "<YYYY-MM-DDTHH:MM:SS or empty>",\n'
    '   "frequency": "once|daily|twice daily|weekly|monthly|yearly",\n'
    '   "count": <integer>}\n\n'
    'count — set this ONLY when the user gives an explicit limit or duration '
    '(e.g. "for 7 days" -> 7, "3 times" -> 3). For an ongoing / open-ended / '
    '"every day" reminder with no stated end, use 0. A one-off ("once") is 0.\n'
    'Resolve relative times against the given "Today" (e.g. "tonight at 9" -> '
    'today 21:00:00; "in 6 months" -> the same day six months on at 09:00:00). '
    'If no time is given, leave "datetime" empty. Keep "title" concise and free '
    'of dosage details. Do not add anything outside the JSON object.'
)

SCHEDULE_SYSTEM_PROMPT = (
    'You are a scheduling assistant. A reminder DRAFT has just been created from '
    'the user\'s request (its details are in the user message). In 1–2 short '
    'sentences, confirm what was drafted — what, when, and how often — and tell '
    'the user to review and confirm it in their Reminders panel before it is '
    'added to their calendar. Do NOT claim it is already scheduled. End by '
    'noting they can edit the time or wording when confirming.'
)


def _assemble(system_prompt: str, question: str, history: list[dict],
              context: str | None = None) -> tuple[list[dict], str]:
    """Build (messages, user_msg) for the general / mixed branches.

    Mirrors core.qa.build_messages: today's date is injected into the system
    prompt so date arithmetic has an absolute anchor. `context` (record
    excerpts) is included only for the mixed branch.
    """
    today = date.today().isoformat()
    if context is not None:
        user_msg = f'Record excerpts:\n{context}\n\nQuestion: {question}'
    else:
        user_msg = question
    messages = [
        {'role': 'system', 'content': f'Today is {today}.\n\n{system_prompt}'},
        *(history or []),
        {'role': 'user', 'content': user_msg},
    ]
    return messages, user_msg


def _reformulate_terms(question: str, prev_terms: str) -> list[str]:
    """One json model call producing broader search terms. [] on failure."""
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': REFORMULATE_PROMPT},
                {'role': 'user',
                 'content': f'Question: {question}\nAlready tried: {prev_terms}'},
            ],
            think=False,
            format='json',
            keep_alive=MODEL_KEEP_ALIVE,
            options={'num_ctx': CHAT_NUM_CTX, 'temperature': 0.0,
                     'num_predict': 128},
        )
        data = json.loads(response['message']['content'])
        return [str(t).strip() for t in data.get('terms', []) if str(t).strip()]
    except Exception:  # noqa: BLE001 - degrade to no reformulation
        return []


def _extract_reminder(question: str) -> dict:
    """Parse a reminder request into {title, datetime, frequency, count}.

    Degrades to a title-only draft (the user fills in the time when confirming)
    on any model/parse failure.
    """
    today = date.today().isoformat()
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': SCHEDULE_EXTRACT_PROMPT},
                {'role': 'user', 'content': f'Today is {today}.\n{question}'},
            ],
            think=False,
            format='json',
            keep_alive=MODEL_KEEP_ALIVE,
            options={'num_ctx': CHAT_NUM_CTX, 'temperature': 0.0,
                     'num_predict': 128},
        )
        data = json.loads(response['message']['content'])
    except Exception:  # noqa: BLE001
        data = {}
    title = str(data.get('title') or question).strip()
    dt = str(data.get('datetime') or '').strip() or None
    frequency = str(data.get('frequency') or 'once').strip().lower()
    try:
        count = int(data.get('count') or 0)
    except (TypeError, ValueError):
        count = 0
    return {'title': title, 'datetime': dt, 'frequency': frequency,
            'count': count}


# --- Nodes -----------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    return {'intent': classify_intent(state['question'], state.get('history'))}


def records_retrieve_node(state: AgentState) -> dict:
    """Hybrid retrieval (reusing core.qa). First attempt expands the query;
    later attempts use the terms reformulate() already wrote into the state."""
    attempts = state.get('retrieval_attempts', 0)
    if attempts == 0:
        vector_text, keywords, is_value_query, date_range = expand_query(
            state['question'])
    else:
        vector_text = state['vector_text']
        keywords = state['keywords']
        is_value_query = state.get('is_value_query', False)
        date_range = state.get('date_range')
    hits = retrieve(vector_text, keywords, is_value_query, date_range)
    return {
        'vector_text': vector_text,
        'keywords': keywords,
        'is_value_query': is_value_query,
        'date_range': date_range,
        'hits': hits,
        'retrieval_attempts': attempts + 1,
    }


def reformulate_node(state: AgentState) -> dict:
    """Broaden the search terms, then loop back into records_retrieve."""
    question = state['question']
    new_terms = _reformulate_terms(question, ', '.join(state.get('keywords', [])))
    if new_terms:
        vector_text = f"{question}\n{', '.join(new_terms)}"
        keywords = [t.lower() for t in new_terms]
    else:
        vector_text = question
        keywords = state.get('keywords', [])
    return {'vector_text': vector_text, 'keywords': keywords}


def assemble_records_node(state: AgentState) -> dict:
    hits = state.get('hits', [])
    messages, user_msg = build_messages(
        state['question'], hits, state.get('history', []))
    return {'messages': messages, 'user_msg': user_msg,
            'sources': sources_from_hits(hits)}


def general_node(state: AgentState) -> dict:
    messages, user_msg = _assemble(
        GENERAL_SYSTEM_PROMPT, state['question'], state.get('history', []))
    return {'messages': messages, 'user_msg': user_msg, 'sources': [],
            'hits': []}


def mixed_node(state: AgentState) -> dict:
    """Single-hop personal retrieval + a prompt that permits general
    interpretation, so personal value and general meaning are synthesised in
    one answer."""
    vector_text, keywords, is_value_query, date_range = expand_query(
        state['question'])
    hits = retrieve(vector_text, keywords, is_value_query, date_range)
    context = format_context(hits) if hits else '(no records ingested yet)'
    messages, user_msg = _assemble(
        MIXED_SYSTEM_PROMPT, state['question'], state.get('history', []),
        context=context)
    return {'hits': hits, 'messages': messages, 'user_msg': user_msg,
            'sources': sources_from_hits(hits)}


def scheduling_node(state: AgentState) -> dict:
    """Draft a reminder from the user's request and persist it as *pending*.

    Nothing is sent to Google here — the draft is stored locally and surfaced as
    a `proposal` for the user to review/edit/confirm. The model then writes a
    short confirmation. The stored user_msg is the original question so the
    conversation history stays coherent.
    """
    question = state['question']
    parsed = _extract_reminder(question)
    rrule = build_rrule(parsed['frequency'], parsed['count'])
    reminder = reminders.add(
        kind='manual',
        title=parsed['title'],
        proposed_text=parsed['title'],
        start=parsed['datetime'],
        rrule=rrule,
    )

    today = date.today().isoformat()
    draft = (
        f"Draft reminder — text: {parsed['title']}; "
        f"when: {parsed['datetime'] or 'not specified'}; "
        f"repeat: {parsed['frequency']}."
    )
    messages = [
        {'role': 'system',
         'content': f'Today is {today}.\n\n{SCHEDULE_SYSTEM_PROMPT}'},
        *state.get('history', []),
        {'role': 'user', 'content': f'{draft}\n\nUser request: {question}'},
    ]
    return {'messages': messages, 'user_msg': question, 'sources': [],
            'proposal': reminder}


# --- Conditional edge functions --------------------------------------------

def route_by_intent(state: AgentState) -> str:
    """Map the router's intent to a branch. clarify falls through to records."""
    intent = state.get('intent', 'records')
    if intent == 'general':
        return 'general'
    if intent == 'mixed':
        return 'mixed'
    if intent == 'schedule':
        return 'schedule'
    return 'records'


def grade_hits(state: AgentState) -> str:
    """Weak retrieval (and budget left) -> reformulate; else assemble."""
    hits = state.get('hits', [])
    attempts = state.get('retrieval_attempts', 0)
    try:
        store_has_data = vector_store.count() > 0
    except Exception:  # noqa: BLE001
        store_has_data = False
    weak = store_has_data and len(hits) < WEAK_HIT_THRESHOLD
    if weak and attempts < MAX_RETRIEVAL_ATTEMPTS:
        return 'reformulate'
    return 'assemble'
