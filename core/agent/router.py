"""Intent router: a single Gemma call that classifies a turn's scope.

This is the "router" of the agent graph — NOT an external model-routing service.
It runs entirely on the local Ollama/Gemma model, like the rest of the app, and
only decides which specialist branch the graph should take next. Routing itself
is then plain graph edges (see core/agent/graph.py); we never rely on Gemma's
weak native tool-calling.

The classification axis here is *scope* (whose knowledge answers this — the
user's own records, general medical knowledge, both, or a scheduling request),
which is orthogonal to expand_query's existing values|general axis in core/qa.py.
On any failure it degrades to 'records', preserving today's behaviour.
"""
import json

import ollama

from core.config import CHAT_MODEL, CHAT_NUM_CTX, MODEL_KEEP_ALIVE

ROUTER_PROMPT = (
    'You are the router for a personal medical assistant. Classify the user\'s '
    'message into exactly one intent and reply with ONLY a JSON object of the '
    'form {"intent": "<value>"}.\n\n'
    'Intents:\n'
    '- "records": the user asks about THEIR OWN medical data — past lab '
    'results, test values, reports, prescriptions, diagnoses or history they '
    'have uploaded. E.g. "what was my CRP in May", "show my latest lipid '
    'panel", "which tablets did the doctor prescribe".\n'
    '- "general": a general medical-knowledge question NOT about the user\'s '
    'own data. E.g. "what does a high CRP mean", "what is hypertension", '
    '"what are the side effects of metformin".\n'
    '- "mixed": needs BOTH the user\'s own value AND general medical knowledge '
    'to interpret it. E.g. "is my CRP of 12 high", "are my latest results '
    'normal", "should I be worried about my LDL".\n'
    '- "schedule": the user wants to set a reminder or schedule something. '
    'E.g. "remind me to take my medicine", "set a reminder for my next scan".\n'
    '- "clarify": the message is too vague or unrelated to act on.\n\n'
    'When unsure between records and mixed: choose "records" if they only want '
    'the value, "mixed" if they ask whether it is normal / high / concerning.\n'
    'Reply with ONLY the JSON object.'
)

_VALID_INTENTS = {'records', 'general', 'mixed', 'schedule', 'clarify'}


def _recent_context(history: list[dict] | None) -> str:
    """Compact tail of the conversation so follow-ups ('what about June?')
    route the same way as the turn they refer to."""
    if not history:
        return ''
    tail = history[-2:]
    lines = [f"{m.get('role', '?')}: {m.get('content', '')}" for m in tail]
    return 'Recent conversation:\n' + '\n'.join(lines) + '\n\n'


def classify_intent(question: str, history: list[dict] | None = None) -> str:
    """Return one of _VALID_INTENTS for the question.

    MUST be called with MODEL_LOCK held — it invokes the chat model. Falls back
    to 'records' on any parse/model failure so chat never breaks.
    """
    user = f'{_recent_context(history)}Message: {question}'
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': ROUTER_PROMPT},
                {'role': 'user', 'content': user},
            ],
            think=False,        # routing never needs chain-of-thought
            format='json',      # structured output instead of tool-calling
            keep_alive=MODEL_KEEP_ALIVE,
            options={'num_ctx': CHAT_NUM_CTX, 'temperature': 0.0,
                     'num_predict': 64},
        )
        data = json.loads(response['message']['content'])
        intent = str(data.get('intent', '')).strip().lower()
        if intent in _VALID_INTENTS:
            return intent
    except Exception:  # noqa: BLE001 - never let routing break chat
        pass
    return 'records'
