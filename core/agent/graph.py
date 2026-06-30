"""The LangGraph chat graph: router -> records (with retry) / general / mixed.

Compiled once at import. `run_turn` is a synchronous generator that yields
progress events as each node runs and finally the assembled state; the server
streams the answer from `state['messages']`. There is no checkpointer — the
graph never pauses (no human-in-the-loop here), and the chat route holds
MODEL_LOCK across the whole turn anyway.
"""
from collections.abc import Iterator

from langgraph.graph import END, START, StateGraph

from core.agent import nodes
from core.agent.state import AgentState

# Node name -> the `status` step label surfaced to the UI. records_retrieve maps
# to 'searching'; the reformulate node is the natural 'retrying' signal.
_STEP_LABELS = {
    'router': 'routing',
    'records_retrieve': 'searching',
    'reformulate': 'retrying',
    'assemble_records': 'answering',
    'general': 'answering',
    'mixed': 'answering',
    'scheduling': 'scheduling',
}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node('router', nodes.router_node)
    g.add_node('records_retrieve', nodes.records_retrieve_node)
    g.add_node('reformulate', nodes.reformulate_node)
    g.add_node('assemble_records', nodes.assemble_records_node)
    g.add_node('general', nodes.general_node)
    g.add_node('mixed', nodes.mixed_node)
    g.add_node('scheduling', nodes.scheduling_node)

    g.add_edge(START, 'router')
    g.add_conditional_edges('router', nodes.route_by_intent, {
        'records': 'records_retrieve',
        'general': 'general',
        'mixed': 'mixed',
        'schedule': 'scheduling',
    })
    g.add_conditional_edges('records_retrieve', nodes.grade_hits, {
        'reformulate': 'reformulate',
        'assemble': 'assemble_records',
    })
    g.add_edge('reformulate', 'records_retrieve')
    g.add_edge('assemble_records', END)
    g.add_edge('general', END)
    g.add_edge('mixed', END)
    g.add_edge('scheduling', END)
    return g.compile()


_GRAPH = build_graph()


def run_turn(question: str, history: list[dict] | None = None) -> Iterator[tuple]:
    """Run one agent turn.

    Yields ('status', {'step': ..., 'intent': ...}) for each node as it runs,
    then exactly one ('result', state) with the final state — `messages`,
    `sources`, `user_msg`, `intent`, `retrieval_attempts`.

    Synchronous and blocking (all local model calls); invoke from a worker
    thread with MODEL_LOCK held. State has no custom reducers, so plain dict
    updates reconstruct the final state.
    """
    initial: AgentState = {
        'question': question,
        'history': list(history or []),
        'retrieval_attempts': 0,
    }
    final: dict = dict(initial)
    for update in _GRAPH.stream(initial, stream_mode='updates'):
        for node, partial in update.items():
            if partial:
                final.update(partial)
            yield ('status', {'step': _STEP_LABELS.get(node, node),
                              'intent': final.get('intent')})
    yield ('result', final)
