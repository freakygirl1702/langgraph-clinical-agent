"""
LangGraph Option A: two graphs.
Graph 1 (process report): extract_node -> rules_node.
Graph 2 (chat): qa_node only.
"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from document_processing import extract_text_from_file, parse_lab_report
from reasoning.llm import get_chat_response
from reasoning.rules import apply_rules

logger = logging.getLogger(__name__)


# Shared state: process report uses file_bytes, file_name -> raw_text, parsed_rows -> structured_report.
# Chat uses structured_report, messages, user_message -> assistant_reply (and updated messages).
def _file_like(data: bytes, name: str):
    class FileLike:
        def __init__(self, d: bytes, n: str):
            self._data = d
            self.name = n

        def read(self) -> bytes:
            return self._data

    return FileLike(data, name)


def extract_node(state: dict[str, Any]) -> dict[str, Any]:
    """Node 1: OCR/extraction + parse. Reads file_bytes, file_name; writes raw_text, parsed_rows."""
    file_bytes = state["file_bytes"]
    file_name = state.get("file_name") or ""
    logger.info("LangGraph extract_node: file_name=%s, size=%s", file_name, len(file_bytes))
    raw_text = extract_text_from_file(_file_like(file_bytes, file_name))
    if not raw_text or len(raw_text.strip()) < 20:
        logger.warning("Insufficient text from extract_node: %s chars", len(raw_text or ""))
        raise ValueError("Could not extract enough text from the report. Try a clearer PDF or image.")
    parsed_rows = parse_lab_report(raw_text)
    if not parsed_rows:
        logger.warning("No lab rows parsed; rules_node will produce empty structured_report")
    logger.info("LangGraph extract_node done: raw_text=%s chars, parsed_rows=%s", len(raw_text), len(parsed_rows))
    return {"raw_text": raw_text, "parsed_rows": parsed_rows}


def rules_node(state: dict[str, Any]) -> dict[str, Any]:
    """Node 2: Rule checking. Reads parsed_rows; writes structured_report."""
    parsed_rows = state.get("parsed_rows") or []
    logger.info("LangGraph rules_node: %s row(s)", len(parsed_rows))
    structured_report = apply_rules(parsed_rows)
    logger.info("LangGraph rules_node done: structured_report=%s row(s)", len(structured_report))
    return {"structured_report": structured_report}


def qa_node(state: dict[str, Any]) -> dict[str, Any]:
    """Node 3: Q&A with LLM. Reads structured_report, messages, user_message; writes assistant_reply, updates messages."""
    structured_report = state.get("structured_report") or []
    messages = list(state.get("messages") or [])
    user_message = state.get("user_message") or ""
    context_inputs = state.get("context_inputs") or {}
    logger.info("LangGraph qa_node: report_tests=%s, history_turns=%s", len(structured_report), len(messages))
    assistant_reply = get_chat_response(
        structured_report=structured_report,
        chat_history=messages,
        user_message=user_message,
        context_inputs=context_inputs,
    )
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_reply})
    logger.info("LangGraph qa_node done: reply_len=%s", len(assistant_reply))
    return {"messages": messages, "assistant_reply": assistant_reply}


def build_process_report_graph():
    """Graph 1: START -> extract -> rules -> END."""
    graph = StateGraph(dict)
    graph.add_node("extract", extract_node)
    graph.add_node("rules", rules_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "rules")
    graph.add_edge("rules", END)
    return graph.compile()


def build_chat_graph():
    """Graph 2: START -> qa -> END."""
    graph = StateGraph(dict)
    graph.add_node("qa", qa_node)
    graph.add_edge(START, "qa")
    graph.add_edge("qa", END)
    return graph.compile()
