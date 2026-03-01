"""Groq LLM: system message = structured report + instructions; messages = history + new user message."""

import json
import logging
import os
from typing import Any

from groq import Groq

logger = logging.getLogger(__name__)


def _format_structured_report(structured_report: list[dict[str, Any]]) -> str:
    """Format structured report (with status, fact) as readable text for the system prompt."""
    lines = []
    for r in structured_report:
        name = r.get("test_name", "")
        value = r.get("value", "")
        unit = r.get("unit", "") or ""
        ref_low = r.get("ref_low")
        ref_high = r.get("ref_high")
        status = r.get("status", "")
        fact = r.get("fact", "")
        ref_str = ""
        if ref_low is not None or ref_high is not None:
            ref_str = f" (Reference: {ref_low or '?'} - {ref_high or '?'} {unit})".strip()
        lines.append(
            f"- {name}: {value} {unit}{ref_str} | Status: {status} | {fact}"
        )
    return "\n".join(lines) if lines else "No test results available."


def _format_context_inputs(context_inputs: dict[str, Any] | None) -> str:
    if not context_inputs:
        return "No additional context provided."
    lines = []
    for k, v in context_inputs.items():
        if v is None or v == "" or v == []:
            continue
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else "No additional context provided."


def _build_messages(system_content: str, chat_history: list[dict[str, str]], user_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for m in chat_history:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    return t


def _extract_first_json_object(text: str) -> str | None:
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _parse_diet_json(text: str) -> dict[str, Any] | None:
    """Parse LLM JSON output for diet guidance (allow extra keys)."""
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    data: Any = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(cleaned)
        if extracted:
            try:
                data = json.loads(extracted)
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(data, dict):
        return None
    required = ("notes", "emphasize", "limit", "sample_day")
    for key in required:
        if key not in data or not isinstance(data[key], list):
            return None
    return {k: data[k] for k in required}


def get_chat_response(
    structured_report: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    user_message: str,
    context_inputs: dict[str, Any] | None = None,
) -> str:
    """
    Call LLM with:
    - System message = full structured report + instructions
    - Messages = conversation history (prev Q&A) + new user message

    chat_history: list of { "role": "user" | "assistant", "content": "..." }
    Returns assistant reply string.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not set")
        return "Error: GROQ_API_KEY is not set. Add it to your .env file."

    report_text = _format_structured_report(structured_report)
    num_tests = len(structured_report)
    history_len = len(chat_history)
    logger.info("Calling Groq: report_tests=%s, chat_history_turns=%s, user_message_len=%s", num_tests, history_len, len(user_message))

    context_text = _format_context_inputs(context_inputs)
    system_content = (
        "You are a helpful assistant that explains lab report results in simple language. "
        "Use ONLY the structured lab data below. Do not invent or assume any values. "
        "When results are abnormal, recommend consulting a doctor. "
        "If results are abnormal, include a short 'Possible causes' section with common possibilities, "
        "clearly labeled as non-diagnostic and dependent on clinical context. "
        "If the user asks about diet/food, include: Foods to emphasize, Foods to limit, and a 1-day sample plan. "
        "Provide a structured response with headings: Summary, Key Abnormalities, Possible causes, Suggested follow-up. "
        "When abnormal results exist, include 6-10 bullet points in Key Abnormalities for richer detail. "
        "Always add a short disclaimer that this is for educational use and not a substitute for medical advice.\n\n"
        "Structured lab data:\n"
        f"{report_text}\n\n"
        "Additional context (may be empty):\n"
        f"{context_text}"
    )

    messages = _build_messages(system_content, chat_history, user_message)

    try:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        logger.debug("Invoking Groq (%s), messages=%s", model, len(messages))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=3072,
        )
        content = response.choices[0].message.content if response.choices else ""
        logger.info("Groq response received: reply_len=%s", len(content))
        return content
    except Exception as e:
        logger.exception("Groq call failed: %s", e)
        return f"Sorry, an error occurred while calling the assistant: {e!s}"


def get_type_likelihood_note(
    structured_report: list[dict[str, Any]],
    context_inputs: dict[str, Any] | None,
) -> str:
    """Generate a short, non-diagnostic Type 1 vs Type 2 likelihood note."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not set")
        return "Error: GROQ_API_KEY is not set. Add it to your .env file."

    report_text = _format_structured_report(structured_report)
    context_text = _format_context_inputs(context_inputs)
    system_content = (
        "You are a cautious medical assistant. Provide a short, non-diagnostic note about Type 1 vs Type 2 likelihood. "
        "Use ONLY the structured lab data and the context inputs below. Do not invent or assume values. "
        "If the provided data is insufficient to suggest anything, say so clearly. "
        "Keep it to 4-6 sentences and end with a brief disclaimer that this is educational only and not medical advice.\n\n"
        "Structured lab data:\n"
        f"{report_text}\n\n"
        "Context inputs:\n"
        f"{context_text}"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Write the short Type 1 vs Type 2 likelihood note now."},
    ]
    try:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        logger.debug("Invoking Groq for type likelihood note (%s)", model)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )
        content = response.choices[0].message.content if response.choices else ""
        logger.info("Type likelihood note received: reply_len=%s", len(content))
        return content
    except Exception as e:
        logger.exception("Groq call failed: %s", e)
        return f"Sorry, an error occurred while calling the assistant: {e!s}"


def get_diet_guidance(
    structured_report: list[dict[str, Any]],
    context_inputs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Generate diet guidance using Groq. Returns dict or None on failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not set")
        return None

    report_text = _format_structured_report(structured_report)
    context_text = _format_context_inputs(context_inputs)
    system_content = (
        "You are a clinical nutrition assistant. "
        "Use ONLY the structured lab data and context inputs below. "
        "Treat insulin_resistance_signs and BMI/body_habitus as PRIMARY drivers of guidance when present. "
        "Use age_at_onset and speed_of_onset as secondary modifiers when provided. "
        "When any of these context signals are present, ensure they clearly shape notes and at least one "
        "item in emphasize and/or limit (avoid generic-only advice). "
        "Make guidance practical with portion-sized suggestions (e.g., '1 cup', 'palm-sized'). "
        "Return STRICT JSON with keys: notes, emphasize, limit, sample_day. "
        "Each value must be a list of short strings. "
        "If data is insufficient, keep lists empty and add a note that guidance is general. "
        "Do not include any additional keys or prose outside JSON.\n\n"
        "Structured lab data:\n"
        f"{report_text}\n\n"
        "Additional context:\n"
        f"{context_text}"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Generate the JSON diet guidance now."},
    ]
    try:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1400,
        )
        content = response.choices[0].message.content if response.choices else ""
        parsed = _parse_diet_json(content)
        if parsed is None:
            logger.warning("Diet guidance JSON parse failed")
        return parsed
    except Exception as e:
        logger.exception("Groq diet guidance call failed: %s", e)
        return None
