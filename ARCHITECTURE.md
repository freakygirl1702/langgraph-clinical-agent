# Medical Lab Report Chatbot – Architecture

## 1. High-Level Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PRESENTATION (Streamlit)                                                     │
│  app.py – Upload UI, chat UI, session state, disclaimers                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION                                                               │
│  • On upload: run process-report graph (LangGraph 1)                         │
│  • On message: run chat graph (LangGraph 2)                                  │
│  • Session state only (no DB, no file storage)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                     ▼
┌─────────────────────────────┐                 ┌─────────────────────────────┐
│  GRAPH 1: Process report   │                 │  GRAPH 2: Chat               │
│  (reasoning/graph.py)       │                 │  (reasoning/graph.py)        │
│  extract_node → rules_node │                 │  qa_node                      │
└─────────────────────────────┘                 └─────────────────────────────┘
              │                                                     │
              ▼                                                     ▼
┌─────────────────────────────┐                 ┌─────────────────────────────┐
│  DOCUMENT PROCESSING        │                 │  REASONING + LLM            │
│  document_processing/      │                 │  reasoning/rules.py         │
│  extractor.py, parser.py   │                 │  reasoning/llm.py (Groq 70B)   │
└─────────────────────────────┘                 └─────────────────────────────┘
```

---

## 2. Project Structure

| Path | Role |
|------|------|
| **`app.py`** | Streamlit entry: file upload, chat UI, session init, invokes both LangGraphs. |
| **`config_logging.py`** | Central logging config (format, level, stderr). |
| **`document_processing/`** | PDF/image → raw text → structured lab rows. |
| **`document_processing/extractor.py`** | PDF (PyMuPDF) and image (Tesseract OCR) → raw text; logs OCR-extracted text. |
| **`document_processing/parser.py`** | Raw text → list of `{ test_name, value, unit, ref_low, ref_high }`. |
| **`reasoning/`** | Rules engine, LLM (Groq), and LangGraph definitions. |
| **`reasoning/rules.py`** | Compare value to ref range → `status`, `fact`, `critical`; logs each generated rule. |
| **`reasoning/llm.py`** | Groq (70B): system = report + instructions, messages = history + user; returns reply. |
| **`reasoning/graph.py`** | LangGraph: Graph 1 (extract → rules), Graph 2 (qa only). |
| **`.env`** | `GROQ_API_KEY` (not committed). |
| **`requirements.txt`** | Dependencies (Streamlit, PyMuPDF, Tesseract deps, LangChain, LangGraph, groq). |

---

## 3. Data Flow

### A. Report Upload

1. User uploads a file in Streamlit.
2. App reads bytes and invokes **Graph 1** with `{ file_bytes, file_name }`.
3. **extract_node:** Calls `extract_text_from_file` (PyMuPDF or Tesseract) → `raw_text`; then `parse_lab_report` → `parsed_rows`. State is updated with `raw_text`, `parsed_rows`.
4. **rules_node:** Calls `apply_rules(parsed_rows)` → `structured_report` (each row gains `status`, `fact`, `critical`). State is updated with `structured_report`.
5. App reads `structured_report` from the graph result and stores it in **`st.session_state.structured_report`**, clears `messages`, and shows “Report processed.”

### B. Chat (Each User Message)

1. User types a message; app appends it to `st.session_state.messages`.
2. App invokes **Graph 2** with `{ structured_report, messages (history), user_message }`.
3. **qa_node:** Builds system message (full structured report + instructions) and message list (history + new user message); calls **Groq** via `get_chat_response`; appends user + assistant to `messages`; returns `{ messages, assistant_reply }`.
4. App displays the reply and sets `st.session_state.messages = result["messages"]`.

All state lives in **session only**; no database or file persistence for reports or chat.

---

## 4. LangGraph Design (Option A)

- **Graph 1 – Process report**  
  `START → extract_node → rules_node → END`  
  Input: `file_bytes`, `file_name`.  
  Output (in state): `raw_text`, `parsed_rows`, `structured_report`.

- **Graph 2 – Chat**  
  `START → qa_node → END`  
  Input: `structured_report`, `messages`, `user_message`.  
  Output: `messages` (updated), `assistant_reply`.

Shared “state” is plain dicts passed in and out of `invoke`; the app holds the canonical state in Streamlit session.

---

## 5. Main Technologies

| Concern | Technology |
|--------|------------|
| UI | Streamlit |
| PDF / images | PyMuPDF, pdf2image, Tesseract (pytesseract, Pillow) |
| Pipeline / workflow | LangGraph (two compiled graphs) |
| LLM | Groq via `groq` client (Llama 70B) |
| Env | python-dotenv, `GROQ_API_KEY` |
| Logging | `logging` + `config_logging` |

---

## 6. Summary Diagram

```
User → [Streamlit app]
         │
         ├─ Upload → Graph1(extract → rules) → structured_report → session
         │
         └─ Message → Graph2(qa_node) → Groq → assistant_reply → session
                         │
                         └─ qa_node uses: structured_report (system) + messages (history) + user_message
```
