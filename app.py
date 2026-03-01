"""
Medical Lab Report Chatbot - Streamlit app.
Upload a lab report (PDF/image), get a summary, and ask questions in the same session.
No data is stored; session state only.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from config_logging import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st

from document_processing import extract_text_from_file, parse_lab_report
from reasoning import build_chat_graph, build_process_report_graph
from reasoning.llm import get_type_likelihood_note, get_diet_guidance


def init_session_state():
    if "structured_report" not in st.session_state:
        st.session_state.structured_report = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "context_inputs" not in st.session_state:
        st.session_state.context_inputs = {}
    if "type_likelihood_note" not in st.session_state:
        st.session_state.type_likelihood_note = ""
    if "context_dirty" not in st.session_state:
        st.session_state.context_dirty = False
    if "diet_guidance" not in st.session_state:
        st.session_state.diet_guidance = None
    if "diet_dirty" not in st.session_state:
        st.session_state.diet_dirty = True


def _build_diet_guidance_fallback(report: list[dict], context_inputs: dict | None = None) -> dict:
    emphasize: set[str] = set()
    limit: set[str] = set()
    notes: list[str] = []
    renal_flag = False
    context_inputs = context_inputs or {}
    bmi = context_inputs.get("bmi")
    body_habitus = context_inputs.get("body_habitus")
    insulin_flags = context_inputs.get("insulin_resistance_signs") or []
    age_at_onset = context_inputs.get("age_at_onset")
    speed_of_onset = context_inputs.get("speed_of_onset")

    for r in report or []:
        name = str(r.get("test_name", "")).lower()
        status = r.get("status", "")
        if status not in ("high", "low"):
            continue

        if any(k in name for k in ("hba1c", "glucose")):
            emphasize.update({
                "Non-starchy vegetables (leafy greens, broccoli, peppers)",
                "Lean proteins (fish, chicken, tofu, eggs)",
                "High-fiber carbs in small portions (oats, beans, lentils)",
                "Healthy fats (olive oil, nuts, seeds)"
            })
            limit.update({
                "Sugary drinks and sweets",
                "Refined carbs (white bread, pastries, white rice)",
                "Large late-night carb-heavy meals"
            })

        if "triglycerides" in name:
            emphasize.update({
                "Omega-3 sources (salmon, sardines, flax, chia)",
                "Whole grains and high-fiber foods"
            })
            limit.update({
                "Alcohol",
                "Added sugars and sweetened beverages"
            })

        if "ldl" in name or "cholesterol" in name:
            emphasize.update({
                "Soluble fiber (oats, barley, beans)",
                "Nuts and seeds",
                "Olive oil and avocado"
            })
            limit.update({
                "Fried/processed meats",
                "Trans fats and high saturated fat foods"
            })

        if "crp" in name:
            emphasize.update({
                "Colorful fruits and vegetables",
                "Fatty fish or plant omega-3s"
            })
            limit.update({
                "Ultra-processed foods"
            })

        if "hemoglobin" in name and status == "low":
            emphasize.update({
                "Iron-rich foods (lean meats, beans, spinach)",
                "Vitamin C foods with iron (citrus, tomatoes, peppers)"
            })

        if any(k in name for k in ("creatinine", "egfr", "potassium", "sodium")):
            renal_flag = True

        if "troponin" in name:
            notes.append("If you have chest pain or shortness of breath, seek urgent medical care.")

    if renal_flag:
        notes.append(
            "Kidney/electrolyte-related results need personalized nutrition guidance. "
            "Avoid major diet changes without clinician approval."
        )

    if bmi is not None:
        if bmi >= 30:
            emphasize.update({
                "Non-starchy vegetables for volume (at least half the plate)",
                "Lean protein each meal (palm-sized portion)",
                "High-fiber carbs in small portions (beans, oats, lentils)"
            })
            limit.update({
                "Sugary drinks and sweets",
                "Refined carbs (white bread, pastries, white rice)",
                "Fried or ultra-processed foods"
            })
            notes.append("BMI suggests obesity: prioritize calorie control, higher protein, and fiber.")
        elif bmi >= 25:
            emphasize.update({
                "Vegetable-forward meals and salads",
                "Lean protein with each meal",
                "Portion-controlled whole grains"
            })
            limit.update({
                "Refined carbs and sweets",
                "Sugar-sweetened beverages",
                "Large late-night meals"
            })
            notes.append("BMI suggests overweight: focus on portion control and lower refined carbs.")
        elif bmi < 18.5:
            emphasize.update({
                "Calorie-dense, nutrient-rich foods (olive oil, avocado, nuts)",
                "Protein + carb at each meal",
                "Small, frequent meals or snacks"
            })
            notes.append("BMI suggests underweight: include calorie-dense, nutrient-rich foods.")

    if body_habitus and body_habitus != "unknown":
        if bmi is None:
            if body_habitus == "obese":
                emphasize.update({
                    "Non-starchy vegetables for volume",
                    "Lean protein each meal",
                    "High-fiber carbs in small portions"
                })
                limit.update({
                    "Sugary drinks and sweets",
                    "Refined carbs",
                    "Fried or ultra-processed foods"
                })
            elif body_habitus == "overweight":
                emphasize.update({
                    "Vegetable-forward meals",
                    "Lean protein",
                    "Portion-controlled whole grains"
                })
                limit.update({
                    "Refined carbs and sweets",
                    "Sugar-sweetened beverages",
                    "Large late-night meals"
                })
            elif body_habitus == "underweight":
                emphasize.update({
                    "Calorie-dense, nutrient-rich foods",
                    "Protein + carb at each meal",
                    "Small, frequent meals or snacks"
                })
        notes.append(f"Body habitus noted: {body_habitus}. Adjust portions accordingly.")

    if insulin_flags and "unknown" not in insulin_flags:
        emphasize.update({
            "Low-glycemic carbs (beans, lentils, oats) in measured portions",
            "Protein + fiber at each meal to blunt glucose spikes",
            "Consistent meal timing"
        })
        limit.update({
            "Sugary drinks and sweets",
            "Refined grains and snacks",
            "Large carb-only meals"
        })
        flags_text = ", ".join(insulin_flags)
        notes.append(
            f"Insulin-resistance signs present ({flags_text}): emphasize low-glycemic foods and steady meal timing."
        )
    if age_at_onset:
        notes.append("Age at onset may influence nutrition priorities; confirm with a clinician.")
    if speed_of_onset and speed_of_onset != "unknown":
        notes.append(f"Speed of onset reported as {speed_of_onset}; tailor follow-up nutrition guidance.")

    if not notes:
        notes.append("Diet guidance is general and should be tailored by a clinician.")

    sample_day = [
        "Breakfast: Greek yogurt or tofu + berries + a small handful of nuts",
        "Lunch: Grilled fish/chicken + large salad + olive oil dressing",
        "Snack: Apple or cucumber with hummus",
        "Dinner: Lentil/bean bowl with vegetables and a small portion of whole grains",
        "Hydration: Water; avoid sugary drinks"
    ]

    return {
        "emphasize": sorted(emphasize),
        "limit": sorted(limit),
        "notes": notes,
        "sample_day": sample_day,
    }


def _run_process_report_graph(file_bytes: bytes, file_name: str) -> list | None:
    """Run LangGraph: extract_node -> rules_node. Returns structured_report or None on error."""
    graph = build_process_report_graph()
    try:
        result = graph.invoke({"file_bytes": file_bytes, "file_name": file_name})
        report = result.get("structured_report") or []
        logger.info("Process report graph done: %s result(s)", len(report))
        return report
    except ValueError as e:
        logger.warning("Process report graph validation: %s", e)
        st.error(str(e))
        return None
    except Exception as e:
        logger.exception("Process report graph failed: %s", e)
        st.error(f"Could not read the file: {e}")
        return None


def main():
    st.set_page_config(page_title="Medical Lab Report Chatbot", page_icon="🩺", layout="centered")
    st.markdown("""
        <style>
        :root {
            --ink: #101418;
            --muted: #4b5563;
            --brand: #0e3a5d;
            --accent: #f59e0b;
            --bg: #f8fafc;
            --card: #ffffff;
            --danger: #b91c1c;
            --warning: #c2410c;
            --ok: #0f766e;
            --border: #e5e7eb;
        }
        .app-shell {
            background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 50%, #ffffff 100%);
            padding: 18px 22px;
            border-radius: 14px;
            border: 1px solid var(--border);
        }
        .hero-title {
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 30px;
            color: var(--brand);
            margin: 0 0 6px 0;
        }
        .hero-sub {
            color: var(--muted);
            font-size: 14px;
            margin: 0 0 10px 0;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 10px;
        }
        .kpi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }
        .kpi-title {
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .kpi-value {
            font-size: 22px;
            font-weight: 700;
            color: var(--ink);
        }
        .kpi-danger { color: var(--danger); }
        .kpi-warning { color: var(--warning); }
        .kpi-ok { color: var(--ok); }
        .section-title {
            font-size: 18px;
            margin: 18px 0 8px 0;
            color: var(--ink);
            font-weight: 700;
        }
        .highlight {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 12px;
            padding: 10px 12px;
            margin: 8px 0;
        }
        .muted { color: var(--muted); }
        @media (max-width: 900px) {
            .kpi-grid {
                grid-template-columns: 1fr;
            }
            .app-shell {
                padding: 14px 16px;
            }
            .hero-title {
                font-size: 24px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="app-shell">
          <div class="hero-title">Medical Lab Report Chatbot</div>
          <div class="hero-sub">Upload a lab report (PDF or image) to get a clear explanation and ask questions. No data is stored.</div>
        </div>
        """, unsafe_allow_html=True)

    init_session_state()

    # Sidebar: optional context inputs for Type 1 vs Type 2 likelihood
    with st.sidebar:
        st.subheader("Optional Context (Non-diagnostic)")
        age_at_onset = st.number_input("Age at onset", min_value=0, max_value=120, value=0, help="Set to 0 if unknown.")
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=0.0, step=0.1, help="Set to 0 if unknown.")
        body_habitus = st.selectbox(
            "Body habitus",
            options=["unknown", "underweight", "normal", "overweight", "obese"],
            index=0,
        )
        speed_of_onset = st.selectbox(
            "Speed of onset",
            options=["unknown", "sudden", "gradual"],
            index=0,
        )
        ketosis_dka_history = st.selectbox(
            "Ketosis or DKA history",
            options=["unknown", "yes", "no"],
            index=0,
        )
        insulin_required_immediately = st.selectbox(
            "Insulin required right after diagnosis",
            options=["unknown", "yes", "no"],
            index=0,
        )
        family_history = st.multiselect(
            "Family history",
            options=["type_1_diabetes", "type_2_diabetes", "autoimmune_disease", "unknown"],
            default=["unknown"],
        )
        insulin_resistance_signs = st.multiselect(
            "Signs of insulin resistance",
            options=[
                "acanthosis_nigricans",
                "high_triglycerides",
                "low_hdl",
                "hypertension",
                "unknown",
            ],
            default=["unknown"],
        )

        context_inputs = {
            "age_at_onset": None if age_at_onset == 0 else age_at_onset,
            "bmi": None if bmi == 0 else bmi,
            "body_habitus": None if body_habitus == "unknown" else body_habitus,
            "speed_of_onset": None if speed_of_onset == "unknown" else speed_of_onset,
            "ketosis_dka_history": None if ketosis_dka_history == "unknown" else ketosis_dka_history,
            "insulin_required_immediately": None if insulin_required_immediately == "unknown" else insulin_required_immediately,
            "family_history": [] if "unknown" in family_history and len(family_history) == 1 else family_history,
            "insulin_resistance_signs": [] if "unknown" in insulin_resistance_signs and len(insulin_resistance_signs) == 1 else insulin_resistance_signs,
        }

        if context_inputs != st.session_state.context_inputs:
            st.session_state.context_inputs = context_inputs
            st.session_state.context_dirty = True
            st.session_state.type_likelihood_note = ""
            st.session_state.diet_dirty = True

    # Disclaimer
    st.info(
        "**For educational use only.** This is not a substitute for a doctor. "
        "Please consult a healthcare provider for diagnosis and treatment."
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload lab report (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="Supported: PDF, PNG, JPG, TIFF, BMP",
    )

    if uploaded_file is not None:
        # New file uploaded: process and reset chat (key by name to avoid re-processing on rerun)
        current_name = getattr(uploaded_file, "name", "") or ""
        if (
            st.session_state.uploaded_file is None
            or getattr(st.session_state.uploaded_file, "name", "") != current_name
        ):
            logger.info("New report uploaded: %s", current_name)
            with st.spinner("Processing report..."):
                file_bytes = uploaded_file.read()
                logger.info("File read: %s bytes", len(file_bytes))
                report = _run_process_report_graph(file_bytes, current_name)
                st.session_state.structured_report = report if report is not None else []
                st.session_state.messages = []
                st.session_state.uploaded_file = type("F", (), {"name": current_name})()
                st.session_state.diet_guidance = None
                st.session_state.diet_dirty = True
            if report:
                logger.info("Report ready for chat: %s result(s)", len(report))
                st.success(f"Report processed. Found {len(report)} result(s). You can ask questions below.")

    # Chat only when we have a report (even if empty, so user can ask)
    if st.session_state.structured_report is not None:
        # Optional: show short summary
        report = st.session_state.structured_report
        if report:
            normal = sum(1 for r in report if r.get("status") == "normal")
            high_low = sum(1 for r in report if r.get("status") in ("high", "low"))
            critical = sum(1 for r in report if r.get("critical"))
            high_count = sum(1 for r in report if r.get("status") == "high")
            low_count = sum(1 for r in report if r.get("status") == "low")
            st.markdown("<div class='section-title'>Summary</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="kpi-grid">
                  <div class="kpi-card">
                    <div class="kpi-title">Normal</div>
                    <div class="kpi-value kpi-ok">{normal}</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-title">Outside Range</div>
                    <div class="kpi-value kpi-warning">{high_low}</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-title">Critical</div>
                    <div class="kpi-value kpi-danger">{critical}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='muted'>High: {high_count} | Low: {low_count} | Critical: {critical}</div>",
                unsafe_allow_html=True,
            )

            critical_rows = [r for r in report if r.get("critical")]
            if critical_rows:
                st.markdown("<div class='section-title'>Top Critical Findings</div>", unsafe_allow_html=True)
                causes = {
                    "hba1c": "Poor glycemic control, recent hyperglycemia, or uncontrolled diabetes.",
                    "fasting glucose": "Acute hyperglycemia, stress response, or uncontrolled diabetes.",
                    "triglycerides": "Metabolic syndrome, uncontrolled diabetes, alcohol use, or genetic dyslipidemia.",
                    "ldl cholesterol": "Dietary factors, genetics, metabolic syndrome, or hypothyroidism.",
                    "creatinine": "Reduced kidney function, dehydration, or medication effects.",
                    "egfr": "Chronic kidney disease, acute kidney injury, or reduced renal perfusion.",
                    "potassium": "Renal impairment, medications, or cellular shifts.",
                    "sodium": "Fluid imbalance, medications, or endocrine causes.",
                    "alt (sgpt)": "Liver inflammation, fatty liver disease, or medication effects.",
                    "ast (sgot)": "Liver injury, muscle injury, or alcohol-related disease.",
                    "troponin i": "Cardiac injury, demand ischemia, or myocarditis.",
                    "crp": "Acute infection, inflammation, or tissue injury.",
                    "tsh": "Hypothyroidism, thyroiditis, or medication effects.",
                }
                for r in critical_rows[:3]:
                    name = r.get("test_name", "")
                    value = r.get("value", "")
                    unit = r.get("unit", "")
                    fact = r.get("fact", "")
                    status = r.get("status", "")
                    is_critical = "Yes" if r.get("critical") else "No"
                    ref_low = r.get("ref_low")
                    ref_high = r.get("ref_high")
                    ref_str = f"{ref_low if ref_low is not None else '?'} - {ref_high if ref_high is not None else '?'} {unit}".strip()
                    cause = causes.get(str(name).lower(), "Depends on clinical context and history.")
                    st.markdown(
                        f"""
                        <div class="highlight">
                          <strong>{name}</strong><br/>
                          <span class="muted">Result:</span> {value} {unit}<br/>
                          <span class="muted">Reference:</span> {ref_str}<br/>
                          <span class="muted">Status:</span> {status} | <span class="muted">Critical:</span> {is_critical}<br/>
                          <span class="muted">Why flagged:</span> {fact}
                          <br/><span class="muted">Possible causes (non-diagnostic):</span> {cause}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Food & diet guidance section
        st.markdown("<div class='section-title'>Food & Diet Guidance (Non-diagnostic)</div>", unsafe_allow_html=True)
        st.caption('Do you want me to strengthen the diet-scoring weights for age/BMI/body habitus/insulin resistance now, or just prepare it and wait for your "say the word"?')
        if st.button("Regenerate diet guidance"):
            st.session_state.diet_dirty = True

        if st.session_state.diet_guidance is None or st.session_state.diet_dirty:
            with st.spinner("Generating diet guidance..."):
                guidance = get_diet_guidance(report or [], st.session_state.context_inputs)
                if guidance is None:
                    guidance = _build_diet_guidance_fallback(report, st.session_state.context_inputs)
                st.session_state.diet_guidance = guidance
                st.session_state.diet_dirty = False
        else:
            guidance = st.session_state.diet_guidance

        tabs = st.tabs(["Overview", "Foods to Emphasize", "Foods to Limit", "Sample Day Plan"])
        with tabs[0]:
            st.write("These suggestions are general and based on your abnormal results. They are not a diagnosis.")
            st.markdown("- " + "\n- ".join(guidance["notes"]))
        with tabs[1]:
            if guidance["emphasize"]:
                st.markdown("- " + "\n- ".join(guidance["emphasize"]))
            else:
                st.write("No specific food emphasis detected from the results.")
        with tabs[2]:
            if guidance["limit"]:
                st.markdown("- " + "\n- ".join(guidance["limit"]))
            else:
                st.write("No specific food limits detected from the results.")
        with tabs[3]:
            st.markdown("- " + "\n- ".join(guidance["sample_day"]))

        # Type 1 vs Type 2 likelihood section (non-diagnostic)
        st.markdown("<div class='section-title'>Type 1 vs Type 2 Likelihood (Non-diagnostic)</div>", unsafe_allow_html=True)
        if st.session_state.context_dirty:
            st.info("Context updated. Click Generate to refresh the likelihood note.")
        if st.button("Generate likelihood note"):
            with st.spinner("Generating likelihood note..."):
                st.session_state.type_likelihood_note = get_type_likelihood_note(
                    structured_report=report or [],
                    context_inputs=st.session_state.context_inputs,
                )
                st.session_state.context_dirty = False
        if st.session_state.type_likelihood_note:
            st.write(st.session_state.type_likelihood_note)
        else:
            st.write("Add optional context in the sidebar and click Generate.")

        # Chat UI
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your report (e.g. 'Summarize my report' or 'Why is X high?')"):
            logger.info("User message: len=%s", len(prompt))
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_graph = build_chat_graph()
                    result = chat_graph.invoke({
                        "structured_report": report or [],
                        "messages": st.session_state.messages[:-1],
                        "user_message": prompt,
                        "context_inputs": st.session_state.context_inputs,
                    })
                    reply = result.get("assistant_reply") or ""
                st.markdown(reply)
            logger.info("Assistant reply: len=%s", len(reply))
            # Chat graph already appended user + assistant to messages; sync session state
            st.session_state.messages = result.get("messages") or st.session_state.messages
            st.rerun()
    else:
        st.write("Upload a lab report above to get started.")


if __name__ == "__main__":
    main()
