"""
agent.py
────────
The AI Agent brain. Orchestrates:
  1. RAG retrieval
  2. LLM answer generation
  3. Email send decision  (rule-based OR LLM-based)
  4. Email dispatch via email_agent.py
"""

import requests
from dataclasses import dataclass

from config       import OLLAMA_BASE_URL, OLLAMA_MODEL
from rag          import get_context_and_source, generate_answer, extract_patient_name
from email_agent  import send_patient_report
from logger       import log


# ── Result container ─────────────────────────────────────────
@dataclass
class AgentResult:
    question     : str
    patient_name : str
    answer       : str
    source       : dict
    emailed      : bool


# ─────────────────────────────────────────────────────────────
# EMAIL DECISION STRATEGIES
# ─────────────────────────────────────────────────────────────
def _decide_by_keywords(answer: str) -> bool:
    """Send email if answer contains medical content keywords."""
    keywords = [
        "patient", "diagnosis", "diagnosed", "medication", "prescribed",
        "treatment", "lab", "test result", "history", "condition", "symptoms"
    ]
    answer_lower = answer.lower()
    matched = [kw for kw in keywords if kw in answer_lower]
    if matched:
        log.info(f"Keyword match → email triggered (matched: {matched})")
    return bool(matched)


def _decide_by_llm(question: str, answer: str) -> bool:
    """Ask LLaMA3 itself whether this result warrants an email."""
    prompt = (
        "You are a medical data routing agent.\n"
        f"A user asked: \"{question}\"\n"
        f"The system answered: \"{answer[:400]}\"\n\n"
        "Should this patient information be emailed to the requesting doctor? "
        "Reply with ONLY the word YES or NO. No explanation."
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0, "num_predict": 5}
            },
            timeout=30
        )
        decision = resp.json().get("response", "NO").strip().upper()
        log.info(f"LLM email decision: {decision}")
        return "YES" in decision
    except Exception as e:
        log.warning(f"LLM decision failed, falling back to keyword check: {e}")
        return _decide_by_keywords(answer)


# ─────────────────────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────────────────────
def run_agent(
    question        : str,
    recipient_email : str,
    auto_send       : bool = False,
    use_llm_decision: bool = True
) -> AgentResult:
    """
    Full pipeline: retrieve → generate → decide → email.

    Args:
        question         : Natural-language medical query
        recipient_email  : Email address to send the report to
        auto_send        : If True, always email regardless of content
        use_llm_decision : If True, let LLaMA decide; else use keyword rules

    Returns:
        AgentResult dataclass
    """

    log.info("=" * 55)
    log.info(f"AGENT START")
    log.info(f"Question  : {question}")
    log.info(f"Recipient : {recipient_email}")
    log.info("=" * 55)

    # ── Step 1: Retrieve context from vector DB ───────────────
    log.info("Step 1 — Retrieving from vector store...")
    context, source = get_context_and_source(question)
    log.info(f"Source: {source['file']} | Page {source['page'] + 1}")

    # ── Step 2: Extract patient name ──────────────────────────
    patient_name = extract_patient_name(context)
    log.info(f"Patient: {patient_name}")

    # ── Step 3: Generate answer ───────────────────────────────
    log.info("Step 2 — Generating answer via LLaMA3...")
    answer = generate_answer(question, context)
    log.info(f"Answer (preview): {answer[:150]}...")

    # ── Step 4: Decide whether to email ──────────────────────
    log.info("Step 3 — Email decision...")
    if auto_send:
        should_email = True
        log.info("auto_send=True → will email unconditionally")
    elif use_llm_decision:
        should_email = _decide_by_llm(question, answer)
    else:
        should_email = _decide_by_keywords(answer)

    # ── Step 5: Send email ────────────────────────────────────
    emailed = False
    if should_email:
        log.info("Step 4 — Sending email...")
        emailed = send_patient_report(
            recipient_email = recipient_email,
            question        = question,
            answer          = answer,
            source          = source,
            patient_name    = patient_name
        )
    else:
        log.info("Step 4 — Email skipped (decision: no relevant patient data)")

    log.info(f"AGENT DONE | emailed={emailed}")
    log.info("=" * 55)

    return AgentResult(
        question     = question,
        patient_name = patient_name,
        answer       = answer,
        source       = source,
        emailed      = emailed
    )