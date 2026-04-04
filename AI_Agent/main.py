"""
main.py
───────
CLI entry point — run the agent directly from terminal.

Usage:
    python main.py
    python main.py --stream          # stream answer token by token
    python main.py --no-email        # skip email sending
"""

import argparse
from rag    import initialize_rag, stream_answer
from agent  import run_agent
from logger import log


def parse_args():
    parser = argparse.ArgumentParser(description="Patient RAG Email Agent")
    parser.add_argument("--stream",   action="store_true", help="Stream answer to terminal")
    parser.add_argument("--no-email", action="store_true", help="Skip email sending")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Boot ─────────────────────────────────────────────────
    initialize_rag()

    # ── Configure your test query here ───────────────────────
    QUESTION        = "What is the diagnosis and prescribed medication for Patient 132?"
    RECIPIENT_EMAIL = "doctor@hospital.com"   # ← change this

    # ── Stream mode (no email) ────────────────────────────────
    if args.stream:
        log.info(f"Streaming answer for: {QUESTION}\n")
        for token in stream_answer(QUESTION):
            print(token, end="", flush=True)
        print()  # newline at end
        return

    # ── Full agent mode ───────────────────────────────────────
    result = run_agent(
        question         = QUESTION,
        recipient_email  = RECIPIENT_EMAIL,
        auto_send        = not args.no_email,
        use_llm_decision = False   # change to True to let LLM decide
    )

    # ── Print result summary ──────────────────────────────────
    print("\n" + "─" * 55)
    print(f"  Patient : {result.patient_name}")
    print(f"  Emailed : {'✅ Yes' if result.emailed else '❌ No'}")
    print(f"  Source  : {result.source['file']} | Page {result.source['page']+1}")
    print("─" * 55)
    print(f"\n{result.answer}\n")


if __name__ == "__main__":
    main()