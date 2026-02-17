#!/usr/bin/env python3
"""Single-thread test automation.

Creates ONE thread per language, then sends every category of question
through it (unauthenticated + authenticated).  Finally fetches all
stored messages via GET.

Output files:
    test_results/single_thread_english.txt
    test_results/single_thread_kannada.txt

Usage:
    python tests/test_single_thread.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from helpers import (
    BASE_URL,
    REQUEST_TIMEOUT,
    ResultWriter,
    create_thread,
    send_message,
    get_messages,
    health_check,
    timestamp_now,
)

# ======================================================================
# Test cases — (label, payload_overrides)
# Fields not specified default to empty string.
# ======================================================================

ENGLISH_CASES = [
    # ---------- Unauthenticated ----------
    (
        "GENERAL (unauth) — Available welfare schemes",
        {"message": "What welfare schemes are available for construction workers?"},
    ),
    (
        "GENERAL (unauth) — Registration process",
        {"message": "How do I register as a construction worker?"},
    ),
    (
        "GENERAL (unauth) — Required documents",
        {"message": "What documents do I need for registration?"},
    ),
    (
        "GENERAL (unauth) — Eligibility criteria",
        {"message": "Who is eligible to register under KBOCWWB?"},
    ),
    (
        "ECARD (unauth) — Should return LOGIN_REQUIRED",
        {"message": "Show me my ecard"},
    ),
    (
        "STATUS_CHECK (unauth) — Should return LOGIN_REQUIRED",
        {"message": "What is my application status?"},
    ),
    (
        "OFF-TOPIC (unauth) — Should redirect to KSK topics",
        {"message": "What is the weather today?"},
    ),
    (
        "GREETING (unauth) — Warm greeting expected",
        {"message": "Hello"},
    ),
    # ---------- Authenticated (userId=123, authToken=abc) ----------
    (
        "GENERAL (auth) — Eligible schemes",
        {"message": "What schemes am I eligible for?", "userId": "123", "authToken": "abc"},
    ),
    (
        "GENERAL (auth) — How to apply online",
        {"message": "How can I apply for a scheme online?", "userId": "123", "authToken": "abc"},
    ),
    (
        "ECARD (auth) — Should return ECARD constant",
        {"message": "Download my ecard", "userId": "123", "authToken": "abc"},
    ),
    (
        "ECARD (auth) — Print card variant",
        {"message": "Print my labour card", "userId": "123", "authToken": "abc"},
    ),
    (
        "STATUS_CHECK (auth) — Application status",
        {"message": "Check my application status", "userId": "123", "authToken": "abc"},
    ),
    (
        "STATUS_CHECK (auth) — Renewal status",
        {"message": "Is my renewal approved?", "userId": "123", "authToken": "abc"},
    ),
    (
        "OFF-TOPIC (auth) — Should redirect",
        {"message": "Tell me a joke", "userId": "123", "authToken": "abc"},
    ),
]

KANNADA_CASES = [
    # ---------- Unauthenticated ----------
    (
        "GENERAL (unauth) — ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು",
        {"message": "ಕಟ್ಟಡ ಕಾರ್ಮಿಕರಿಗೆ ಯಾವ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು ಲಭ್ಯವಿದೆ?", "language": "kn"},
    ),
    (
        "GENERAL (unauth) — ನೋಂದಣಿ ಪ್ರಕ್ರಿಯೆ",
        {"message": "ಕಟ್ಟಡ ಕಾರ್ಮಿಕನಾಗಿ ನೋಂದಣಿ ಹೇಗೆ ಮಾಡುವುದು?", "language": "kn"},
    ),
    (
        "GENERAL (unauth) — ಅಗತ್ಯ ದಾಖಲೆಗಳು",
        {"message": "ನೋಂದಣಿಗೆ ಯಾವ ದಾಖಲೆಗಳು ಬೇಕು?", "language": "kn"},
    ),
    (
        "GENERAL (unauth) — ಅರ್ಹತೆ",
        {"message": "KBOCWWB ಅಡಿಯಲ್ಲಿ ನೋಂದಣಿಗೆ ಯಾರು ಅರ್ಹರು?", "language": "kn"},
    ),
    (
        "ECARD (unauth) — LOGIN_REQUIRED expected",
        {"message": "ನನ್ನ ಇ-ಕಾರ್ಡ್ ತೋರಿಸಿ", "language": "kn"},
    ),
    (
        "STATUS_CHECK (unauth) — LOGIN_REQUIRED expected",
        {"message": "ನನ್ನ ಅರ್ಜಿ ಸ್ಥಿತಿ ಏನು?", "language": "kn"},
    ),
    (
        "OFF-TOPIC (unauth) — KSK ಪುನರ್ನಿರ್ದೇಶನ expected",
        {"message": "ಇಂದು ಹವಾಮಾನ ಹೇಗಿದೆ?", "language": "kn"},
    ),
    (
        "GREETING (unauth) — ಶುಭಾಶಯ",
        {"message": "ನಮಸ್ಕಾರ", "language": "kn"},
    ),
    # ---------- Authenticated (userId=123, authToken=abc) ----------
    (
        "GENERAL (auth) — ಅರ್ಹ ಯೋಜನೆಗಳು",
        {"message": "ನಾನು ಯಾವ ಯೋಜನೆಗಳಿಗೆ ಅರ್ಹ?", "userId": "123", "authToken": "abc", "language": "kn"},
    ),
    (
        "GENERAL (auth) — ಆನ್\u200cಲೈನ್ ಅರ್ಜಿ",
        {"message": "ಆನ್\u200cಲೈನ್\u200cನಲ್ಲಿ ಯೋಜನೆಗೆ ಅರ್ಜಿ ಹೇಗೆ ಸಲ್ಲಿಸುವುದು?",
         "userId": "123", "authToken": "abc", "language": "kn"},
    ),
    (
        "ECARD (auth) — ECARD constant expected",
        {"message": "ನನ್ನ ಕಾರ್ಡ್ ಡೌನ್\u200cಲೋಡ್ ಮಾಡಿ", "userId": "123", "authToken": "abc", "language": "kn"},
    ),
    (
        "ECARD (auth) — ಕಾರ್ಡ್ ಪ್ರಿಂಟ್ variant",
        {"message": "ನನ್ನ ಕಾರ್ಮಿಕ ಕಾರ್ಡ್ ಪ್ರಿಂಟ್ ಮಾಡಿ", "userId": "123", "authToken": "abc", "language": "kn"},
    ),
    (
        "STATUS_CHECK (auth) — ಅರ್ಜಿ ಸ್ಥಿತಿ",
        {"message": "ನನ್ನ ಅರ್ಜಿ ಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ", "userId": "123", "authToken": "abc", "language": "kn"},
    ),
    (
        "STATUS_CHECK (auth) — ನವೀಕರಣ ಸ್ಥಿತಿ",
        {"message": "ನನ್ನ ನವೀಕರಣ ಅನುಮೋದಿಸಲಾಗಿದೆಯೇ?", "userId": "123", "authToken": "abc", "language": "kn"},
    ),
    (
        "OFF-TOPIC (auth) — ಪುನರ್ನಿರ್ದೇಶನ expected",
        {"message": "ಜೋಕ್ ಹೇಳಿ", "userId": "123", "authToken": "abc", "language": "kn"},
    ),
]


# ======================================================================
# Runner
# ======================================================================
def run_suite(client: httpx.Client, cases: list, output_file: str, lang_label: str):
    """Run a list of test cases on a single thread, write results to file."""
    rw = ResultWriter(output_file)
    rw.header(f"SINGLE THREAD TEST — {lang_label.upper()}")
    rw.w(f"Timestamp : {timestamp_now()}")
    rw.w(f"Base URL  : {BASE_URL}")
    rw.w()

    # 1. Create thread ------------------------------------------------
    print(f"    Creating thread ...")
    tid, status, body, ms = create_thread(client)
    rw.log_call(
        "CREATE THREAD", "POST",
        f"{BASE_URL}/api/chat/threads",
        None, status, body, ms,
    )
    if not tid:
        rw.w("FATAL: Could not create thread. Aborting.")
        rw.save()
        print("    ERROR: Could not create thread!")
        return
    rw.w(f"Thread ID: {tid}")
    rw.w()
    print(f"    Thread: {tid}")

    # 2. Send each test message ---------------------------------------
    for i, (label, overrides) in enumerate(cases, 1):
        payload = {"message": "", "userId": "", "authToken": "", "language": ""}
        payload.update(overrides)

        tag = f"TEST {i}/{len(cases)}"
        print(f"    [{tag}] {label} ... ", end="", flush=True)

        status, body, ms = send_message(client, tid, payload)
        rw.log_call(
            f"{tag}: {label}",
            "POST",
            f"{BASE_URL}/api/chat/threads/{tid}/messages",
            payload, status, body, ms,
        )
        print(f"status={status}  TAT={ms:.0f}ms")

    # 3. GET all stored messages --------------------------------------
    print(f"    Fetching stored messages ...")
    rw.header("GET ALL STORED MESSAGES")
    status, body, ms = get_messages(client, tid, limit=200, offset=0)
    rw.log_call(
        "GET MESSAGES (limit=200, offset=0)",
        "GET",
        f"{BASE_URL}/api/chat/threads/{tid}/messages?limit=200&offset=0",
        None, status, body, ms,
    )
    print(f"    GET messages: status={status}  TAT={ms:.0f}ms")

    rw.save()


# ======================================================================
# Main
# ======================================================================
def main():
    print()
    print("=" * 60)
    print("  SINGLE THREAD TEST AUTOMATION")
    print("=" * 60)

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        if not health_check(client):
            print(f"\nERROR: Server not reachable at {BASE_URL}")
            print("Start it first:  uvicorn app.main:app --reload")
            sys.exit(1)
        print(f"\nServer OK at {BASE_URL}\n")

        print("[1/2] English test suite")
        run_suite(client, ENGLISH_CASES, "single_thread_english.txt", "English")

        print()
        print("[2/2] Kannada test suite")
        run_suite(client, KANNADA_CASES, "single_thread_kannada.txt", "Kannada")

    print("\nAll single-thread tests complete.\n")


if __name__ == "__main__":
    main()
