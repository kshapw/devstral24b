#!/usr/bin/env python3
"""Concurrent-user test automation.

Three users (A, B, C) each get their own thread.  In each round the three
users fire their query simultaneously via ``asyncio.gather``.

Output structure (per language):
    * sending_order file  — shows the order queries were dispatched and
      which came back first.
    * per-user files      — full payload / response / TAT for every query.

Output files (English):
    test_results/concurrent_english_sending_order.txt
    test_results/concurrent_english_user_A.txt
    test_results/concurrent_english_user_B.txt
    test_results/concurrent_english_user_C.txt

Output files (Kannada):
    test_results/concurrent_kannada_sending_order.txt
    test_results/concurrent_kannada_user_A.txt
    test_results/concurrent_kannada_user_B.txt
    test_results/concurrent_kannada_user_C.txt

Usage:
    python tests/test_concurrent_users.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from helpers import (
    BASE_URL,
    REQUEST_TIMEOUT,
    ResultWriter,
    async_create_thread,
    async_send_message,
    epoch_to_str,
    timestamp_now,
)

# ======================================================================
# User definitions — all authenticated
# ======================================================================
USERS = ["A", "B", "C"]

USER_CREDS = {
    "A": {"userId": "user_A_101", "authToken": "token_A"},
    "B": {"userId": "user_B_202", "authToken": "token_B"},
    "C": {"userId": "user_C_303", "authToken": "token_C"},
}

# ======================================================================
# Queries per user (3 rounds).
# User A = general questions, B = ecard/status, C = mixed.
# ======================================================================
ENGLISH_QUERIES = {
    "A": [
        "What welfare schemes are available for construction workers?",
        "How do I register as a construction worker?",
        "What benefits does the maternity scheme provide?",
    ],
    "B": [
        "Show me my ecard",
        "What is my application status?",
        "What documents do I need for registration?",
    ],
    "C": [
        "Download my ecard",
        "Is my renewal approved?",
        "What schemes am I eligible for?",
    ],
}

KANNADA_QUERIES = {
    "A": [
        "ಕಟ್ಟಡ ಕಾರ್ಮಿಕರಿಗೆ ಯಾವ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು ಲಭ್ಯವಿದೆ?",
        "ಕಟ್ಟಡ ಕಾರ್ಮಿಕನಾಗಿ ನೋಂದಣಿ ಹೇಗೆ ಮಾಡುವುದು?",
        "ಹೆರಿಗೆ ಯೋಜನೆಯ ಪ್ರಯೋಜನಗಳು ಏನು?",
    ],
    "B": [
        "ನನ್ನ ಇ-ಕಾರ್ಡ್ ತೋರಿಸಿ",
        "ನನ್ನ ಅರ್ಜಿ ಸ್ಥಿತಿ ಏನು?",
        "ನೋಂದಣಿಗೆ ಯಾವ ದಾಖಲೆಗಳು ಬೇಕು?",
    ],
    "C": [
        "ನನ್ನ ಕಾರ್ಡ್ ಡೌನ್\u200cಲೋಡ್ ಮಾಡಿ",
        "ನನ್ನ ನವೀಕರಣ ಅನುಮೋದಿಸಲಾಗಿದೆಯೇ?",
        "ನಾನು ಯಾವ ಯೋಜನೆಗಳಿಗೆ ಅರ್ಹ?",
    ],
}


# ======================================================================
# Runner
# ======================================================================
async def run_concurrent_suite(
    queries: dict[str, list[str]],
    lang_code: str,
    lang_label: str,
):
    """Execute one full concurrent suite (all rounds) for one language."""
    prefix = f"concurrent_{lang_label.lower()}"

    # Result writers
    order_rw = ResultWriter(f"{prefix}_sending_order.txt")
    user_rws = {u: ResultWriter(f"{prefix}_user_{u}.txt") for u in USERS}

    order_rw.header(f"CONCURRENT TEST — {lang_label.upper()} — SENDING ORDER")
    order_rw.w(f"Timestamp : {timestamp_now()}")
    order_rw.w(f"Base URL  : {BASE_URL}")
    order_rw.w(f"Users     : A (general), B (ecard/status), C (mixed)")
    order_rw.w()

    for u in USERS:
        user_rws[u].header(
            f"CONCURRENT TEST — {lang_label.upper()} — USER {u}"
        )
        user_rws[u].w(f"Credentials : {USER_CREDS[u]}")
        user_rws[u].w()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # --- Create threads ------------------------------------------------
        print(f"    Creating threads for A, B, C ...")
        thread_ids: dict[str, str] = {}
        for u in USERS:
            tid, status, body, ms = await async_create_thread(client)
            thread_ids[u] = tid
            user_rws[u].log_call(
                "CREATE THREAD", "POST",
                f"{BASE_URL}/api/chat/threads",
                None, status, body, ms,
            )
            user_rws[u].w(f"Thread ID : {tid}")
            user_rws[u].w()
            print(f"      User {u}: thread={tid}")

        num_rounds = len(queries["A"])

        # --- Round-by-round concurrent dispatch ----------------------------
        for rnd in range(num_rounds):
            q_num = rnd + 1
            print(f"    Round {q_num}/{num_rounds} — sending concurrently ...")

            # Build payloads
            payloads: dict[str, dict] = {}
            for u in USERS:
                payloads[u] = {
                    "message": queries[u][rnd],
                    "userId": USER_CREDS[u]["userId"],
                    "authToken": USER_CREDS[u]["authToken"],
                    "language": lang_code,
                }

            # Fire all three at once
            async def _send(user: str):
                status, body, ms, send_epoch = await async_send_message(
                    client, thread_ids[user], payloads[user],
                )
                return user, status, body, ms, send_epoch

            results = await asyncio.gather(
                _send("A"), _send("B"), _send("C"),
            )

            # Sort by send timestamp for order log
            by_send = sorted(results, key=lambda r: r[4])
            # Sort by elapsed (fastest response first)
            by_response = sorted(results, key=lambda r: r[3])

            # --- Order file ------------------------------------------------
            order_rw.sep("-", 60)
            order_rw.w(f"ROUND {q_num}")
            order_rw.w()

            # Sending order — group into "same time" (within 50 ms)
            groups: list[list[str]] = []
            current_group: list[str] = []
            prev_epoch: float = 0.0
            for user, _s, _b, _m, send_epoch in by_send:
                if not current_group or (send_epoch - prev_epoch) < 0.050:
                    current_group.append(user)
                else:
                    groups.append(current_group)
                    current_group = [user]
                prev_epoch = send_epoch

            if current_group:
                groups.append(current_group)

            order_rw.w("  Queries sent:")
            for grp in groups:
                ts = epoch_to_str(
                    next(r[4] for r in by_send if r[0] == grp[0])
                )
                parts = [
                    f"User {u} — Query {q_num}" for u in grp
                ]
                order_rw.w(f"    [{ts}]  {' | '.join(parts)}")

            order_rw.w()
            order_rw.w("  Response order (fastest first):")
            for rank, (user, status, _body, ms, _ep) in enumerate(by_response, 1):
                order_rw.w(
                    f"    {rank}. User {user} — status={status}  TAT={ms:.0f} ms"
                )
            order_rw.w()

            # --- Per-user detail -------------------------------------------
            for user, status, body, ms, _ep in results:
                q_text = queries[user][rnd]
                user_rws[user].log_call(
                    f"QUERY {q_num}: {q_text[:70]}",
                    "POST",
                    f"{BASE_URL}/api/chat/threads/{thread_ids[user]}/messages",
                    payloads[user], status, body, ms,
                )
                print(
                    f"      User {user}: status={status}  TAT={ms:.0f} ms"
                )

    # Save everything
    order_rw.save()
    for rw in user_rws.values():
        rw.save()


# ======================================================================
# Main
# ======================================================================
async def main():
    print()
    print("=" * 60)
    print("  CONCURRENT USERS TEST AUTOMATION")
    print("=" * 60)

    # Health check
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{BASE_URL}/")
            if r.status_code != 200:
                raise RuntimeError("bad status")
        except Exception:
            print(f"\nERROR: Server not reachable at {BASE_URL}")
            print("Start it first:  uvicorn app.main:app --reload")
            sys.exit(1)
    print(f"\nServer OK at {BASE_URL}\n")

    print("[1/2] English concurrent suite")
    await run_concurrent_suite(ENGLISH_QUERIES, "", "english")

    print()
    print("[2/2] Kannada concurrent suite")
    await run_concurrent_suite(KANNADA_QUERIES, "kn", "kannada")

    print("\nAll concurrent tests complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
