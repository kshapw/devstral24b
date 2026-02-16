"""
KSK Chatbot Endpoint Test Suite
================================
Tests all endpoints with 10 English + 10 Kannada queries derived from data/ksk.md.

Usage:
    python test_endpoints.py                        # default: http://localhost:2024
    python test_endpoints.py http://192.168.1.5:2024  # custom server URL
"""

import json
import os
import sys
import time
from datetime import datetime

import httpx


class TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath: str):
        self._stdout = sys.stdout
        self._file = open(filepath, "w", encoding="utf-8")

    def write(self, text: str):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:2024"
TIMEOUT = 180.0  # devstral:24b can be slow on first inference

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

ENGLISH_TESTS = [
    {
        "message": "What is Karmika Seva Kendra and what services does it offer?",
        "language": "",
        "description": "Organization overview + services",
    },
    {
        "message": "How much compensation is given if a worker dies in an accident at the workplace?",
        "language": "",
        "description": "Accident Assistance — workplace death ₹8,00,000",
    },
    {
        "message": "What documents are needed to claim accident assistance?",
        "language": "",
        "description": "Accident Assistance — required documents",
    },
    {
        "message": "Tell me about the Thayi Magu Sahaya Hasta scheme and its benefits",
        "language": "",
        "description": "Thayi Magu — ₹6,000, 3 years, first two children",
    },
    {
        "message": "How much financial support is available for delivery of a child?",
        "language": "",
        "description": "Delivery Assistance — ₹50,000 per delivery",
    },
    {
        "message": "What is the pension amount for workers above 60 years?",
        "language": "",
        "description": "Pension — up to ₹3,000/month",
    },
    {
        "message": "How much money does the family receive when a registered worker dies?",
        "language": "",
        "description": "Funeral & Ex-Gratia — ₹4,000 + ₹1,46,000",
    },
    {
        "message": "What is the marriage assistance amount and how many times can I claim it?",
        "language": "",
        "description": "Marriage Assistance — ₹60,000, max 2 claims",
    },
    {
        "message": "What is the maximum amount available for major ailment treatment?",
        "language": "",
        "description": "Major Ailments — up to ₹2,00,000",
    },
    {
        "message": "What is the weather in Bangalore today?",
        "language": "",
        "description": "Off-topic — should get polite redirect",
    },
]

KANNADA_TESTS = [
    {
        "message": "ಕಾರ್ಮಿಕ ಸೇವಾ ಕೇಂದ್ರ ಎಂದರೇನು ಮತ್ತು ಅದರ ಸೇವೆಗಳು ಯಾವುವು?",
        "language": "kn",
        "description": "Organization overview (Kannada)",
    },
    {
        "message": "ಕೆಲಸದ ಸ್ಥಳದಲ್ಲಿ ಅಪಘಾತದಲ್ಲಿ ಕಾರ್ಮಿಕ ಮರಣ ಹೊಂದಿದರೆ ಎಷ್ಟು ಪರಿಹಾರ ಸಿಗುತ್ತದೆ?",
        "language": "kn",
        "description": "Accident — workplace death (Kannada)",
    },
    {
        "message": "ಅಪಘಾತ ಸಹಾಯಕ್ಕೆ ಯಾವ ದಾಖಲೆಗಳು ಬೇಕು?",
        "language": "kn",
        "description": "Accident — required documents (Kannada)",
    },
    {
        "message": "ತಾಯಿ ಮಗು ಸಹಾಯ ಹಸ್ತ ಯೋಜನೆಯ ಪ್ರಯೋಜನಗಳೇನು?",
        "language": "kn",
        "description": "Thayi Magu benefits (Kannada)",
    },
    {
        "message": "ಹೆರಿಗೆ ಸಹಾಯಧನ ಎಷ್ಟು ಸಿಗುತ್ತದೆ?",
        "language": "kn",
        "description": "Delivery Assistance (Kannada)",
    },
    {
        "message": "60 ವರ್ಷ ಮೇಲ್ಪಟ್ಟ ಕಾರ್ಮಿಕರಿಗೆ ಪಿಂಚಣಿ ಎಷ್ಟು?",
        "language": "kn",
        "description": "Pension (Kannada)",
    },
    {
        "message": "ನೋಂದಾಯಿತ ಕಾರ್ಮಿಕ ಮರಣ ಹೊಂದಿದರೆ ಕುಟುಂಬಕ್ಕೆ ಎಷ್ಟು ಹಣ ಸಿಗುತ್ತದೆ?",
        "language": "kn",
        "description": "Funeral & Ex-Gratia (Kannada)",
    },
    {
        "message": "ಮದುವೆ ಸಹಾಯಧನ ಎಷ್ಟು ಮತ್ತು ಎಷ್ಟು ಬಾರಿ ಪಡೆಯಬಹುದು?",
        "language": "kn",
        "description": "Marriage Assistance (Kannada)",
    },
    {
        "message": "ದೊಡ್ಡ ಕಾಯಿಲೆಗೆ ಎಷ್ಟು ಹಣ ಸಿಗುತ್ತದೆ?",
        "language": "kn",
        "description": "Major Ailments (Kannada)",
    },
    {
        "message": "ಇಂದು ಬೆಂಗಳೂರಿನ ಹವಾಮಾನ ಹೇಗಿದೆ?",
        "language": "kn",
        "description": "Off-topic redirect (Kannada)",
    },
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def print_separator():
    print("=" * 80)


def print_result(index: int, test: dict, status: int, duration: float, answer: str):
    status_icon = "PASS" if status == 201 else f"FAIL ({status})"
    print(f"\n  [{status_icon}] Test {index}: {test['description']}")
    print(f"  Message:  {test['message'][:80]}{'...' if len(test['message']) > 80 else ''}")
    print(f"  Language: {test['language'] or 'en (default)'}")
    print(f"  Time:     {duration:.2f}s")
    print(f"  Answer:   {answer[:200]}{'...' if len(answer) > 200 else ''}")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
def main():
    # Set up output to both console and file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.txt"
    tee = TeeWriter(output_file)
    sys.stdout = tee

    client = httpx.Client(timeout=TIMEOUT)

    print_separator()
    print(f"KSK Chatbot Endpoint Tests")
    print(f"Server: {BASE_URL}")
    print(f"Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    # ------- 1. Health check -------
    print("\n[1] Health Check: GET /")
    try:
        r = client.get(f"{BASE_URL}/")
        print(f"  Status: {r.status_code}  Body: {r.json()}")
        if r.status_code != 200:
            print("  FAIL: Health check failed. Is the server running?")
            sys.exit(1)
    except httpx.ConnectError:
        print(f"  FAIL: Cannot connect to {BASE_URL}. Is the server running?")
        sys.exit(1)

    results = []

    # ------- 2. English tests -------
    print(f"\n{'=' * 80}")
    print("ENGLISH TESTS (10 cases)")
    print(f"{'=' * 80}")

    print("\n  Creating thread...")
    r = client.post(f"{BASE_URL}/api/chat/threads")
    if r.status_code != 201:
        print(f"  FAIL: Could not create thread. Status: {r.status_code}")
        sys.exit(1)
    en_thread_id = r.json()["threadId"]
    print(f"  Thread ID: {en_thread_id}")

    for i, test in enumerate(ENGLISH_TESTS, 1):
        payload = {
            "message": test["message"],
            "authToken": "",
            "userId": "",
            "language": test["language"],
        }
        start = time.time()
        r = client.post(
            f"{BASE_URL}/api/chat/threads/{en_thread_id}/messages",
            json=payload,
        )
        duration = time.time() - start

        if r.status_code == 201:
            answer = r.json().get("answer", "")
        else:
            answer = r.text

        print_result(i, test, r.status_code, duration, answer)
        results.append({
            "index": i,
            "lang": "EN",
            "desc": test["description"],
            "status": r.status_code,
            "duration": duration,
            "pass": r.status_code == 201,
        })

    # ------- 3. Kannada tests -------
    print(f"\n{'=' * 80}")
    print("KANNADA TESTS (10 cases, language='kn')")
    print(f"{'=' * 80}")

    print("\n  Creating thread...")
    r = client.post(f"{BASE_URL}/api/chat/threads")
    if r.status_code != 201:
        print(f"  FAIL: Could not create thread. Status: {r.status_code}")
        sys.exit(1)
    kn_thread_id = r.json()["threadId"]
    print(f"  Thread ID: {kn_thread_id}")

    for i, test in enumerate(KANNADA_TESTS, 1):
        payload = {
            "message": test["message"],
            "authToken": "",
            "userId": "",
            "language": test["language"],
        }
        start = time.time()
        r = client.post(
            f"{BASE_URL}/api/chat/threads/{kn_thread_id}/messages",
            json=payload,
        )
        duration = time.time() - start

        if r.status_code == 201:
            answer = r.json().get("answer", "")
        else:
            answer = r.text

        print_result(10 + i, test, r.status_code, duration, answer)
        results.append({
            "index": 10 + i,
            "lang": "KN",
            "desc": test["description"],
            "status": r.status_code,
            "duration": duration,
            "pass": r.status_code == 201,
        })

    # ------- 4. Error path tests -------
    print(f"\n{'=' * 80}")
    print("ERROR PATH TESTS")
    print(f"{'=' * 80}")

    error_tests = [
        {
            "description": "Invalid thread ID format (non-UUID) → 400",
            "method": "POST",
            "url": f"{BASE_URL}/api/chat/threads/not-a-uuid/messages",
            "json": {"message": "Hello", "authToken": "", "userId": "", "language": ""},
            "expected_status": 400,
        },
        {
            "description": "Non-existent thread → 404",
            "method": "POST",
            "url": f"{BASE_URL}/api/chat/threads/00000000-0000-0000-0000-000000000000/messages",
            "json": {"message": "Hello", "authToken": "", "userId": "", "language": ""},
            "expected_status": 404,
        },
        {
            "description": "Invalid language code → 422",
            "method": "POST",
            "url": f"{BASE_URL}/api/chat/threads/{en_thread_id}/messages",
            "json": {"message": "Hello", "authToken": "", "userId": "", "language": "zz_invalid"},
            "expected_status": 422,
        },
        {
            "description": "Blank message (whitespace only) → 422",
            "method": "POST",
            "url": f"{BASE_URL}/api/chat/threads/{en_thread_id}/messages",
            "json": {"message": "   ", "authToken": "", "userId": "", "language": ""},
            "expected_status": 422,
        },
        {
            "description": "GET messages with invalid thread ID → 400",
            "method": "GET",
            "url": f"{BASE_URL}/api/chat/threads/bad-id/messages",
            "json": None,
            "expected_status": 400,
        },
        {
            "description": "GET messages for non-existent thread → 404",
            "method": "GET",
            "url": f"{BASE_URL}/api/chat/threads/00000000-0000-0000-0000-000000000000/messages",
            "json": None,
            "expected_status": 404,
        },
        {
            "description": "GET messages for valid thread → 200",
            "method": "GET",
            "url": f"{BASE_URL}/api/chat/threads/{en_thread_id}/messages",
            "json": None,
            "expected_status": 200,
        },
    ]

    for i, test in enumerate(error_tests, 1):
        start = time.time()
        if test["method"] == "POST":
            r = client.post(test["url"], json=test["json"])
        else:
            r = client.get(test["url"])
        duration = time.time() - start

        passed = r.status_code == test["expected_status"]
        status_icon = "PASS" if passed else "FAIL"
        print(f"\n  [{status_icon}] Error Test {i}: {test['description']}")
        print(f"  Expected: {test['expected_status']}  Got: {r.status_code}  Time: {duration:.2f}s")
        if not passed:
            print(f"  Body: {r.text[:200]}")

        results.append({
            "index": 20 + i,
            "lang": "ERR",
            "desc": test["description"],
            "status": r.status_code,
            "duration": duration,
            "pass": passed,
        })

    # ------- 5. Streaming endpoint test -------
    print(f"\n{'=' * 80}")
    print("STREAMING ENDPOINT TEST")
    print(f"{'=' * 80}")

    print("\n  Creating thread for streaming test...")
    r = client.post(f"{BASE_URL}/api/chat/threads")
    if r.status_code == 201:
        stream_thread_id = r.json()["threadId"]
        print(f"  Thread ID: {stream_thread_id}")

        print("\n  Sending streaming request...")
        start = time.time()
        with client.stream(
            "POST",
            f"{BASE_URL}/api/chat/threads/{stream_thread_id}/messages/stream",
            json={"message": "What is KSK?", "authToken": "", "userId": "", "language": ""},
        ) as response:
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    events.append(event)

        duration = time.time() - start
        chunk_events = [e for e in events if e.get("event") == "chunk"]
        done_events = [e for e in events if e.get("event") == "done"]
        error_events = [e for e in events if e.get("event") == "error"]

        stream_pass = len(chunk_events) > 0 and len(done_events) == 1 and len(error_events) == 0
        status_icon = "PASS" if stream_pass else "FAIL"
        print(f"\n  [{status_icon}] Streaming Test")
        print(f"  Chunks: {len(chunk_events)}  Done events: {len(done_events)}  Errors: {len(error_events)}")
        print(f"  Time: {duration:.2f}s")
        if done_events:
            full_answer = done_events[0].get("fullAnswer", "")
            print(f"  Answer: {full_answer[:200]}{'...' if len(full_answer) > 200 else ''}")

        results.append({
            "index": 30,
            "lang": "STREAM",
            "desc": "SSE streaming endpoint",
            "status": 200 if stream_pass else 0,
            "duration": duration,
            "pass": stream_pass,
        })
    else:
        print(f"  FAIL: Could not create thread for streaming test. Status: {r.status_code}")
        results.append({
            "index": 30,
            "lang": "STREAM",
            "desc": "SSE streaming endpoint",
            "status": r.status_code,
            "duration": 0,
            "pass": False,
        })

    # ------- 6. Summary -------
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    passed = sum(1 for r in results if r["pass"])
    failed = len(results) - passed
    en_times = [r["duration"] for r in results if r["lang"] == "EN" and r["pass"]]
    kn_times = [r["duration"] for r in results if r["lang"] == "KN" and r["pass"]]

    print(f"\n  Total:  {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    if en_times:
        print(f"\n  English avg response time: {sum(en_times) / len(en_times):.2f}s")
    if kn_times:
        print(f"  Kannada avg response time: {sum(kn_times) / len(kn_times):.2f}s")

    if failed:
        print("\n  FAILED TESTS:")
        for r in results:
            if not r["pass"]:
                print(f"    - Test {r['index']}: {r['desc']} (HTTP {r['status']})")

    print()
    print(f"Results saved to: {output_file}")
    print()
    client.close()
    tee.close()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
