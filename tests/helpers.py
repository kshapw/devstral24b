"""Shared utilities for API test automation.

All test scripts import from this module for consistent API calls,
result logging, and output formatting.
"""

import json
import os
import time
from datetime import datetime

import httpx

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
REQUEST_TIMEOUT = 180.0  # seconds — generous for LLM responses


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def results_path(filename: str) -> str:
    ensure_results_dir()
    return os.path.join(RESULTS_DIR, filename)


def pretty_json(obj) -> str:
    """Pretty-print a JSON-serializable object or JSON string."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return obj
    return json.dumps(obj, indent=2, ensure_ascii=False)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def epoch_to_str(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# Result writer — collects lines, writes a .txt file
# ---------------------------------------------------------------------------
class ResultWriter:
    """Accumulates test output and writes to a .txt file."""

    def __init__(self, filename: str):
        self.filepath = results_path(filename)
        self._lines: list[str] = []
        print(f"    -> Will save to: {self.filepath}")

    def w(self, text: str = ""):
        self._lines.append(text)

    def sep(self, char: str = "=", width: int = 80):
        self._lines.append(char * width)

    def header(self, title: str):
        self.w()
        self.sep()
        self.w(f"  {title}")
        self.sep()
        self.w()

    def log_call(
        self,
        label: str,
        method: str,
        url: str,
        payload: dict | None,
        status: int,
        body: str,
        elapsed_ms: float,
    ):
        self.sep("-", 60)
        self.w(f"[{label}]")
        self.w(f"  Method : {method}")
        self.w(f"  URL    : {url}")
        if payload is not None:
            self.w("  Payload:")
            for line in pretty_json(payload).split("\n"):
                self.w(f"    {line}")
        self.w(f"  Status : {status}")
        self.w("  Response:")
        for line in pretty_json(body).split("\n"):
            self.w(f"    {line}")
        self.w(f"  Turnaround Time: {elapsed_ms:.2f} ms")
        self.w()

    def save(self):
        content = "\n".join(self._lines)
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"    -> Saved: {self.filepath}")


# ---------------------------------------------------------------------------
# Sync API calls (single thread & edge case tests)
# ---------------------------------------------------------------------------
def create_thread(client: httpx.Client) -> tuple[str, int, str, float]:
    """POST /api/chat/threads.  Returns (thread_id, status, body, elapsed_ms)."""
    url = f"{BASE_URL}/api/chat/threads"
    t0 = time.perf_counter()
    r = client.post(url)
    ms = (time.perf_counter() - t0) * 1000
    tid = ""
    try:
        tid = r.json().get("threadId", "")
    except Exception:
        pass
    return tid, r.status_code, r.text, ms


def send_message(
    client: httpx.Client, thread_id: str, payload: dict
) -> tuple[int, str, float]:
    """POST /api/chat/threads/{id}/messages.  Returns (status, body, elapsed_ms)."""
    url = f"{BASE_URL}/api/chat/threads/{thread_id}/messages"
    t0 = time.perf_counter()
    r = client.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    ms = (time.perf_counter() - t0) * 1000
    return r.status_code, r.text, ms


def send_raw(
    client: httpx.Client, thread_id: str, raw_body: str
) -> tuple[int, str, float]:
    """POST raw JSON string (for malformed-payload tests)."""
    url = f"{BASE_URL}/api/chat/threads/{thread_id}/messages"
    t0 = time.perf_counter()
    r = client.post(url, content=raw_body.encode(), headers={"Content-Type": "application/json"})
    ms = (time.perf_counter() - t0) * 1000
    return r.status_code, r.text, ms


def get_messages(
    client: httpx.Client, thread_id: str, **params
) -> tuple[int, str, float]:
    """GET /api/chat/threads/{id}/messages.  Returns (status, body, elapsed_ms)."""
    url = f"{BASE_URL}/api/chat/threads/{thread_id}/messages"
    t0 = time.perf_counter()
    r = client.get(url, params=params)
    ms = (time.perf_counter() - t0) * 1000
    return r.status_code, r.text, ms


def send_stream(
    client: httpx.Client, thread_id: str, payload: dict
) -> tuple[int, str, float]:
    """POST /api/chat/threads/{id}/messages/stream.  Returns (status, body, elapsed_ms).

    Collects the full SSE body as a string.
    """
    url = f"{BASE_URL}/api/chat/threads/{thread_id}/messages/stream"
    t0 = time.perf_counter()
    with client.stream("POST", url, json=payload, timeout=REQUEST_TIMEOUT) as r:
        body = r.read().decode("utf-8", errors="replace")
        status = r.status_code
    ms = (time.perf_counter() - t0) * 1000
    return status, body, ms


def health_check(client: httpx.Client) -> bool:
    """Quick check that server is reachable."""
    try:
        r = client.get(f"{BASE_URL}/", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Async API calls (concurrent tests)
# ---------------------------------------------------------------------------
async def async_create_thread(
    client: httpx.AsyncClient,
) -> tuple[str, int, str, float]:
    url = f"{BASE_URL}/api/chat/threads"
    t0 = time.perf_counter()
    r = await client.post(url)
    ms = (time.perf_counter() - t0) * 1000
    tid = ""
    try:
        tid = r.json().get("threadId", "")
    except Exception:
        pass
    return tid, r.status_code, r.text, ms


async def async_send_message(
    client: httpx.AsyncClient, thread_id: str, payload: dict
) -> tuple[int, str, float, float]:
    """Returns (status, body, elapsed_ms, send_epoch)."""
    url = f"{BASE_URL}/api/chat/threads/{thread_id}/messages"
    send_epoch = time.time()
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    ms = (time.perf_counter() - t0) * 1000
    return r.status_code, r.text, ms, send_epoch
