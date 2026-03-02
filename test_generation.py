import httpx
import sys
import time

BASE_URL = "http://localhost:2024"

def test_query(message):
    client = httpx.Client(timeout=180.0)
    
    print(f"\nCreating thread for: '{message}'")
    r = client.post(f"{BASE_URL}/api/chat/threads")
    thread_id = r.json()["threadId"]
    
    print(f"Sending message...")
    payload = {
        "message": message,
        "authToken": "",
        "userId": "",
        "language": ""
    }
    
    start = time.time()
    r = client.post(
        f"{BASE_URL}/api/chat/threads/{thread_id}/messages",
        json=payload,
    )
    duration = time.time() - start
    
    if r.status_code == 201:
        answer = r.json().get("answer", "")
        print(f"Success ({duration:.2f}s):\n{answer}")
    else:
        print(f"Error ({r.status_code}): {r.text}")
        
    client.close()

if __name__ == "__main__":
    queries = ["Registration", "Renewal", "Worker Registration", "Worker Renewal"]
    for q in queries:
        test_query(q)
