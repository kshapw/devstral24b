import httpx
import json
import datetime

BASE_URL = "https://llm.karmikakendra.com"
OUTPUT_FILE = "remote_api_test_results.txt"

def test_remote():
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"\n--- Test Run at {datetime.datetime.now()} ---\n")
        
        client = httpx.Client(timeout=30.0)
        
        # Create thread
        r = client.post(f"{BASE_URL}/api/chat/threads")
        if r.status_code != 200:
            f.write(f"Failed to create thread: {r.text}\n")
            return
        thread_id = r.json()["id"]
        f.write(f"Thread ID: {thread_id}\n")
        
        # Send message via stream
        payload = {
            "message": "registration",
            "authToken": "",
            "userId": "",
            "language": "en"
        }
        f.write(f"Sending Payload: {json.dumps(payload)}\n")
        f.write("Normal response:\n")
        
        r2 = client.post(
            f"{BASE_URL}/api/chat/threads/{thread_id}/messages",
            json=payload,
        )
        if r2.status_code == 200 or r2.status_code == 201:
            full_output = r2.json().get("answer", r2.text)
        else:
            full_output = f"Error: {r2.status_code} {r2.text}"
        
        f.write(f"{full_output}\n")
        f.write("-" * 40 + "\n")
        print(f"Test completed. Results saved to {OUTPUT_FILE}")
        
if __name__ == "__main__":
    test_remote()
