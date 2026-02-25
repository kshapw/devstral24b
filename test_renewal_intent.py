import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        # Create thread
        r = await client.post("http://localhost:2024/api/chat/threads")
        thread_id = r.json()["threadId"]
        
        # Send message
        payload = {
            "message": "how to renew",
            "authToken": "",
            "userId": "",
            "language": ""
        }
        r2 = await client.post(
            f"http://localhost:2024/api/chat/threads/{thread_id}/messages", 
            json=payload,
            timeout=120.0
        )
        print("Response:", r2.text)

if __name__ == "__main__":
    asyncio.run(main())
