import httpx
import json
import os
import time

BASE_URL = "https://llm.karmikakendra.com"
OUTPUT_DIR = "extensive_test_results"

TEST_CASES = [
    # Registration & Renewal
    {"msg": "How to do new worker registration?", "lang": "en", "desc": "registration"},
    {"msg": "What documents are required for registration?", "lang": "en", "desc": "registration_docs"},
    {"msg": "ನೋಂದಣಿ ಮಾಡುವುದು ಹೇಗೆ?", "lang": "kn", "desc": "registration_kn"},
    {"msg": "What is the procedure for renewal of labour card?", "lang": "en", "desc": "renewal"},
    
    # Schemes - English
    {"msg": "Tell me about Thayi Magu Sahaya Hasta scheme", "lang": "en", "desc": "thayi_magu"},
    {"msg": "What are the benefits of Delivery Assistance?", "lang": "en", "desc": "delivery"},
    {"msg": "I want to apply for Marriage Assistance, what is the amount?", "lang": "en", "desc": "marriage"},
    {"msg": "What is the pension amount for construction workers?", "lang": "en", "desc": "pension"},
    {"msg": "How much compensation for death in a workplace accident?", "lang": "en", "desc": "accident_death"},
    {"msg": "What are the documents needed for funeral assistance?", "lang": "en", "desc": "funeral_docs"},
    {"msg": "Is there any assistance for major ailments?", "lang": "en", "desc": "major_ailments"},
    
    # Schemes - Kannada
    {"msg": "ಹೆರಿಗೆ ಸಹಾಯಧನ ಎಷ್ಟು?", "lang": "kn", "desc": "delivery_kn"},
    {"msg": "ಮದುವೆ ಸಹಾಯಧನಕ್ಕೆ ಯಾರು ಅರ್ಜಿ ಸಲ್ಲಿಸಬಹುದು?", "lang": "kn", "desc": "marriage_eligibility_kn"},
    {"msg": "ಅಪಘಾತದಲ್ಲಿ ಮರಣ ಹೊಂದಿದರೆ ಎಷ್ಟು ಹಣ ಸಿಗುತ್ತದೆ?", "lang": "kn", "desc": "accident_death_kn"},
    {"msg": "ಪಿಂಚಣಿ ಯೋಜನೆಗೆ ಬೇಕಾದ ದಾಖಲೆಗಳು ಯಾವುವು?", "lang": "kn", "desc": "pension_docs_kn"},
    
    # Guardrails / Out of Scope
    {"msg": "Who is the Chief Minister of Karnataka?", "lang": "en", "desc": "guardrail_cm"},
    {"msg": "What is the weather in Bangalore?", "lang": "en", "desc": "guardrail_weather"},
    {"msg": "ಭಾರತದ ಪ್ರಧಾನ ಮಂತ್ರಿ ಯಾರು?", "lang": "kn", "desc": "guardrail_pm_kn"},
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = httpx.Client(timeout=60.0)
    
    print(f"Starting execution of {len(TEST_CASES)} tests against {BASE_URL}...")
    
    for i, test in enumerate(TEST_CASES, 1):
        filename = f"{i:03d}_{test['desc']}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        print(f"[{i}/{len(TEST_CASES)}] Testing: {test['msg']} ({test['lang']})")
        
        # Create a new thread for each test to ensure isolated context
        r = client.post(f"{BASE_URL}/api/chat/threads")
        if r.status_code != 200:
            result_text = f"FAILED to create thread: {r.status_code} {r.text}"
            print("  -> Failed to create thread")
        else:
            thread_id = r.json().get("id", "")
            if not thread_id:
                # the prod endpoint sometimes returns "id", our local test uses "threadId"
                thread_id = r.json().get("threadId", "")
                
            payload = {
                "message": test["msg"],
                "authToken": "",
                "userId": "",
                "language": test["lang"]
            }
            
            start_time = time.time()
            # Calling the normal endpoint
            r2 = client.post(
                f"{BASE_URL}/api/chat/threads/{thread_id}/messages",
                json=payload
            )
            duration = time.time() - start_time
            
            if r2.status_code in (200, 201):
                ans = r2.json().get("answer", r2.text)
                result_text = f"Time: {duration:.2f}s\n\nQuery: {test['msg']}\nLanguage: {test['lang']}\n\nResponse:\n{ans}"
                print(f"  -> Success ({duration:.2f}s)")
            else:
                result_text = f"FAILED message post: {r2.status_code} {r2.text}"
                print(f"  -> Failed endpoint ({r2.status_code})")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result_text)
            
    print(f"All tests completed. Results saved in ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
