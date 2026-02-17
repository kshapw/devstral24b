import os
import re

RESULTS_DIR = "/Users/kshamawari/Downloads/tar2/test_results"

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    filename = os.path.basename(filepath)
    print(f"Analyzing {filename}...")

    # Pattern to find test blocks
    # [TEST 1/15: ...] or [TEST 1: ...] or [QUERY 1: ...] or [SETUP: ...] or [CREATE THREAD]
    # We look for lines starting with dashed lines, then a header like [X]
    
    # We'll split by separator lines
    sections = re.split(r'-{60,}', content)
    
    total = 0
    passed = 0
    failed = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        lines = section.split('\n')
        header = lines[0].strip()
        
        # Check if it's a test block
        if not (header.startswith('[') and header.endswith(']')):
            continue
            
        # Determine expected status
        expected_status = 201 # Default for message posts
        if "GET" in header or "Health" in header or "STREAM" in header:
            expected_status = 200
        
        # Look for "expect <code" in header
        match_expect = re.search(r'expect\s+(\d+)', header, re.IGNORECASE)
        if match_expect:
            expected_status = int(match_expect.group(1))
        elif "unauth" in header and "LOGIN_REQUIRED" in header:
             expected_status = 201 # The status is 201 but body has LOGIN_REQUIRED
        elif "ECARD" in header and "unauth" in header:
             expected_status = 201
        elif "STATUS_CHECK" in header and "unauth" in header:
             expected_status = 201
             
        # Extract actual status
        actual_status = None
        response_body = ""
        for line in lines:
            if line.strip().startswith("Status :"):
                try:
                    actual_status = int(line.split(':')[1].strip())
                except:
                    pass
            if "Response:" in line:
                # Naive capture of response, actually we just need to detect specific strings for some tests
                pass
        
        if actual_status is None:
            # Maybe a setup block without status or just noise
            continue
            
        total += 1
        
        # Special handling for "LOGIN_REQUIRED" tests which return 201 but with specific body
        is_login_test = "LOGIN_REQUIRED" in header or ("unauth" in header and ("ECARD" in header or "STATUS_CHECK" in header))
        
        if is_login_test:
            if actual_status == 201 and "<<LOGIN_MODAL_REQUIRED>>" in section:
                passed += 1
            else:
                print(f"  FAIL: {header}")
                print(f"    Expected 201 with LOGIN_REQUIRED, got {actual_status}")
                failed += 1
        elif actual_status == expected_status:
            passed += 1
        else:
            print(f"  FAIL: {header}")
            print(f"    Expected {expected_status}, got {actual_status}")
            failed += 1

    return total, passed, failed

def main():
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.txt')]
    
    grand_total = 0
    grand_passed = 0
    grand_failed = 0
    
    print(f"Found {len(files)} result files.")
    
    for f in sorted(files):
        t, p, fl = analyze_file(os.path.join(RESULTS_DIR, f))
        grand_total += t
        grand_passed += p
        grand_failed += fl
        print(f"  -> {f}: {p}/{t} passed ({fl} failed)\n")
        
    print("="*40)
    print(f"TOTAL SUMMARY:")
    print(f"Total Tests: {grand_total}")
    print(f"Passed:      {grand_passed}")
    print(f"Failed:      {grand_failed}")
    if grand_total > 0:
        print(f"Success Rate: {grand_passed/grand_total*100:.1f}%")

if __name__ == "__main__":
    main()
