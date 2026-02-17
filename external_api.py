import requests
from urllib.parse import urlparse

# ------------------------------------------------------------------
# Configuration (Hardcoded for Production to bypass environment errors)
# ------------------------------------------------------------------
# URL taken from your .env file's BASE_URL_PRODUCTION
# backend_base_url = "https://apikbocwwb.karnataka.gov.in/api"
# backend_base_url = "https://apikbocwwb.karnataka.gov.in/preprod/api"
backend_base_url = "https://staging.ka.karmikakendra.com/api/"
# Derive base origin (e.g. https://apikbocwwb.karnataka.gov.in)
parsed_url = urlparse(backend_base_url)
BASE_ORIGIN = f"{parsed_url.scheme}://{parsed_url.netloc}"

def fetch_user_data(user_id: str, auth_token: str) -> dict:
    """
    Fetches user data from multiple endpoints and aggregates them.
    Prints detailed debugging information instead of using logging.
    """

    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Authorization': f'Bearer {auth_token}',
        'Content-Type': 'application/json',
        'Origin': BASE_ORIGIN,
        'Referer': f'{BASE_ORIGIN}/u/home',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
    }

    aggregated_data = {
        "user_id": user_id,
        "schemes": None,
        "renewal_date": None,
        "registration_details": None,
        "fetch_status": "partial"
    }

    print("\nStarting user data aggregation")
    print(f"Target User ID           : {user_id}")
    print(f"Base API URL             : {backend_base_url}")
    print("This function will call 3 APIs:")
    print("1. Schemes eligibility")
    print("2. Renewal date")
    print("3. User registration details")
    print("--------" * 20)

    # ------------------------------------------------------------------
    # 1. Get Schemes by Labour (Enhanced Processing)
    # ------------------------------------------------------------------
    try:
        url = f"{backend_base_url}/schemes/get_schemes_by_labor"
        payload = {
            "board_id": 1,
            "labour_user_id": int(user_id) if user_id.isdigit() else user_id
        }

        print("SCHEMES API CALL")
        print(f"Purpose                  : Fetch schemes applicable to the labour user")
        print(f"Request URL              : {url}")
        print(f"Request Payload          : {payload}")

        response = requests.post(url, headers=headers, json=payload, timeout=10)

        print(f"HTTP Status Code         : {response.status_code}")
        
        if response.status_code == 200:
            raw_data = response.json()
            # Check if 'data' key exists and is a list
            schemes_list = raw_data.get("data", []) if isinstance(raw_data, dict) else []
            
            if not schemes_list:
                print("No schemes found in response.")
                aggregated_data["schemes"] = "No schemes applied."
            else:
                print(f"Raw schemes count: {len(schemes_list)}. Processing deduplication and status...")
                
                # --- Step 1: Deduplication by scheme_id (keeping latest applied_date) ---
                from datetime import datetime
                unique_schemes = {}
                
                for scheme in schemes_list:
                    s_id = scheme.get("scheme_id")
                    app_date_str = scheme.get("applied_date")
                    
                    if not s_id: 
                        continue
                        
                    # Parse date
                    current_dt = datetime.min
                    if app_date_str:
                        try:
                            # Handle ISO format Z (e.g. 2024-08-02T00:00:00.000Z)
                            curr_str = app_date_str.replace("Z", "+00:00")
                            current_dt = datetime.fromisoformat(curr_str)
                        except Exception as e:
                            print(f"Date parse error for scheme {s_id}: {e}")

                    if s_id not in unique_schemes:
                        unique_schemes[s_id] = {"data": scheme, "date": current_dt}
                    else:
                        # Compare dates
                        if current_dt > unique_schemes[s_id]["date"]:
                            unique_schemes[s_id] = {"data": scheme, "date": current_dt}
                
                processed_list = [v["data"] for v in unique_schemes.values()]
                print(f"Unique schemes count: {len(processed_list)}")
                
                # --- Step 2 & 3: Fetch Status and Rejection Reasons ---
                final_schemes_info = []
                
                for scheme in processed_list:
                    scheme_id = scheme.get("scheme_id")
                    app_code = scheme.get("scheme_application_code")
                    scheme_name = scheme.get("scheme_name", "Unknown Scheme")
                    
                    print(f"Fetching status for Scheme: {scheme_name} ({app_code})...")
                    
                    # Call Status API
                    status_url = f"{backend_base_url}/public/schemes/status"
                    status_payload = {
                        "schemeId": scheme_id,
                        "schemeApplicationCode": app_code,
                        "mobileNumber": ""
                    }
                    
                    try:
                        status_resp = requests.post(status_url, json=status_payload, timeout=5)
                        status_data_full = status_resp.json()
                        status_success = status_data_full.get("success", False)
                        status_items = status_data_full.get("data", [])
                        
                        scheme_status_text = "Status check failed"
                        reasons_list = []
                        
                        if status_success and status_items:
                            # Use the first item from status data
                            item = status_items[0]
                            application_status = item.get("application_status")
                            # "status" field has the descriptive text
                            full_status_desc = item.get("status") 
                            avail_id = item.get("id")
                            
                            scheme_status_text = f"Application Status: {application_status}. ({full_status_desc})"
                            
                            # Validates Rejection
                            if application_status == "Rejected" and avail_id:
                                print(f"Scheme Rejected. Fetching reason for ID {avail_id}...")
                                # Call Rejection Reason API
                                reason_url = f"{backend_base_url}/public/schemes/rejection-reason?availId={avail_id}&reasonType=FINAL"
                                try:
                                    r_resp = requests.get(reason_url, timeout=5)
                                    r_data = r_resp.json()
                                    if r_data.get("success"):
                                        reasons_data = r_data.get("data", [])
                                        for r in reasons_data:
                                            # Collect English reasons
                                            if r.get("rejection_reason"):
                                                reasons_list.append(r.get("rejection_reason"))
                                except Exception as re:
                                    print(f"Error fetching rejection reason: {re}")
                        
                        # Format for LLM
                        info_block = {
                            "Scheme Name": scheme_name,
                            "Applied Date": scheme.get("applied_date", "").split("T")[0], # Simple date
                            "Status Details": scheme_status_text
                        }
                        if reasons_list:
                            info_block["Rejection Reasons"] = "; ".join(reasons_list)
                            
                        # Add raw status for programmatic checks if needed, but phrased for LLM
                        final_schemes_info.append(info_block)
                            
                    except Exception as se:
                        print(f"Error fetching status for {scheme_name}: {se}")
                        final_schemes_info.append({
                            "Scheme Name": scheme_name,
                            "Status": "Could not fetch real-time status."
                        })

                aggregated_data["schemes"] = {"data": final_schemes_info} # Wrap in data key to match agent.py expectation
                print("Schemes processed successfully.")

        else:
            aggregated_data["schemes_error"] = response.text
            print("ERROR: Failed to fetch schemes data.")

    except Exception as e:
        aggregated_data["schemes_error"] = str(e)
        print("EXCEPTION while fetching schemes:")
        print(str(e))

    print("--------" * 20)

    # ------------------------------------------------------------------
    # 2. Get Renewal Date
    # ------------------------------------------------------------------
    try:
        url = f"{backend_base_url}/user/get-renewal-date"
        payload = {"user_id": str(user_id)}

        print("RENEWAL DATE API CALL")
        print(f"Purpose                  : Fetch the card / registration renewal date")
        print(f"Request URL              : {url}")
        print(f"Request Payload          : {payload}")

        response = requests.post(url, headers=headers, json=payload, timeout=10)

        print(f"HTTP Status Code         : {response.status_code}")
        print(f"Raw Response Body        : {response.text}")

        if response.status_code == 200:
            aggregated_data["renewal_date"] = response.json()
            print("Renewal date successfully fetched and stored.")
        else:
            aggregated_data["renewal_date_error"] = response.text
            print("ERROR: Failed to fetch renewal date.")

    except Exception as e:
        aggregated_data["renewal_date_error"] = str(e)
        print("EXCEPTION while fetching renewal date:")
        print(str(e))

    print("--------" * 20)

    # 3. Get User Registration Details (Enhanced Processing)
    # ------------------------------------------------------------------
    try:
        url = f"{backend_base_url}/user/get-user-registration-details"
        payload = {
            "key": "user_id",
            "value": str(user_id),
            "board_id": 1,
            "procedure_name": "all"
        }

        print("REGISTRATION DETAILS API CALL")
        print("Purpose                  : Fetch complete user registration profile")
        print(f"Request URL              : {url}")
        print(f"Request Payload          : {payload}")

        response = requests.post(url, headers=headers, json=payload, timeout=10)

        print(f"HTTP Status Code         : {response.status_code}")

        if response.status_code == 200:
            reg_resp_data = response.json()
            
            # Extract Registration Code
            reg_code = None
            personal_details = []
            if reg_resp_data.get("success"):
                data_block = reg_resp_data.get("data", {})
                if data_block:
                    personal_details = data_block.get("personal_details", [])
                    if personal_details and len(personal_details) > 0:
                        reg_code = personal_details[0].get("registration_code")
            
            final_reg_status = "Registration details not found."
            
            if reg_code:
                print(f"Found Registration Code: {reg_code}. Checking Status...")
                
                # 1. Check Registration Status
                status_url = f"{backend_base_url}/public/labour/status"
                reg_status_payload = {"type":"register", "applicationNumber": reg_code, "mobileNumber":""}
                
                try:
                    r_stat_resp = requests.post(status_url, json=reg_status_payload, timeout=5)
                    r_stat_data = r_stat_resp.json()
                    
                    if r_stat_data.get("success") and r_stat_data.get("data"):
                        r_data = r_stat_data.get("data")
                        status_str = r_data.get("status")
                        labour_user_id_stat = r_data.get("labour_user_id")
                        cert_id = r_data.get("labour_work_certificate_id")
                        
                        final_reg_status = f"Registration Status: {status_str}."
                        
                        if status_str == "Approved":
                            # 2. Check Renewal Status if Approved
                            print("Registration Approved. Checking Renewal Status...")
                            ren_status_payload = {"type":"renewal", "applicationNumber": reg_code, "mobileNumber":""}
                            ren_resp = requests.post(status_url, json=ren_status_payload, timeout=5)
                            ren_data_full = ren_resp.json()
                            
                            if ren_data_full.get("success") and ren_data_full.get("data"):
                                ren_data = ren_data_full.get("data")
                                ren_status_str = ren_data.get("status")
                                ren_cert_id = ren_data.get("labour_work_certificate_id")
                                
                                final_reg_status += f" Renewal Status: {ren_status_str}."
                                
                                if ren_status_str == "Rejected":
                                    # Fetch Rejection Reason for Renewal
                                    print("Renewal Rejected. Fetching reason...")
                                    rej_url = f"{backend_base_url}/public/v2/registration-renewal/rejection-reason"
                                    rej_payload = {
                                        "labourUserId": labour_user_id_stat,
                                        "certificateId": ren_cert_id,
                                        "reasonType": "FINAL"
                                    }
                                    rej_resp = requests.post(rej_url, json=rej_payload, timeout=5)
                                    rej_data_full = rej_resp.json()
                                    
                                    reasons = []
                                    if rej_data_full.get("success"):
                                        for r in rej_data_full.get("data", []):
                                            if r.get("rejection_reason"):
                                                reasons.append(r.get("rejection_reason"))
                                    
                                    if reasons:
                                        final_reg_status += f" (Reason: {'; '.join(reasons)})"

                        elif status_str == "Rejected":
                            # Fetch Rejection Reason for Registration
                            print("Registration Rejected. Fetching reason...")
                            rej_url = f"{backend_base_url}/public/v2/registration-renewal/rejection-reason"
                            rej_payload = {
                                "labourUserId": labour_user_id_stat,
                                "certificateId": cert_id,
                                "reasonType": "FINAL"
                            }
                            rej_resp = requests.post(rej_url, json=rej_payload, timeout=5)
                            rej_data_full = rej_resp.json()
                            
                            reasons = []
                            if rej_data_full.get("success"):
                                for r in rej_data_full.get("data", []):
                                    if r.get("rejection_reason"):
                                        reasons.append(r.get("rejection_reason"))
                            if reasons:
                                final_reg_status += f" (Reason: {'; '.join(reasons)})"
                                
                except Exception as ex:
                    print(f"Error checking registration status logic: {ex}")
                    final_reg_status += " (verification failed)"
            else:
                print("Registration Code not found in initial details.")

            aggregated_data["registration_details"] = {"summary": final_reg_status}
            print("Registration details processed and stored.")
            
        else:
            aggregated_data["registration_details_error"] = response.text
            print("ERROR: Failed to fetch registration details.")

    except Exception as e:
        aggregated_data["registration_details_error"] = str(e)
        print("EXCEPTION while fetching registration details:")
        print(str(e))
        
    print("--------" * 20)

    aggregated_data["fetch_status"] = "completed"

    print("DATA AGGREGATION COMPLETED")
    print("Final Aggregated Result:")
    print(aggregated_data)
    print("--------" * 20)

    return aggregated_data