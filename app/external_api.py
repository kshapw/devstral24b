"""
External API client — fetches user data from the Karnataka KBOCWWB backend.

This is a faithful async port of the proven working user_service.py,
using httpx.AsyncClient instead of synchronous requests.
"""

import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_parsed = urlparse(settings.BACKEND_API_URL)
_BASE_ORIGIN = f"{_parsed.scheme}://{_parsed.netloc}"


def _build_headers(auth_token: str) -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Origin": _BASE_ORIGIN,
        "Referer": f"{_BASE_ORIGIN}/u/home",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/143.0.0.0 Safari/537.36"
        ),
    }


async def fetch_user_data(
    client: httpx.AsyncClient, user_id: str, auth_token: str
) -> dict:
    """
    Fetches user data from multiple endpoints and aggregates them.
    Direct async port of the proven working user_service.py.
    """
    base_url = settings.BACKEND_API_URL.rstrip("/")
    headers = _build_headers(auth_token)

    aggregated_data = {
        "user_id": user_id,
        "schemes": None,
        "renewal_date": None,
        "registration_details": None,
        "fetch_status": "partial",
    }

    print(f"\n[EXT-API] Starting user data aggregation for user_id={user_id}")
    print(f"[EXT-API] Base API URL: {base_url}")

    # ------------------------------------------------------------------
    # 1. Get Schemes by Labour (Enhanced Processing)
    # ------------------------------------------------------------------
    try:
        url = f"{base_url}/schemes/get_schemes_by_labor"
        payload = {
            "board_id": 1,
            "labour_user_id": int(user_id) if user_id.isdigit() else user_id,
        }

        print(f"[EXT-API] SCHEMES API CALL: {url}")
        resp = await client.post(url, headers=headers, json=payload, timeout=10)
        print(f"[EXT-API] Schemes HTTP Status: {resp.status_code}")

        if resp.status_code == 200:
            raw_data = resp.json()
            schemes_list = raw_data.get("data", []) if isinstance(raw_data, dict) else []

            if not schemes_list:
                print("[EXT-API] No schemes found in response.")
                aggregated_data["schemes"] = "No schemes applied."
            else:
                print(f"[EXT-API] Raw schemes count: {len(schemes_list)}. Processing...")

                # Step 1: Deduplication by scheme_id (keeping latest applied_date)
                unique_schemes: dict = {}
                for scheme in schemes_list:
                    s_id = scheme.get("scheme_id")
                    app_date_str = scheme.get("applied_date")
                    if not s_id:
                        continue

                    current_dt = datetime.min
                    if app_date_str:
                        try:
                            curr_str = app_date_str.replace("Z", "+00:00")
                            current_dt = datetime.fromisoformat(curr_str)
                        except Exception as e:
                            print(f"[EXT-API] Date parse error for scheme {s_id}: {e}")

                    if s_id not in unique_schemes:
                        unique_schemes[s_id] = {"data": scheme, "date": current_dt}
                    elif current_dt > unique_schemes[s_id]["date"]:
                        unique_schemes[s_id] = {"data": scheme, "date": current_dt}

                processed_list = [v["data"] for v in unique_schemes.values()]
                print(f"[EXT-API] Unique schemes count: {len(processed_list)}")

                # Step 2 & 3: Fetch Status and Rejection Reasons
                final_schemes_info = []
                for scheme in processed_list:
                    scheme_id = scheme.get("scheme_id")
                    app_code = scheme.get("scheme_application_code")
                    scheme_name = scheme.get("scheme_name", "Unknown Scheme")

                    print(f"[EXT-API] Fetching status for: {scheme_name} ({app_code})...")

                    status_url = f"{base_url}/public/schemes/status"
                    status_payload = {
                        "schemeId": scheme_id,
                        "schemeApplicationCode": app_code,
                        "mobileNumber": "",
                    }

                    try:
                        status_resp = await client.post(
                            status_url, json=status_payload, timeout=5
                        )
                        status_data_full = status_resp.json()
                        status_success = status_data_full.get("success", False)
                        status_items = status_data_full.get("data", [])

                        scheme_status_text = "Status check failed"
                        reasons_list = []

                        if status_success and status_items:
                            item = status_items[0]
                            application_status = item.get("application_status")
                            full_status_desc = item.get("status")
                            avail_id = item.get("id")

                            scheme_status_text = (
                                f"Application Status: {application_status}. "
                                f"({full_status_desc})"
                            )

                            if application_status == "Rejected" and avail_id:
                                print(f"[EXT-API] Scheme Rejected. Fetching reason for ID {avail_id}...")
                                reason_url = (
                                    f"{base_url}/public/schemes/rejection-reason"
                                    f"?availId={avail_id}&reasonType=FINAL"
                                )
                                try:
                                    r_resp = await client.get(reason_url, timeout=5)
                                    r_data = r_resp.json()
                                    if r_data.get("success"):
                                        for r in r_data.get("data", []):
                                            if r.get("rejection_reason"):
                                                reasons_list.append(r["rejection_reason"])
                                except Exception as re:
                                    print(f"[EXT-API] Error fetching rejection reason: {re}")

                        info_block = {
                            "Scheme Name": scheme_name,
                            "Applied Date": scheme.get("applied_date", "").split("T")[0],
                            "Status Details": scheme_status_text,
                        }
                        if reasons_list:
                            info_block["Rejection Reasons"] = "; ".join(reasons_list)
                        final_schemes_info.append(info_block)

                    except Exception as se:
                        print(f"[EXT-API] Error fetching status for {scheme_name}: {se}")
                        final_schemes_info.append({
                            "Scheme Name": scheme_name,
                            "Status": "Could not fetch real-time status.",
                        })

                aggregated_data["schemes"] = {"data": final_schemes_info}
                print("[EXT-API] Schemes processed successfully.")
        else:
            aggregated_data["schemes_error"] = resp.text
            print("[EXT-API] ERROR: Failed to fetch schemes data.")

    except Exception as e:
        aggregated_data["schemes_error"] = str(e)
        print(f"[EXT-API] EXCEPTION while fetching schemes: {e}")

    # ------------------------------------------------------------------
    # 2. Get Renewal Date
    # ------------------------------------------------------------------
    try:
        url = f"{base_url}/user/get-renewal-date"
        payload = {"user_id": str(user_id)}

        print(f"[EXT-API] RENEWAL DATE API CALL: {url}")
        resp = await client.post(url, headers=headers, json=payload, timeout=10)
        print(f"[EXT-API] Renewal HTTP Status: {resp.status_code}")

        if resp.status_code == 200:
            aggregated_data["renewal_date"] = resp.json()
            print("[EXT-API] Renewal date fetched.")
        else:
            aggregated_data["renewal_date_error"] = resp.text
            print("[EXT-API] ERROR: Failed to fetch renewal date.")

    except Exception as e:
        aggregated_data["renewal_date_error"] = str(e)
        print(f"[EXT-API] EXCEPTION while fetching renewal date: {e}")

    # ------------------------------------------------------------------
    # 3. Get User Registration Details (Enhanced Processing)
    # ------------------------------------------------------------------
    try:
        url = f"{base_url}/user/get-user-registration-details"
        payload = {
            "key": "user_id",
            "value": str(user_id),
            "board_id": 1,
            "procedure_name": "all",
        }

        print(f"[EXT-API] REGISTRATION DETAILS API CALL: {url}")
        resp = await client.post(url, headers=headers, json=payload, timeout=10)
        print(f"[EXT-API] Registration HTTP Status: {resp.status_code}")

        if resp.status_code == 200:
            reg_resp_data = resp.json()

            reg_code = None
            full_registration_data = {}

            if reg_resp_data.get("success"):
                data_block = reg_resp_data.get("data", {})
                if data_block:
                    personal_list = data_block.get("personal_details", [])
                    if personal_list and len(personal_list) > 0:
                        personal = personal_list[0]
                        reg_code = personal.get("registration_code")

                        # --- Enhanced Data Extraction ---
                        try:
                            # 1. Personal Details
                            dob_str = personal.get("date_of_birth")
                            dob = None
                            age = None
                            if dob_str:
                                try:
                                    dob_dt = datetime.fromisoformat(
                                        dob_str.replace("Z", "+00:00")
                                    )
                                    dob = dob_dt.strftime("%Y-%m-%d")
                                    today = datetime.now(dob_dt.tzinfo)
                                    age = (today - dob_dt).days // 365
                                except Exception as e:
                                    print(f"[EXT-API] Error parsing DOB: {e}")

                            validity_to_str = personal.get("validity_to_date")
                            validity_from_str = personal.get("validity_from_date")
                            validity_status = "Unknown"

                            is_active = False
                            is_buffer = False
                            current_dt = datetime.now()

                            if validity_to_str:
                                try:
                                    val_to_dt = datetime.fromisoformat(
                                        validity_to_str.replace("Z", "+00:00")
                                    )
                                    current_dt = datetime.now(val_to_dt.tzinfo)

                                    one_year_later = val_to_dt + timedelta(days=365)
                                    inactive_end = one_year_later + timedelta(days=90)

                                    if current_dt <= val_to_dt:
                                        validity_status = "Active"
                                        is_active = True
                                    elif current_dt <= one_year_later:
                                        validity_status = "Active (Buffer Period)"
                                        is_buffer = True
                                    elif current_dt <= inactive_end:
                                        validity_status = "Inactive (Waiting Period)"
                                    else:
                                        validity_status = "Expired (Re-registration Required)"
                                except Exception as e:
                                    print(f"[EXT-API] Error calculating validity: {e}")

                            # Scheme Eligibility Logic
                            eligible_schemes = []
                            try:
                                existing_schemes = aggregated_data.get("schemes", {})
                                if isinstance(existing_schemes, dict):
                                    existing_schemes = existing_schemes.get("data", [])
                                else:
                                    existing_schemes = []

                                pension_approved = False
                                disability_approved = False
                                disability_applied_date = None

                                if isinstance(existing_schemes, list):
                                    for s in existing_schemes:
                                        s_name = s.get("Scheme Name", "").lower()
                                        s_status = s.get("Status Details", "").lower()
                                        s_date = s.get("Applied Date", "")

                                        if "approved" in s_status:
                                            if "pension" in s_name:
                                                pension_approved = True
                                            if "disability" in s_name:
                                                disability_approved = True
                                                if s_date:
                                                    try:
                                                        disability_applied_date = datetime.strptime(
                                                            s_date, "%Y-%m-%d"
                                                        )
                                                        if current_dt.tzinfo:
                                                            disability_applied_date = (
                                                                disability_applied_date.replace(
                                                                    tzinfo=current_dt.tzinfo
                                                                )
                                                            )
                                                    except Exception:
                                                        pass

                                # Rules (exact same as working user_service.py)
                                if is_active or is_buffer:
                                    eligible_schemes.append("Accident Compensation")
                                    eligible_schemes.append("Funeral Assistance")

                                if is_active:
                                    eligible_schemes.append("Medical Assistance")
                                    eligible_schemes.append("Major Ailments Assistance")

                                val_from_year = 9999
                                if validity_from_str:
                                    try:
                                        val_from_dt = datetime.fromisoformat(
                                            validity_from_str.replace("Z", "+00:00")
                                        )
                                        val_from_year = val_from_dt.year
                                    except Exception:
                                        pass

                                current_year = current_dt.year
                                gender = personal.get("gender", "").lower()

                                if current_year > val_from_year:
                                    eligible_schemes.append("Marriage Assistance")

                                if current_year > val_from_year and gender == "female":
                                    eligible_schemes.append("Maternity Assistance (Delivery)")
                                    eligible_schemes.append("Thayi Magu Assistance")

                                if age is not None and age >= 60:
                                    eligible_schemes.append("Pension Scheme")

                                if age is not None and age > 60 and pension_approved:
                                    eligible_schemes.append("Continuation of Pension")

                                eligible_schemes.append("Disability Pension")

                                if disability_approved and disability_applied_date:
                                    one_year_after = disability_applied_date + timedelta(days=365)
                                    if current_dt > one_year_after:
                                        eligible_schemes.append(
                                            "Continuation of Disability Pension"
                                        )

                            except Exception as e:
                                print(f"[EXT-API] Error calculating eligibility: {e}")
                                eligible_schemes.append("Error calculating schemes")

                            extracted_personal = {
                                "first_name": personal.get("first_name"),
                                "last_name": personal.get("last_name"),
                                "registration_code": reg_code,
                                "mobile_no": personal.get("mobile_no"),
                                "marital_status": personal.get("marital_status"),
                                "date_of_birth": dob,
                                "age": age,
                                "nature_of_work": personal.get(
                                    "nature_of_work", "Labour Work"
                                ),
                                "gender": personal.get("gender"),
                                "is_approved": personal.get("is_approved"),
                                "approved_date": personal.get("approved_date"),
                                "validity_from_date": personal.get("validity_from_date"),
                                "validity_to_date": personal.get("validity_to_date"),
                                "calculated_status": validity_status,
                                "eligible_schemes": eligible_schemes,
                            }

                            print(f"[EXT-API] Extracted personal: name={extracted_personal['first_name']}, "
                                  f"age={age}, gender={personal.get('gender')}, "
                                  f"status={validity_status}")
                            print(f"[EXT-API] Eligible schemes: {eligible_schemes}")

                            # 2. Address Details
                            address_list = data_block.get("address_details", [])
                            district = None
                            if address_list:
                                district = address_list[0].get("district")
                            extracted_address = {"district": district}

                            # 3. Family Details (Dependents & Nominees)
                            family_list = data_block.get("family_details", [])
                            dependents = []
                            nominees = []

                            for member in family_list:
                                member_info = {
                                    "relation": member.get("parent_child_relation"),
                                    "first_name": member.get("first_name"),
                                    "last_name": member.get("last_name"),
                                }
                                dependents.append(member_info)
                                if member.get("is_nominee"):
                                    nominees.append(member_info)

                            full_registration_data = {
                                "personal_details": extracted_personal,
                                "address_details": extracted_address,
                                "family_details": dependents,
                                "nominees": nominees,
                            }

                            print(f"[EXT-API] Family: {len(dependents)} dependents, "
                                  f"{len(nominees)} nominees")

                        except Exception as e:
                            print(f"[EXT-API] Error extracting details: {e}")
                            full_registration_data = {}

            # Check registration/renewal status
            final_reg_status = "Registration details not found."

            if reg_code:
                print(f"[EXT-API] Found reg code: {reg_code}. Checking status...")

                status_url = f"{base_url}/public/labour/status"
                reg_status_payload = {
                    "type": "register",
                    "applicationNumber": reg_code,
                    "mobileNumber": "",
                }

                try:
                    r_stat_resp = await client.post(
                        status_url, json=reg_status_payload, timeout=5
                    )
                    r_stat_data = r_stat_resp.json()

                    if r_stat_data.get("success") and r_stat_data.get("data"):
                        r_data = r_stat_data["data"]
                        status_str = r_data.get("status")
                        labour_user_id_stat = r_data.get("labour_user_id")
                        cert_id = r_data.get("labour_work_certificate_id")

                        final_reg_status = f"Registration Status: {status_str}."

                        if status_str == "Approved":
                            print("[EXT-API] Registration Approved. Checking Renewal...")
                            ren_payload = {
                                "type": "renewal",
                                "applicationNumber": reg_code,
                                "mobileNumber": "",
                            }
                            ren_resp = await client.post(
                                status_url, json=ren_payload, timeout=5
                            )
                            ren_data_full = ren_resp.json()

                            if ren_data_full.get("success") and ren_data_full.get("data"):
                                ren_data = ren_data_full["data"]
                                ren_status_str = ren_data.get("status")
                                ren_cert_id = ren_data.get("labour_work_certificate_id")

                                final_reg_status += f" Renewal Status: {ren_status_str}."

                                if ren_status_str == "Rejected":
                                    print("[EXT-API] Renewal Rejected. Fetching reason...")
                                    reasons = await _fetch_rejection_reasons(
                                        client, base_url, headers,
                                        labour_user_id_stat, ren_cert_id,
                                    )
                                    if reasons:
                                        final_reg_status += f" (Reason: {'; '.join(reasons)})"

                        elif status_str == "Rejected":
                            print("[EXT-API] Registration Rejected. Fetching reason...")
                            reasons = await _fetch_rejection_reasons(
                                client, base_url, headers,
                                labour_user_id_stat, cert_id,
                            )
                            if reasons:
                                final_reg_status += f" (Reason: {'; '.join(reasons)})"

                except Exception as ex:
                    print(f"[EXT-API] Error checking reg status: {ex}")
                    final_reg_status += " (verification failed)"
            else:
                print("[EXT-API] Registration Code not found.")

            # Merge summary + personal/family data (critical!)
            aggregated_data["registration_details"] = {
                "summary": final_reg_status,
                **full_registration_data,
            }
            print(f"[EXT-API] Registration details stored. Keys: "
                  f"{list(aggregated_data['registration_details'].keys())}")

        else:
            aggregated_data["registration_details_error"] = resp.text
            print("[EXT-API] ERROR: Failed to fetch registration details.")

    except Exception as e:
        aggregated_data["registration_details_error"] = str(e)
        print(f"[EXT-API] EXCEPTION while fetching registration details: {e}")

    aggregated_data["fetch_status"] = "completed"
    print(f"[EXT-API] DATA AGGREGATION COMPLETED for user {user_id}")

    return aggregated_data


async def _fetch_rejection_reasons(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    labour_user_id,
    certificate_id,
) -> list[str]:
    """Fetch rejection reasons for registration/renewal."""
    reasons = []
    try:
        rej_url = f"{base_url}/public/v2/registration-renewal/rejection-reason"
        rej_payload = {
            "labourUserId": labour_user_id,
            "certificateId": certificate_id,
            "reasonType": "FINAL",
        }
        rej_resp = await client.post(rej_url, json=rej_payload, timeout=5)
        rej_data_full = rej_resp.json()

        if rej_data_full.get("success"):
            for r in rej_data_full.get("data", []):
                if r.get("rejection_reason"):
                    reasons.append(r["rejection_reason"])
    except Exception as e:
        print(f"[EXT-API] Error fetching rejection reason: {e}")

    return reasons
