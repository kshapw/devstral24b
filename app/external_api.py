import asyncio
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_parsed = urlparse(settings.BACKEND_API_URL)
_BASE_ORIGIN = f"{_parsed.scheme}://{_parsed.netloc}"


def _build_headers(auth_token: str) -> dict[str, str]:
    """Build request headers for the Karnataka backend API."""
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


async def _post(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: float | None = None,
) -> dict | None:
    """POST helper with error handling. Returns parsed JSON or None on failure."""
    timeout = timeout or settings.EXTERNAL_API_TIMEOUT
    try:
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            logger.warning(
                "External API returned %d for %s", response.status_code, url,
            )
            return None
        return response.json()
    except httpx.TimeoutException:
        logger.error("External API request timed out: %s", url)
        return None
    except httpx.HTTPError as e:
        logger.error("External API HTTP error for %s: %s", url, e)
        return None
    except Exception as exc:
        logger.error("Unexpected error calling %s: %s", url, type(exc).__name__)
        return None


async def _get(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict | None:
    """GET helper with error handling."""
    timeout = timeout or settings.EXTERNAL_API_TIMEOUT
    try:
        response = await client.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logger.warning(
                "External API GET returned %d for %s", response.status_code, url,
            )
            return None
        return response.json()
    except httpx.TimeoutException:
        logger.error("External API GET timed out: %s", url)
        return None
    except httpx.HTTPError as e:
        logger.error("External API GET HTTP error for %s: %s", url, e)
        return None
    except Exception as exc:
        logger.error("Unexpected error calling GET %s: %s", url, type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# 1. Schemes
# ---------------------------------------------------------------------------
async def _fetch_schemes(
    client: httpx.AsyncClient, headers: dict[str, str], user_id: str
) -> dict | str | None:
    """Fetch schemes, deduplicate, fetch status + rejection reasons."""
    base = settings.BACKEND_API_URL
    url = f"{base}/schemes/get_schemes_by_labor"
    payload = {
        "board_id": 1,
        "labour_user_id": int(user_id) if user_id.isdigit() else user_id,
    }
    logger.info("Fetching schemes for user %s", user_id)

    raw_data = await _post(client, url, headers, payload)
    if raw_data is None:
        return None

    schemes_list = raw_data.get("data", []) if isinstance(raw_data, dict) else []
    if not schemes_list:
        logger.info("No schemes found for user %s", user_id)
        return "No schemes applied."

    # Deduplicate by scheme_id (keep latest applied_date)
    unique_schemes: dict[int | str, dict] = {}
    for scheme in schemes_list:
        s_id = scheme.get("scheme_id")
        if not s_id:
            continue
        app_date_str = scheme.get("applied_date", "")
        current_dt = datetime.min
        if app_date_str:
            try:
                current_dt = datetime.fromisoformat(app_date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError) as e:
                logger.debug("Date parse error for scheme %s: %s", s_id, e)
        if s_id not in unique_schemes or current_dt > unique_schemes[s_id]["date"]:
            unique_schemes[s_id] = {"data": scheme, "date": current_dt}

    processed = [v["data"] for v in unique_schemes.values()]
    logger.info("Unique schemes: %d (from %d raw)", len(processed), len(schemes_list))

    # Fetch status and rejection reasons for each scheme (in parallel)
    async def _fetch_scheme_detail(scheme: dict) -> dict:
        scheme_id = scheme.get("scheme_id")
        app_code = scheme.get("scheme_application_code")
        scheme_name = scheme.get("scheme_name", "Unknown Scheme")

        info_block: dict[str, str] = {
            "Scheme Name": scheme_name,
            "Applied Date": scheme.get("applied_date", "").split("T")[0],
        }

        status_data = await _post(
            client,
            f"{base}/public/schemes/status",
            headers,
            {
                "schemeId": scheme_id,
                "schemeApplicationCode": app_code,
                "mobileNumber": "",
            },
            timeout=settings.EXTERNAL_API_TIMEOUT,
        )

        if status_data and status_data.get("success") and status_data.get("data"):
            item = status_data["data"][0]
            app_status = item.get("application_status", "Unknown")
            status_desc = item.get("status", "")
            avail_id = item.get("id")

            info_block["Status Details"] = (
                f"Application Status: {app_status}. ({status_desc})"
            )

            # Fetch rejection reason if rejected
            if app_status == "Rejected" and avail_id:
                rej_url = (
                    f"{base}/public/schemes/rejection-reason"
                    f"?availId={avail_id}&reasonType=FINAL"
                )
                rej_data = await _get(client, rej_url)
                if rej_data and rej_data.get("success"):
                    reasons = [
                        r["rejection_reason"]
                        for r in rej_data.get("data", [])
                        if r.get("rejection_reason")
                    ]
                    if reasons:
                        info_block["Rejection Reasons"] = "; ".join(reasons)
        else:
            info_block["Status Details"] = "Could not fetch real-time status."

        return info_block

    final_info = list(await asyncio.gather(
        *(_fetch_scheme_detail(s) for s in processed)
    ))

    return {"data": final_info}


# ---------------------------------------------------------------------------
# 2. Renewal date
# ---------------------------------------------------------------------------
async def _fetch_renewal_date(
    client: httpx.AsyncClient, headers: dict[str, str], user_id: str
) -> dict | None:
    """Fetch card/registration renewal date."""
    url = f"{settings.BACKEND_API_URL}/user/get-renewal-date"
    logger.info("Fetching renewal date for user %s", user_id)
    return await _post(client, url, headers, {"user_id": str(user_id)})


# ---------------------------------------------------------------------------
# 3. Registration details (enhanced processing)
# ---------------------------------------------------------------------------
async def _fetch_registration_details(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    user_id: str,
    schemes_data: dict | str | None = None,
) -> dict | None:
    """Fetch registration details with enhanced processing.

    Extracts personal details, calculates validity status, determines scheme
    eligibility, and gathers family/address/nominee information.

    Args:
        schemes_data: Previously fetched schemes result, used for eligibility
                      cross-referencing.
    """
    base = settings.BACKEND_API_URL
    url = f"{base}/user/get-user-registration-details"
    payload = {
        "key": "user_id",
        "value": str(user_id),
        "board_id": 1,
        "procedure_name": "all",
    }
    logger.info("Fetching registration details for user %s", user_id)

    reg_resp = await _post(client, url, headers, payload)
    if reg_resp is None:
        return None

    # Extract registration code and personal details
    reg_code = None
    full_registration_data: dict = {}

    if reg_resp.get("success"):
        data_block = reg_resp.get("data", {})
        if data_block:
            personal_list = data_block.get("personal_details", [])
            if personal_list and len(personal_list) > 0:
                personal = personal_list[0]
                reg_code = personal.get("registration_code")

                try:
                    # ---- Personal Details Extraction ----
                    dob_str = personal.get("date_of_birth")
                    dob = None
                    age = None
                    if dob_str:
                        try:
                            dob_dt = datetime.fromisoformat(dob_str.replace("Z", "+00:00"))
                            dob = dob_dt.strftime("%Y-%m-%d")
                            today = datetime.now(dob_dt.tzinfo)
                            age = (today - dob_dt).days // 365
                        except Exception as e:
                            logger.warning("Error parsing DOB: %s", e)

                    # ---- Validity Status Calculation ----
                    validity_to_str = personal.get("validity_to_date")
                    validity_from_str = personal.get("validity_from_date")
                    validity_status = "Unknown"
                    is_active = False
                    is_buffer = False
                    current_dt = datetime.now()

                    if validity_to_str:
                        try:
                            val_to_dt = datetime.fromisoformat(validity_to_str.replace("Z", "+00:00"))
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
                            logger.warning("Error calculating validity status: %s", e)

                    # ---- Scheme Eligibility Logic ----
                    eligible_schemes = _calculate_eligibility(
                        schemes_data, age, is_active, is_buffer,
                        validity_from_str, personal.get("gender", ""),
                        current_dt,
                    )

                    extracted_personal = {
                        "first_name": personal.get("first_name"),
                        "registration_code": reg_code,
                        "mobile_no": personal.get("mobile_no"),
                        "marital_status": personal.get("marital_status"),
                        "date_of_birth": dob,
                        "age": age,
                        "nature_of_work": personal.get("nature_of_work", "Labour Work"),
                        "gender": personal.get("gender"),
                        "is_approved": personal.get("is_approved"),
                        "approved_date": personal.get("approved_date"),
                        "validity_from_date": personal.get("validity_from_date"),
                        "validity_to_date": personal.get("validity_to_date"),
                        "calculated_status": validity_status,
                        "eligible_schemes": eligible_schemes,
                    }

                    # ---- Address Details ----
                    address_list = data_block.get("address_details", [])
                    district = None
                    if address_list:
                        district = address_list[0].get("district")
                    extracted_address = {"district": district}

                    # ---- Family Details (Dependents & Nominees) ----
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

                except Exception as e:
                    logger.error("Error extracting registration details: %s", e, exc_info=True)

    # ---- Registration / Renewal Status ----
    final_reg_status = "Registration details not found."

    if reg_code:
        logger.info("Found registration code %s, checking status", reg_code)
        status_url = f"{base}/public/labour/status"

        reg_status_data = await _post(
            client, status_url, headers,
            {"type": "register", "applicationNumber": reg_code, "mobileNumber": ""},
        )

        if reg_status_data and reg_status_data.get("success") and reg_status_data.get("data"):
            r_data = reg_status_data["data"]
            status_str = r_data.get("status", "Unknown")
            labour_user_id_stat = r_data.get("labour_user_id")
            cert_id = r_data.get("labour_work_certificate_id")
            final_reg_status = f"Registration Status: {status_str}."

            if status_str == "Approved":
                # Check renewal status
                logger.info("Registration Approved. Checking Renewal Status...")
                ren_data = await _post(
                    client, status_url, headers,
                    {"type": "renewal", "applicationNumber": reg_code, "mobileNumber": ""},
                )
                if ren_data and ren_data.get("success") and ren_data.get("data"):
                    ren_info = ren_data["data"]
                    ren_status = ren_info.get("status", "Unknown")
                    final_reg_status += f" Renewal Status: {ren_status}."

                    if ren_status == "Rejected":
                        logger.info("Renewal Rejected. Fetching reason...")
                        reasons = await _fetch_rejection_reasons(
                            client, headers, labour_user_id_stat,
                            ren_info.get("labour_work_certificate_id"),
                        )
                        if reasons:
                            final_reg_status += f" (Reason: {reasons})"

            elif status_str == "Rejected":
                logger.info("Registration Rejected. Fetching reason...")
                reasons = await _fetch_rejection_reasons(
                    client, headers, labour_user_id_stat, cert_id,
                )
                if reasons:
                    final_reg_status += f" (Reason: {reasons})"
        else:
            final_reg_status += " (status verification failed)"
    else:
        logger.info("Registration Code not found for user %s", user_id)

    return {
        "summary": final_reg_status,
        **full_registration_data,
    }


def _calculate_eligibility(
    schemes_data: dict | str | None,
    age: int | None,
    is_active: bool,
    is_buffer: bool,
    validity_from_str: str | None,
    gender: str,
    current_dt: datetime,
) -> list[str]:
    """Calculate which schemes the user is eligible for based on their profile."""
    eligible_schemes: list[str] = []

    try:
        # Extract existing scheme statuses
        existing_schemes: list[dict] = []
        if isinstance(schemes_data, dict):
            existing_schemes = schemes_data.get("data", [])
            if not isinstance(existing_schemes, list):
                existing_schemes = []

        pension_approved = False
        disability_approved = False
        disability_applied_date = None

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
                            disability_applied_date = datetime.strptime(s_date, "%Y-%m-%d")
                            if current_dt.tzinfo:
                                disability_applied_date = disability_applied_date.replace(
                                    tzinfo=current_dt.tzinfo
                                )
                        except Exception:
                            pass

        # Rule 1: Accident, Funeral — Active OR Buffer
        if is_active or is_buffer:
            eligible_schemes.append("Accident Compensation")
            eligible_schemes.append("Funeral Assistance")

        # Rule 2: Medical, Major Ailments — Active only
        if is_active:
            eligible_schemes.append("Medical Assistance")
            eligible_schemes.append("Major Ailments Assistance")

        # Year check for Marriage/Delivery
        val_from_year = 9999
        if validity_from_str:
            try:
                val_from_dt = datetime.fromisoformat(validity_from_str.replace("Z", "+00:00"))
                val_from_year = val_from_dt.year
            except Exception:
                pass

        current_year = current_dt.year
        gender_lower = gender.lower()

        # Rule 3: Marriage — Current Year > Validity From Year
        if current_year > val_from_year:
            eligible_schemes.append("Marriage Assistance")

        # Rule 4: Delivery, Thayi Magu — Current Year > Validity From Year AND Female
        if current_year > val_from_year and gender_lower == "female":
            eligible_schemes.append("Maternity Assistance (Delivery)")
            eligible_schemes.append("Thayi Magu Assistance")

        # Rule 5: Pension — Age >= 60
        if age is not None and age >= 60:
            eligible_schemes.append("Pension Scheme")

        # Rule 6: Continuation of Pension — Age > 60 AND Pension Approved
        if age is not None and age > 60 and pension_approved:
            eligible_schemes.append("Continuation of Pension")

        # Rule 7: Disability Pension — Eligible all time (if registered)
        eligible_schemes.append("Disability Pension")

        # Rule 8: Continuation of Disability — Disability Approved AND > 1 year since applied
        if disability_approved and disability_applied_date:
            one_year_after_apply = disability_applied_date + timedelta(days=365)
            if current_dt > one_year_after_apply:
                eligible_schemes.append("Continuation of Disability Pension")

    except Exception as e:
        logger.error("Error calculating scheme eligibility: %s", e)
        eligible_schemes.append("Error calculating schemes")

    return eligible_schemes


async def _fetch_rejection_reasons(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    labour_user_id: str | int | None,
    certificate_id: str | int | None,
) -> str:
    """Fetch rejection reasons for registration/renewal."""
    if not labour_user_id or not certificate_id:
        return ""
    url = f"{settings.BACKEND_API_URL}/public/v2/registration-renewal/rejection-reason"
    payload = {
        "labourUserId": labour_user_id,
        "certificateId": certificate_id,
        "reasonType": "FINAL",
    }
    data = await _post(client, url, headers, payload)
    if data and data.get("success"):
        reasons = [
            r["rejection_reason"]
            for r in data.get("data", [])
            if r.get("rejection_reason")
        ]
        return "; ".join(reasons)
    return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def fetch_user_data(
    client: httpx.AsyncClient, user_id: str, auth_token: str
) -> dict:
    """Fetch and aggregate all user data from the Karnataka backend API.

    Schemes are fetched first (along with renewal date in parallel), then
    registration details are fetched with the schemes data for eligibility
    cross-referencing.

    Returns a dict with keys: user_id, schemes, renewal_date,
    registration_details, fetch_status.
    """
    logger.info("Starting user data aggregation for user_id=%s", user_id)
    headers = _build_headers(auth_token)

    aggregated: dict = {
        "user_id": user_id,
        "schemes": None,
        "renewal_date": None,
        "registration_details": None,
        "fetch_status": "partial",
    }

    # Phase 1: Fetch schemes and renewal date in parallel
    async def _safe_schemes():
        try:
            return await _fetch_schemes(client, headers, user_id)
        except Exception as exc:
            logger.error("Failed to fetch schemes for user %s: %s", user_id, type(exc).__name__)
            return None

    async def _safe_renewal():
        try:
            return await _fetch_renewal_date(client, headers, user_id)
        except Exception as exc:
            logger.error("Failed to fetch renewal_date for user %s: %s", user_id, type(exc).__name__)
            return None

    schemes, renewal = await asyncio.gather(_safe_schemes(), _safe_renewal())
    aggregated["schemes"] = schemes
    aggregated["renewal_date"] = renewal

    # Phase 2: Fetch registration details (needs schemes data for eligibility)
    try:
        registration = await _fetch_registration_details(
            client, headers, user_id, schemes_data=schemes,
        )
    except Exception as exc:
        logger.error(
            "Failed to fetch registration_details for user %s: %s",
            user_id, type(exc).__name__,
        )
        registration = None

    aggregated["registration_details"] = registration
    aggregated["fetch_status"] = "completed"
    logger.info("User data aggregation completed for user_id=%s", user_id)
    return aggregated
