import asyncio
import logging
from datetime import datetime
from urllib.parse import urlparse

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_parsed = urlparse(settings.BACKEND_API_URL)
_BASE_ORIGIN = f"{_parsed.scheme}://{_parsed.netloc}"


def _build_headers(auth_token: str) -> dict[str, str]:
    """Build request headers for the Karnataka backend API."""
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Origin": _BASE_ORIGIN,
        "Referer": f"{_BASE_ORIGIN}/u/home",
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
# 3. Registration details
# ---------------------------------------------------------------------------
async def _fetch_registration_details(
    client: httpx.AsyncClient, headers: dict[str, str], user_id: str
) -> dict | None:
    """Fetch registration details and check registration/renewal status."""
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

    # Extract registration code
    reg_code = None
    if reg_resp.get("success"):
        data_block = reg_resp.get("data", {})
        if data_block:
            personal_details = data_block.get("personal_details", [])
            if personal_details:
                reg_code = personal_details[0].get("registration_code")

    if not reg_code:
        logger.info("No registration code found for user %s", user_id)
        return {"summary": "Registration details not found."}

    logger.info("Found registration code %s, checking status", reg_code)
    status_url = f"{base}/public/labour/status"
    final_status = f"Registration Code: {reg_code}."

    # Check registration status
    reg_status_data = await _post(
        client, status_url, headers,
        {"type": "register", "applicationNumber": reg_code, "mobileNumber": ""},
    )

    if not (reg_status_data and reg_status_data.get("success") and reg_status_data.get("data")):
        return {"summary": final_status + " (status verification failed)"}

    r_data = reg_status_data["data"]
    status_str = r_data.get("status", "Unknown")
    labour_user_id_stat = r_data.get("labour_user_id")
    cert_id = r_data.get("labour_work_certificate_id")
    final_status += f" Registration Status: {status_str}."

    # If approved, check renewal status
    if status_str == "Approved":
        ren_data = await _post(
            client, status_url, headers,
            {"type": "renewal", "applicationNumber": reg_code, "mobileNumber": ""},
        )
        if ren_data and ren_data.get("success") and ren_data.get("data"):
            ren_info = ren_data["data"]
            ren_status = ren_info.get("status", "Unknown")
            final_status += f" Renewal Status: {ren_status}."

            if ren_status == "Rejected":
                reasons = await _fetch_rejection_reasons(
                    client, headers, labour_user_id_stat,
                    ren_info.get("labour_work_certificate_id"),
                )
                if reasons:
                    final_status += f" (Reason: {reasons})"

    elif status_str == "Rejected":
        reasons = await _fetch_rejection_reasons(
            client, headers, labour_user_id_stat, cert_id,
        )
        if reasons:
            final_status += f" (Reason: {reasons})"

    return {"summary": final_status}


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

    async def _safe_fetch(fn, label: str):
        try:
            return await fn(client, headers, user_id)
        except Exception as exc:
            logger.error("Failed to fetch %s for user %s: %s", label, user_id, type(exc).__name__)
            return None

    schemes, renewal, registration = await asyncio.gather(
        _safe_fetch(_fetch_schemes, "schemes"),
        _safe_fetch(_fetch_renewal_date, "renewal_date"),
        _safe_fetch(_fetch_registration_details, "registration_details"),
    )

    aggregated["schemes"] = schemes
    aggregated["renewal_date"] = renewal
    aggregated["registration_details"] = registration
    aggregated["fetch_status"] = "completed"
    logger.info("User data aggregation completed for user_id=%s", user_id)
    return aggregated
