import json
import logging
import random
import unicodedata
from typing import AsyncIterator, Optional

import aiosqlite
import httpx
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.database import get_cached_user_data, save_user_data
from app.external_api import fetch_user_data
from app.ollama_client import OllamaClient, default_ollama
from app.qdrant_service import get_qdrant_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the FULL knowledge base at startup — this is injected into every
# GENERAL system prompt instead of using RAG retrieval.
# At ~33KB / ~8-10K tokens, it fits easily in the context window and
# guarantees the model ALWAYS has ALL the information.
# ---------------------------------------------------------------------------
_FULL_KNOWLEDGE_BASE: str = ""
try:
    import pathlib
    _kb_path = pathlib.Path(settings.DATA_PATH)
    # If relative path, resolve from project root (parent of app/ directory)
    if not _kb_path.is_absolute():
        _project_root = pathlib.Path(__file__).resolve().parent.parent
        _kb_path = _project_root / _kb_path
    print(f"[STARTUP] Knowledge base path: {_kb_path}")
    print(f"[STARTUP] Path exists: {_kb_path.exists()}")
    if _kb_path.exists():
        _FULL_KNOWLEDGE_BASE = _kb_path.read_text(encoding="utf-8")
        print(f"[STARTUP] Loaded knowledge base: {len(_FULL_KNOWLEDGE_BASE)} chars, ~{len(_FULL_KNOWLEDGE_BASE) // 4} tokens")
        logger.info(
            "Loaded full knowledge base from %s (%d chars, ~%d tokens)",
            _kb_path, len(_FULL_KNOWLEDGE_BASE), len(_FULL_KNOWLEDGE_BASE) // 4,
        )
    else:
        print(f"[STARTUP] ERROR: Knowledge base file NOT FOUND at: {_kb_path}")
        logger.error("Knowledge base file not found: %s", _kb_path)
except Exception as e:
    print(f"[STARTUP] ERROR loading knowledge base: {e}")
    logger.error("Failed to load knowledge base", exc_info=True)

# ---------------------------------------------------------------------------
# Section-based direct responses — bypass LLM entirely for known queries.
# Parses ksk.md into sections keyed by heading, then maps user queries to
# the EXACT content, eliminating hallucination completely.
# ---------------------------------------------------------------------------
import re as _re

def _parse_kb_sections(text: str) -> dict[str, str]:
    """Split the knowledge base into sections by '---' separators and ## headings."""
    sections: dict[str, str] = {}
    if not text:
        return sections

    # Split by horizontal rules (---) to get major blocks
    blocks = _re.split(r'\n---\n', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Find the first ## heading in this block
        heading_match = _re.search(r'^##\s+(?:Scheme:\s*)?(.+)$', block, _re.MULTILINE)
        if heading_match:
            key = heading_match.group(1).strip().lower()
            sections[key] = block
        # Also check for # heading (top-level sections like Registration, Renewal)
        heading_match_h1 = _re.search(r'^#\s+(.+)$', block, _re.MULTILINE)
        if heading_match_h1:
            key_h1 = heading_match_h1.group(1).strip().lower()
            if key_h1 not in sections:  # Don't overwrite ## matches
                sections[key_h1] = block

    return sections


_KB_SECTIONS: dict[str, str] = _parse_kb_sections(_FULL_KNOWLEDGE_BASE)
print(f"[STARTUP] Parsed {len(_KB_SECTIONS)} sections from knowledge base")
print(f"[STARTUP] Section keys: {list(_KB_SECTIONS.keys())}")
logger.info("Parsed %d sections from knowledge base: %s", len(_KB_SECTIONS), list(_KB_SECTIONS.keys()))

# Keyword → section key mapping for direct responses
_DIRECT_RESPONSE_MAP: list[tuple[list[str], str]] = [
    # Registration
    (["registration", "register", "how to register", "worker registration",
      "ನೋಂದಣಿ", "ನೊಂದಣಿ", "ನೋಂದಾಯಿಸಿ"], "registration information"),
    # Renewal
    (["renewal", "renew", "how to renew", "worker renewal",
      "ನವೀಕರಣ"], "renewal information"),
    # Schemes
    (["accident", "accident benefit", "accident assistance",
      "ಅಪಘಾತ"], "accident benefits"),
    (["major ailment", "major ailments", "karmika chikitsa", "chikitsa bhagya",
      "ದೊಡ್ಡ ಕಾಯಿಲೆ"], "assistance for major ailments"),
    (["thayi magu", "nutritional support", "pre-school", "sahaya hasta",
      "ತಾಯಿ ಮಗು"], "thayi magu sahaya hasta (nutritional support)"),
    (["delivery assistance", "tayi lakshmi", "delivery benefit", "maternity",
      "ಹೆರಿಗೆ ಸಹಾಯ", "ಹೆರಿಗೆ"], "delivery assistance"),
    (["medical assistance", "medical help", "hospitalization", "hospital",
      "ವೈದ್ಯಕೀಯ ಸಹಾಯ"], "medical assistance"),
    (["disability pension", "disabled", "disability",
      "ಅಂಗವಿಕಲ ಪಿಂಚಣಿ"], "disability pension & ex-gratia"),
    (["continuation of disability"], "continuation of disability pension"),
    (["pension", "old age pension", "retirement",
      "ಪಿಂಚಣಿ"], "pension (old age pension for construction workers)"),
    (["continuation of pension", "pension continuation"],
     "continuation of pension"),
    (["funeral", "ex-gratia", "death assistance", "death benefit",
      "ಅಂತ್ಯಕ್ರಿಯೆ"], "funeral and ex-gratia"),
    (["marriage", "marriage assistance", "vivaha", "wedding",
      "ಮದುವೆ ಸಹಾಯ", "ಮದುವೆ"], "marriage assistance"),
    # Migration
    (["migration", "seva sindhu", "e-karmika", "transfer",
      "ವರ್ಗಾವಣೆ"], "migration from seva sindhu to ksk (important)"),
]


def _find_direct_section(message: str) -> str | None:
    """Return the exact ksk.md section content if the query matches a known topic.

    Returns None if no match found (falls through to LLM).
    """
    msg = unicodedata.normalize("NFC", message.lower()).strip()

    # Special handler: "What schemes are available?" / "List all schemes"
    _SCHEME_LIST_KEYWORDS = [
        "what schemes", "list schemes", "all schemes", "available schemes",
        "schemes available", "scheme list", "which schemes", "tell me about schemes",
        "ಯೋಜನೆಗಳು", "ಯೋಜನೆ ಪಟ್ಟಿ",
    ]
    if any(kw in msg for kw in _SCHEME_LIST_KEYWORDS) or msg in ["schemes", "scheme"]:
        return (
            "# Available Welfare Schemes under KBOCWWB\n\n"
            "The following welfare schemes are available for registered construction workers:\n\n"
            "1. **Accident Benefits** — Up to ₹8 Lakh for death, ₹2 Lakh for permanent total disablement, ₹1 Lakh for partial disablement\n"
            "2. **Assistance for Major Ailments (Karmika Chikitsa Bhagya)** — Up to ₹2,00,000 for treatment of major ailments\n"
            "3. **Thayi Magu Sahaya Hasta (Nutritional Support)** — ₹6,000 (₹500/month) for pre-school education & nutritional support of child\n"
            "4. **Delivery Assistance (Tayi Lakshmi Bond)** — ₹50,000 per delivery (first two living children only)\n"
            "5. **Medical Assistance** — ₹300 per day of hospitalization, maximum ₹20,000 (minimum 48 hours hospitalization required)\n"
            "6. **Disability Pension & Ex-Gratia** — ₹2,000/month pension + up to ₹2,00,000 ex-gratia based on disability percentage\n"
            "7. **Continuation of Disability Pension** — Continuation of existing disability pension (annual Living Certificate required)\n"
            "8. **Pension (Old Age Pension)** — Up to ₹3,000/month for workers who completed 60 years of age\n"
            "9. **Continuation of Pension** — Annual continuation of pension (Living Certificate required every December)\n"
            "10. **Funeral and Ex-Gratia** — ₹1,46,000 for funeral expenses and ex-gratia to nominee\n"
            "11. **Marriage Assistance** — ₹60,000 for marriage of worker or dependent children (maximum twice per family)\n\n"
            "**Note:** Most schemes require a valid 90 Days Work Certificate and active registration.\n"
            "For detailed information about any specific scheme, please ask about it by name."
        )

    print(f"[DEBUG] _find_direct_section called with msg='{msg}', _KB_SECTIONS has {len(_KB_SECTIONS)} keys")

    for keywords, section_key in _DIRECT_RESPONSE_MAP:
        for kw in keywords:
            if kw in msg or msg in kw:
                print(f"[DEBUG]   Keyword '{kw}' matched! Looking for section_key='{section_key}'")
                # Find the section (try exact key, then partial match)
                if section_key in _KB_SECTIONS:
                    print(f"[DEBUG]   Found exact section key, returning {len(_KB_SECTIONS[section_key])} chars")
                    return _KB_SECTIONS[section_key]
                # Partial match on section keys
                for sk, content in _KB_SECTIONS.items():
                    if section_key in sk or sk in section_key:
                        print(f"[DEBUG]   Found partial section key '{sk}', returning {len(content)} chars")
                        return content
                print(f"[DEBUG]   WARNING: keyword matched but section_key '{section_key}' NOT found in _KB_SECTIONS!")
    print(f"[DEBUG]   No keyword match found, falling through to LLM")
    return None


# ---------------------------------------------------------------------------
# Sub-topic Q&A system — answers specific questions WITHOUT the LLM.
# Two-level matching: Topic (registration, renewal, scheme) × Sub-topic
# (fee, documents, eligibility, procedure, timeline, benefit).
# ---------------------------------------------------------------------------

# Sub-topic keyword groups
_SUBTOPIC_FEE = ["how much", "fee", "cost", "pay", "price", "charge", "amount to pay",
                 "ಶುಲ್ಕ", "ಎಷ್ಟು"]
_SUBTOPIC_DOCS = ["document", "documents", "what documents", "required documents", "papers",
                  "what do i need", "what i need", "ದಾಖಲೆ", "ದಾಖಲೆಗಳು"]
_SUBTOPIC_ELIGIBILITY = ["eligib", "who can", "am i eligible", "criteria", "qualify",
                         "condition", "requirement", "ಅರ್ಹತೆ"]
_SUBTOPIC_PROCEDURE = ["how to apply", "procedure", "process", "steps", "apply",
                       "how do i", "how can i", "ಅರ್ಜಿ", "ಹೇಗೆ"]
_SUBTOPIC_TIMELINE = ["how long", "time", "timeline", "days", "when will",
                      "delivery", "ಸಮಯ", "ಎಷ್ಟು ದಿನ"]
_SUBTOPIC_BENEFIT = ["benefit", "what do i get", "how much will i get", "compensation",
                     "assistance amount", "ಸೌಲಭ್ಯ", "ಪ್ರಯೋಜನ"]

# Per-topic sub-answers (copied from ksk.md, guaranteed accurate)
_TOPIC_SUBANSWERS: dict[str, dict[str, str]] = {
    "registration": {
        "fee": "The **application fee for Worker Registration** is **Rs.100** (Rupees One Hundred only).",
        "documents": (
            "**Required Documents for Worker Registration:**\n\n"
            "1. **Employment Certificate** — In Form V(A) / V(B) / V(C) / V(D), issued by authorized employer / contractor / competent authority\n"
            "2. **Aadhaar Card** — Self-attested copy\n"
            "3. **Ration Card** — Non-Mandatory, for family details verification\n"
            "4. **Age Proof** — Any one of: Aadhaar Card, Voter ID Card, or any Government-issued age proof document"
        ),
        "eligibility": (
            "**Eligibility Criteria for Worker Registration:**\n\n"
            "- Must satisfy the eligibility criteria prescribed under the BOCW Act\n"
            "- Must have worked for a minimum of **90 days** in building or other construction work during the preceding 12 months\n"
            "- Age must be between **18 to 60 years**\n"
            "- Must not be a member of any other Welfare Board"
        ),
        "procedure": (
            "**Procedure for Worker Registration:**\n\n"
            "1. Submit the duly filled application along with required documents\n"
            "2. Application is scrutinized by the Registering Authority (Labour Inspector / Senior Labour Inspector)\n"
            "3. Verification of employment certificate and eligibility\n"
            "4. Approval / Rejection by the competent authority\n"
            "5. Issuance of Beneficiary Registration (Labour Card)"
        ),
        "timeline": "The **delivery timeline for Worker Registration** is **15 Working Days** (subject to document verification and field validation).",
        "benefit": "Registration enables eligible construction workers to **avail all welfare benefits** under the BOCW Act, including accident benefits, medical assistance, pension, delivery assistance, and more.",
    },
    "renewal": {
        "fee": "The renewal process requires a **valid 90 Days Work Certificate**. Please visit your nearest KSK for the current renewal fee details.",
        "documents": (
            "**Required for Worker Renewal:**\n\n"
            "- **90 Days Work Certificate** — Mandatory, issued by a competent authority (Builder / Contractor / Engineer / Local Authority)\n"
            "- Valid proof of having worked at least 90 days in the last 12 months\n"
            "- Aadhaar-based authentication"
        ),
        "eligibility": (
            "**Eligibility for Worker Renewal:**\n\n"
            "- Must have worked at least **90 days** in the last 12 months\n"
            "- Must have a **valid 90 Days Work Certificate** issued by a competent authority\n"
            "- Can only apply **after registration has expired**\n"
            "- Must apply within **365 days** (1 year) from the expiry date (buffer period)\n"
            "- If more than 365 days have passed since expiry, must apply for **New Registration** instead"
        ),
        "procedure": (
            "**When Can Renewal Be Applied:**\n\n"
            "1. Renewal can be applied **only after the registration has expired**\n"
            "2. Worker enters a **buffer period of 365 days** (1 year) from the date of expiry\n"
            "3. During the buffer period, the worker is eligible to apply for Renewal\n"
            "4. Submit the 90 Days Work Certificate and required documents\n\n"
            "**Important:** If the worker does not apply within 365 days, Renewal is not permitted — must apply for New Registration."
        ),
        "timeline": "Renewal is **only available after registration expiry**, within a **buffer period of 365 days** (1 year) from the expiry date.",
        "benefit": "Renewal of registration allows continued access to **all welfare benefits** under the Board. Without renewal, scheme applications cannot be submitted.",
    },
    "accident": {
        "fee": "There is **no application fee** for Accident Benefits. The application must be submitted **within 1 year** from the date of the accident.",
        "documents": (
            "**Required Documents for Accident Benefits:**\n\n"
            "- Photocopy of ID card attested by gazetted officer\n"
            "- Photocopy of bank passbook\n"
            "- Beneficiary / original Identity card\n"
            "- Application in **Form XXI or XXI-B**\n"
            "- Death Certificate (in case of death)\n"
            "- Post mortem report\n"
            "- FIR copy\n"
            "- Medical report\n"
            "- Employer Certificate\n"
            "- **Form XXI-A**"
        ),
        "benefit": (
            "**Accident Benefits:**\n\n"
            "- **Rs.8 Lakh** for death\n"
            "- **Rs.2 Lakh** for permanent total disablement\n"
            "- **Rs.1 Lakh** for permanent partial disablement\n"
            "- **Rs.2 Lakh** compensation for accident during employment resulting in death\n"
            "- Up to **Rs.2 Lakh** for grievous injury during employment"
        ),
        "eligibility": "Must be a **registered construction worker** with a valid registration. Application must be submitted **within 1 year** from the date of the accident.",
        "procedure": "Submit the application in **Form XXI or XXI-B** along with all required documents to the Board office **within 1 year** from the date of accident.",
        "timeline": "Application must be submitted **within 1 year** from the date of the accident.",
    },
    "delivery": {
        "fee": "There is **no application fee** for Delivery Assistance.",
        "benefit": "**Rs.50,000** financial assistance per delivery, applicable only for the **first two living children**.",
        "documents": (
            "**Required Documents for Delivery Assistance:**\n\n"
            "- Proof of Identity / Smart card issued by the Board\n"
            "- Proof of Bank Account\n"
            "- Birth Certificate of Child\n"
            "- Discharge Summary from hospital\n"
            "- Photo of Child\n"
            "- Employment Certificate\n"
            "- Affidavit for second child (if applicable)"
        ),
        "eligibility": (
            "**Eligibility for Delivery Assistance:**\n\n"
            "- Must be a **registered woman construction worker**\n"
            "- Applicable for **first two living children** only\n"
            "- Application within **6 months** of the child's birth\n"
            "- Affidavit required when applying for the second child"
        ),
        "procedure": "Submit the application with all required documents **within 6 months** of the child's birth.",
        "timeline": "Application must be submitted **within 6 months** of the child's birth.",
    },
    "pension": {
        "fee": "There is **no application fee** for Old Age Pension.",
        "benefit": "The pension amount shall **not exceed Rs.3,000 per month**.",
        "eligibility": (
            "**Eligibility for Old Age Pension:**\n\n"
            "- Must have completed **60 years of age**\n"
            "- Must have continued as a **beneficiary of the Board for at least 3 years** before the age of 60\n"
            "- Cannot avail similar benefit under any other Government schemes"
        ),
        "documents": (
            "**Required Documents for Pension:**\n\n"
            "- Original Identity Card issued by the Board\n"
            "- Employer Certificate\n"
            "- Ration Card\n"
            "- Photocopy of bank passbook\n"
            "- Passport size photo\n"
            "- **Living Certificate** must be provided every year"
        ),
        "procedure": "Submit the application with required documents after completing 60 years of age. A **Living Certificate** must be provided every year to continue receiving pension.",
        "timeline": "Pension is provided **monthly** after approval. Annual **Living Certificate** required for continuation.",
    },
    "medical": {
        "fee": "There is **no application fee** for Medical Assistance.",
        "benefit": "**Rs.300 per day** of hospitalization, maximum assistance up to **Rs.20,000** for continuous hospitalization.",
        "eligibility": (
            "**Eligibility for Medical Assistance:**\n\n"
            "- Must be a registered construction worker or their dependent\n"
            "- Hospitalization must be for a **minimum of 48 hours** continuously\n"
            "- Application within **6 months** of hospitalization"
        ),
        "documents": (
            "**Required Documents for Medical Assistance:**\n\n"
            "- Proof of Identity / Smart card issued by the Board\n"
            "- Employment Certificate\n"
            "- Proof of Bank Account\n"
            "- Bills showing admission and discharge dates\n"
            "- **Form XXII-A**"
        ),
        "procedure": "Submit the application with **Form XXII-A** and hospital bills **within 6 months** of hospitalization.",
        "timeline": "Application must be submitted **within 6 months** of hospitalization.",
    },
    "marriage": {
        "fee": "There is **no application fee** for Marriage Assistance.",
        "benefit": "**Rs.60,000** assistance for marriage of worker or dependent children. Available **maximum twice** per family.",
        "eligibility": (
            "**Eligibility for Marriage Assistance:**\n\n"
            "- Minimum **1 year** since registration to marriage date\n"
            "- Available only **twice per family**\n"
            "- Only one claim per marriage\n"
            "- Son/daughter must have attained **legal age** for marriage\n"
            "- Application within **6 months** of marriage"
        ),
        "documents": (
            "**Required Documents for Marriage Assistance:**\n\n"
            "- Identity card issued by the Board\n"
            "- Employment Certificate\n"
            "- Bank Account details\n"
            "- **Marriage Certificate** by Registrar\n"
            "- Marriage Invitation Card\n"
            "- Affidavit (if marriage outside Karnataka)\n"
            "- Ration Card"
        ),
        "procedure": "Submit the application with Marriage Certificate and required documents **within 6 months** of marriage.",
        "timeline": "Application must be submitted **within 6 months** of marriage.",
    },
    "funeral": {
        "fee": "There is **no application fee** for Funeral and Ex-Gratia.",
        "benefit": "**Rs.1,46,000** for funeral expenses and ex-gratia to the nominee.",
        "documents": (
            "**Required Documents for Funeral and Ex-Gratia:**\n\n"
            "- Identity card issued by the Board\n"
            "- Bank passbook of Nominee\n"
            "- Death Certificate attested by gazetted officer\n"
            "- Ration Card\n"
            "- Aadhaar Card\n"
            "- Employer Certificate\n"
            "- Photo ID of Nominee"
        ),
        "eligibility": "The **nominee** of the deceased registered construction worker can apply **within 1 year** of death.",
        "procedure": "The nominee submits the application in **Form XVIII** with Death Certificate and required documents **within 1 year** of death.",
        "timeline": "Application must be submitted **within 1 year** of death.",
    },
    "major ailment": {
        "fee": "There is **no application fee** for Assistance for Major Ailments.",
        "benefit": "Up to **Rs.2,00,000** for treatment of major ailments at **CGHS rates**.",
        "eligibility": (
            "**Eligibility for Major Ailments Assistance:**\n\n"
            "- Must be a registered construction worker or dependent\n"
            "- Treatment must be for a recognized major ailment\n"
            "- Application within **6 months** of hospitalization"
        ),
        "documents": (
            "**Required Documents for Major Ailments:**\n\n"
            "- Proof of Identity / Smart card\n"
            "- Employment Certificate\n"
            "- Proof of Bank Account\n"
            "- Hospital bills showing admission/discharge dates\n"
            "- **Form XXII-A**"
        ),
        "procedure": "Submit application with Form XXII-A and hospital bills **within 6 months** of hospitalization.",
        "timeline": "Application must be submitted **within 6 months** of hospitalization.",
    },
    "thayi magu": {
        "fee": "There is **no application fee** for Thayi Magu Sahaya Hasta.",
        "benefit": "**Rs.6,000** (at the rate of Rs.500 per month) for pre-school education and nutritional support of the child.",
        "eligibility": (
            "**Eligibility for Thayi Magu Sahaya Hasta:**\n\n"
            "- Registered **woman construction worker** who has delivered a child\n"
            "- Available for **3 years** from the date of delivery\n"
            "- Can be availed **twice** (first two children only)"
        ),
        "documents": (
            "**Required Documents for Thayi Magu Sahaya Hasta:**\n\n"
            "- Proof of Identity / Smart card\n"
            "- Affidavit for second delivery\n"
            "- Proof of Bank Account\n"
            "- Photo of Child\n"
            "- Employment Certificate\n"
            "- Discharge Summary\n"
            "- Birth Certificate of Child\n"
            "- Child living Affidavit for 2nd and 3rd year"
        ),
        "procedure": "Submit application with Birth Certificate and required documents. Annual **Child living Affidavit** required for 2nd and 3rd year.",
        "timeline": "Available for **3 years** from the date of delivery. Annual renewal required.",
    },
    "disability": {
        "fee": "There is **no application fee** for Disability Pension & Ex-Gratia.",
        "benefit": (
            "**Disability Pension & Ex-Gratia:**\n\n"
            "- **Rs.2,000 per month** disability pension\n"
            "- Up to **Rs.2,00,000** ex-gratia based on disability percentage\n"
            "- Formula: Rs.2,00,000 × Percentage of disability = Ex-gratia amount"
        ),
        "eligibility": (
            "**Eligibility for Disability Pension:**\n\n"
            "- Beneficiary partially disabled due to disease or accident\n"
            "- Cannot avail if already received accident assistance\n"
            "- Must obtain ID card from Department for empowerment of differently abled\n"
            "- Pension **discontinued at age 60**"
        ),
        "documents": (
            "**Required Documents for Disability Pension:**\n\n"
            "- Identity card issued by the Board\n"
            "- Bank passbook\n"
            "- **Living Certificate** every year\n"
            "- Ration Card\n"
            "- Employer Certificate\n"
            "- Medical Report\n"
            "- **Disability ID card**"
        ),
        "procedure": "Submit application with Disability ID card and required documents **within 6 months** from disability ID card issue. Annual **Living Certificate** required.",
        "timeline": "Application within **6 months** from disability ID card issue. Pension continues until age 60.",
    },
}


def _detect_subtopic(msg: str) -> str | None:
    """Detect the sub-topic (fee, documents, eligibility, procedure, timeline, benefit) from message."""
    if any(kw in msg for kw in _SUBTOPIC_FEE):
        return "fee"
    if any(kw in msg for kw in _SUBTOPIC_DOCS):
        return "documents"
    if any(kw in msg for kw in _SUBTOPIC_ELIGIBILITY):
        return "eligibility"
    if any(kw in msg for kw in _SUBTOPIC_PROCEDURE):
        return "procedure"
    if any(kw in msg for kw in _SUBTOPIC_TIMELINE):
        return "timeline"
    if any(kw in msg for kw in _SUBTOPIC_BENEFIT):
        return "benefit"
    return None


# Topic keyword → topic key mapping
_TOPIC_KEYWORDS: list[tuple[list[str], str]] = [
    (["registration", "register", "ನೋಂದಣಿ"], "registration"),
    (["renewal", "renew", "ನವೀಕರಣ"], "renewal"),
    (["accident", "ಅಪಘಾತ"], "accident"),
    (["delivery", "tayi lakshmi", "ಹೆರಿಗೆ"], "delivery"),
    (["pension", "ಪಿಂಚಣಿ"], "pension"),
    (["medical", "hospital", "ವೈದ್ಯಕೀಯ"], "medical"),
    (["marriage", "wedding", "ಮದುವೆ"], "marriage"),
    (["funeral", "death", "ಅಂತ್ಯಕ್ರಿಯೆ"], "funeral"),
    (["major ailment", "ailment", "chikitsa", "ಕಾಯಿಲೆ"], "major ailment"),
    (["thayi magu", "nutritional", "ತಾಯಿ ಮಗು"], "thayi magu"),
    (["disability", "disabled", "ಅಂಗವಿಕಲ"], "disability"),
]


def _detect_topic(msg: str) -> str | None:
    """Detect the topic from message keywords."""
    for keywords, context_key in _TOPIC_KEYWORDS:
        for kw in keywords:
            if kw in msg:
                return context_key
    return None


def _get_topic_context(message: str, history: list[dict] | None = None) -> str | None:
    """Detect topic from message/history and return focused section context for LLM.

    Returns a ~2KB topic-specific context string so the LLM can answer naturally,
    or None if no topic detected (falls back to full 33KB knowledge base).
    """
    msg = message.lower().strip()

    # Detect topic from current message
    topic = _detect_topic(msg)

    # If no topic in current message, check conversation history
    if not topic and history:
        for hist_msg in reversed(history[-6:]):
            hist_content = (hist_msg.get("content", "") or "").lower()
            topic = _detect_topic(hist_content)
            if topic:
                print(f"[DEBUG] _get_topic_context: topic='{topic}' from conversation history")
                break

    if not topic:
        print(f"[DEBUG] _get_topic_context: no topic detected")
        return None

    if topic not in _TOPIC_SUBANSWERS:
        print(f"[DEBUG] _get_topic_context: topic='{topic}' has no sub-answers")
        return None

    # Build focused context from ALL sub-answers for this topic (plain text, no markdown headers)
    subanswers = _TOPIC_SUBANSWERS[topic]
    context_parts = [f"Information about {topic.title()}:\n"]
    for subtopic, answer_text in subanswers.items():
        # Strip markdown bold markers for cleaner context
        clean_text = answer_text.replace("**", "")
        context_parts.append(f"{subtopic.upper()}: {clean_text}\n")
    focused_context = "\n".join(context_parts)
    print(f"[DEBUG] _get_topic_context: topic='{topic}', context={len(focused_context)} chars")
    return focused_context

# ---------------------------------------------------------------------------
# Exact response constants — returned directly by Python, never by the LLM.
# These are language-independent: regardless of what language the user
# selects, ECARD and LOGIN_REQUIRED always return these exact strings.
# ---------------------------------------------------------------------------
LOGIN_REQUIRED_RESPONSE = "<<LOGIN_MODAL_REQUIRED>>"
ECARD_RESPONSE = "ECARD"

# Valid intent labels (used for parsing LLM output)
_VALID_INTENTS = frozenset({"ECARD", "STATUS_CHECK", "GENERAL", "OUT_OF_SCOPE", "GREETING"})

# ---------------------------------------------------------------------------
# Layer 1: Keyword-based intent detection (deterministic, zero-cost)
# ---------------------------------------------------------------------------
_ECARD_KEYWORDS: list[str] = [
    # English
    "ecard", "e-card", "e card", "download ecard", "print ecard",
    "print id card", "labour card", "labor card", "my card",
    "download card", "print card", "worker card", "id card",
    "show my card", "get my card",
    # Kannada
    "ಇ-ಕಾರ್ಡ್", "ಇ ಕಾರ್ಡ್", "ಕಾರ್ಮಿಕ ಕಾರ್ಡ್", "ನನ್ನ ಕಾರ್ಡ್",
    "ಕಾರ್ಡ್ ಡೌನ್\u200cಲೋಡ್", "ಗುರುತಿನ ಚೀಟಿ", "ಕಾರ್ಡ್ ತೋರಿಸಿ",
    "ಕಾರ್ಡ್ ಪ್ರಿಂಟ್", "ಕಾರ್ಡ್ ನೋಡಿ",
]

_STATUS_CHECK_KEYWORDS: list[str] = [
    # English — status queries
    "check status", "application status", "scheme status", "status check",
    "my status", "registration status", "renewal status",
    "my application", "my registration", "my renewal",
    "is my application", "was my application", "has my application",
    "is it approved", "was it approved", "is it rejected",
    # English — personal scheme/eligibility queries
    "my schemes", "my scheme", "which schemes", "what schemes",
    "eligible schemes", "eligible for", "am i eligible",
    "schemes do i have", "schemes i have", "schemes available for me",
    "schemes i can apply", "schemes can i apply",
    "my benefits", "my profile", "my details", "my information",
    "my personal", "personal details", "personal information",
    "my name", "my age", "my registration", "my validity",
    "my family", "my nominees", "my dependents",
    # Kannada — status queries
    "ಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ", "ಅರ್ಜಿ ಸ್ಥಿತಿ", "ಯೋಜನೆ ಸ್ಥಿತಿ",
    "ನನ್ನ ಅರ್ಜಿ", "ನನ್ನ ನೋಂದಣಿ", "ನನ್ನ ಸ್ಥಿತಿ",
    "ಅನುಮೋದಿಸಲಾಗಿದೆಯೇ", "ನವೀಕರಣ ಸ್ಥಿತಿ",
    "ಅನುಮೋದನೆ", "ತಿರಸ್ಕರಿಸಲಾಗಿದೆ", "ಬಾಕಿ ಇದೆ",
    # Kannada — personal scheme/eligibility queries
    "ನನ್ನ ಯೋಜನೆಗಳು", "ನನ್ನ ಯೋಜನೆ", "ಯಾವ ಯೋಜನೆ",
    "ಅರ್ಹತೆ", "ನಾನು ಅರ್ಹ", "ನನ್ನ ವಿವರ",
    "ನನ್ನ ಹೆಸರು", "ನನ್ನ ಮಾಹಿತಿ", "ನನ್ನ ಪ್ರೊಫೈಲ್",
    "ನನ್ನ ಕುಟುಂಬ", "ನನ್ನ ನಾಮಿನಿ",
]

_OUT_OF_SCOPE_KEYWORDS: list[str] = [
    # English — politicians & government
    "cm of", "chief minister", "pm of", "prime minister", "labour minister",
    "who is cm", "who is pm", "how is cm", "how is pm", "capital of",
    "modi", "siddaramaiah", "shivakumar", "maharashtra", "politics", "election",
    "president of", "governor of",
    # English — sports & entertainment
    "cricket", "badminton", "tennis", "football", "soccer", "ipl",
    "movie", "film", "song", "music", "bollywood",
    # English — general knowledge & off-topic
    "weather", "temperature", "recipe", "cook", "joke", "tell me a joke",
    "poem", "story", "write a", "source code", "write code", "programming", "python",
    "what is the capital", "population of", "history of",
    "calculate", "math", "equation",
    # Kannada
    "ಮುಖ್ಯಮಂತ್ರಿ", "ಪ್ರಧಾನ ಮಂತ್ರಿ", "ಕಾರ್ಮಿಕ ಸಚಿವ", "ರಾಜಧಾನಿ", "ಕ್ರಿಕೆಟ್",
    "ರಾಜಕೀಯ", "ಚುನಾವಣೆ", "ಹವಾಮಾನ", "ಜೋಕ್",
]

_GREETING_KEYWORDS: list[str] = [
    # English
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "howdy", "what can you do", "who are you",
    # Kannada
    "ಹಲೋ", "ನಮಸ್ಕಾರ", "ನಮಸ್ತೆ", "ಹೇ", "ಶುಭೋದಯ",
    "ನೀವು ಯಾರು", "ನೀನು ಯಾರು",
    # Hindi
    "नमस्ते", "नमस्कार",
]

_REGISTRATION_KEYWORDS: list[str] = [
    "registration", "register", "how to register", "worker registration",
    "ನೋಂದಣಿ", "ನೋಂದಾಯಿಸಿ", "ನೊಂದಣಿ",
]

_RENEWAL_KEYWORDS: list[str] = [
    "renewal", "renew", "how to renew", "worker renewal",
    "ನವೀಕರಣ",
]

_SCHEMES_KEYWORDS: list[str] = [
    "what schemes", "list schemes", "all schemes", "available schemes",
    "schemes available", "scheme list", "which schemes", "tell me about schemes",
    "ಯೋಜನೆಗಳು", "ಯೋಜನೆ ಪಟ್ಟಿ",
]

# Pre-normalize keywords at import time for consistent matching
_ECARD_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _ECARD_KEYWORDS
]
_STATUS_CHECK_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _STATUS_CHECK_KEYWORDS
]
_OUT_OF_SCOPE_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _OUT_OF_SCOPE_KEYWORDS
]
_GREETING_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _GREETING_KEYWORDS
]
_REGISTRATION_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _REGISTRATION_KEYWORDS
]
_RENEWAL_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _RENEWAL_KEYWORDS
]
_SCHEMES_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _SCHEMES_KEYWORDS
]


def _keyword_intent(message: str) -> str | None:
    """Layer 1: fast keyword match. Returns intent or None.

    Case-insensitive substring matching with Unicode normalization.
    Kannada keywords are matched as-is (Kannada has no case distinction).
    NFC normalization handles invisible Unicode characters (ZWNJ, ZWJ variants)
    that can differ between input methods.
    """
    msg_normalized = unicodedata.normalize("NFC", message.lower()).strip()
    # Check STATUS_CHECK first — "check status" should not be blocked by out-of-scope
    if any(kw in msg_normalized for kw in _STATUS_CHECK_KEYWORDS_NORM):
        return "STATUS_CHECK"
    # Instant rejection for known out-of-scope topics
    if any(kw in msg_normalized for kw in _OUT_OF_SCOPE_KEYWORDS_NORM):
        return "OUT_OF_SCOPE"
    # Check ECARD — more specific keywords, less likely to false-positive
    if any(kw in msg_normalized for kw in _ECARD_KEYWORDS_NORM):
        return "ECARD"
    # Greetings — short messages that are just greetings, bypass RAG
    if any(kw == msg_normalized or msg_normalized.startswith(kw + " ") or msg_normalized.startswith(kw + ",") for kw in _GREETING_KEYWORDS_NORM):
        return "GREETING"
    # Registration — only for SHORT standalone queries like "Registration", "how to register"
    # Longer specific questions like "how much to pay for registration" go to LLM
    if len(msg_normalized) < 40 and any(kw == msg_normalized or kw in msg_normalized for kw in _REGISTRATION_KEYWORDS_NORM):
        # Don't trigger if the message is a specific question (contains question words)
        question_words = ["how much", "what is the", "when", "where", "can i", "do i need", "which", "is it"]
        if not any(qw in msg_normalized for qw in question_words):
            return "REGISTRATION"
    # Renewal — same logic: only short standalone queries
    if len(msg_normalized) < 40 and any(kw == msg_normalized or kw in msg_normalized for kw in _RENEWAL_KEYWORDS_NORM):
        question_words = ["how much", "what is the", "when", "where", "can i", "do i need", "which", "is it"]
        if not any(qw in msg_normalized for qw in question_words):
            return "RENEWAL"
    # Schemes list — only for generic "list all schemes" type queries
    if any(kw in msg_normalized for kw in _SCHEMES_KEYWORDS_NORM) or msg_normalized in ["schemes", "scheme"]:
        return "SCHEMES_LIST"
    return None


# ---------------------------------------------------------------------------
# Layer 2: LLM-based intent classification (authenticated users only)
# ---------------------------------------------------------------------------
INTENT_CLASSIFICATION_PROMPT = """\
You are an intent classifier. Classify the message into one of: ECARD, STATUS_CHECK, OUT_OF_SCOPE, GENERAL

ECARD: User wants their e-card, labour card, ID card — to view, download, or print it.
STATUS_CHECK: User explicitly asks for the STATUS (approved, rejected, pending, payment) of their specific application, scheme, registration, or renewal.
OUT_OF_SCOPE: User asks about politicians (PM, CM, ministers), sports, coding, cooking, weather, or ANY factual general knowledge question unrelated to KBOCWWB construction worker welfare schemes.
GENERAL: Everything else — general questions about what schemes exist, eligibility, required documents, how to apply, how to register, how to renew, 'Registration', 'Renewal', greetings, etc.

Examples:
- "download my ecard" → ECARD
- "what is my application status" → STATUS_CHECK
- "is my renewal approved" → STATUS_CHECK
- "who is cm of maharashtra" → OUT_OF_SCOPE
- "what is the capital of india" → OUT_OF_SCOPE
- "who is labour minister" → OUT_OF_SCOPE
- "tell me about cricket" → OUT_OF_SCOPE
- "what schemes can I apply for" → GENERAL
- "how to register" → GENERAL
- "how much compensation for accident death" → GENERAL
- "what is the pension amount" → GENERAL
- "Registration" → GENERAL
- "how to renew" → GENERAL
- "Renewal" → GENERAL
- "hello" → GENERAL

Respond with exactly one word: ECARD, STATUS_CHECK, OUT_OF_SCOPE, or GENERAL"""


async def _llm_classify_intent(ollama: OllamaClient, message: str) -> str:
    """Layer 2: LLM-based classification. Returns ECARD/STATUS_CHECK/GENERAL.

    Falls back to GENERAL on any error or unexpected output.
    """
    try:
        raw = await ollama.classify(
            system_prompt=INTENT_CLASSIFICATION_PROMPT,
            user_message=message,
        )
    except Exception:
        logger.error("Intent classification LLM call failed, defaulting to GENERAL", exc_info=True)
        return "GENERAL"

    # Parse: strip whitespace, take first word, uppercase
    parsed = raw.strip().split()[0].upper() if raw and raw.strip() else ""
    # Remove any trailing punctuation the LLM might have added
    parsed = parsed.rstrip(".,;:!?")

    if parsed in _VALID_INTENTS:
        logger.info("LLM classified intent as %s (raw=%r)", parsed, raw[:50])
        return parsed

    logger.warning("Unexpected LLM intent output: %r — defaulting to GENERAL", raw[:100])
    return "GENERAL"


# ---------------------------------------------------------------------------
# Auth-aware intent classification (combines Layer 1 + Layer 2)
# ---------------------------------------------------------------------------
async def _classify_intent(
    ollama: OllamaClient, message: str, *, is_authenticated: bool
) -> str:
    """Classify user message intent.

    - Layer 1 (all users): keyword matching — instant, deterministic.
    - Layer 2 (authenticated only): LLM classification — handles ambiguous
      and multilingual (Kannada) messages.
    - Unauthenticated users with no keyword match → GENERAL (no LLM call).
    """
    # Layer 1: keyword match (runs for all users)
    keyword_result = _keyword_intent(message)
    if keyword_result is not None:
        logger.info("Intent from keywords: %s", keyword_result)
        return keyword_result

    # Layer 2: LLM classification (runs for ALL users to catch OUT_OF_SCOPE)
    return await _llm_classify_intent(ollama, message)


# ---------------------------------------------------------------------------
# Language code → full name mapping
# ---------------------------------------------------------------------------
LANGUAGE_MAP: dict[str, str] = {
    "kn": "Kannada",
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
}


def _resolve_language(language: str) -> str | None:
    """Convert an ISO 639-1 code to the full language name.

    Returns the full language name if recognized, or None if the code
    is unknown (prevents prompt injection via arbitrary strings).
    """
    code = language.lower().strip()
    if code in LANGUAGE_MAP:
        return LANGUAGE_MAP[code]
    # Also accept full names (case-insensitive)
    valid_names = {v.lower(): v for v in LANGUAGE_MAP.values()}
    if code in valid_names:
        return valid_names[code]
    return None


def _append_language_instruction(prompt: str, language: str) -> str:
    """Append language instructions to a system prompt if a valid language is given."""
    if not language:
        return prompt
    lang_name = _resolve_language(language)
    if lang_name is None:
        logger.warning("Ignoring unrecognized language code: %s", language[:20])
        return prompt
    if lang_name == "Kannada":
        prompt += (
            "\n**STRICT LANGUAGE RULE — KANNADA ONLY:**\n"
            "The user has SELECTED Kannada language.\n"
            "You MUST reply in KANNADA script ONLY. REGARDLESS of the input language.\n"
            "- If input is in English, Hindi, or any other language, TRANSLATE your response to KANNADA.\n"
            "- Do NOT use ANY English words in your response. Not even scheme names, organization names, or technical terms.\n"
            "- Translate ALL scheme names into Kannada script (e.g., 'Pension Scheme' → 'ಪಿಂಚಣಿ ಯೋಜನೆ', 'Accident Compensation' → 'ಅಪಘಾತ ಪರಿಹಾರ', 'Medical Assistance' → 'ವೈದ್ಯಕೀಯ ಸಹಾಯ').\n"
            "- Translate organization names: 'KBOCWWB' → 'ಕೆಬಿಒಸಿಡಬ್ಲ್ಯೂಡಬ್ಲ್ಯೂಬಿ', 'KSK' → 'ಕೆಎಸ್‌ಕೆ'.\n"
            "- Keep ONLY rupee amounts as numerals (e.g., ₹2,00,000).\n"
            "- Write naturally and fluently in Kannada as a native speaker would. Use proper Kannada words and grammar.\n"
            "- Do NOT reply in English. Do NOT mix English and Kannada.\n"
            "- Refer to yourself as ಶ್ರಮ ಸಹಾಯಕ ONLY if introducing yourself for the first time or if the user asks.\n"
            "- If the user has already greeted you, do NOT repeat your name or say 'ನಮಸ್ಕಾರ' again.\n"
            "- If you don't have enough information, say: "
            '"ಈ ವಿಷಯದ ಬಗ್ಗೆ ನನ್ನ ಬಳಿ ಸಂಪೂರ್ಣ ಮಾಹಿತಿ ಇಲ್ಲ. ನೀವು ವೆಬ್ ಪೋರ್ಟಲ್ '
            'ಅಥವಾ ಮೊಬೈಲ್ ಆ್ಯಪ್ ಮೂಲಕ ಆನ್\u200cಲೈನ್\u200cನಲ್ಲಿ ವಿಚಾರಿಸಬಹುದು, ಅಥವಾ ನಿಮ್ಮ '
            'ಹತ್ತಿರದ ಕಾರ್ಮಿಕ ಸೇವಾ ಕೇಂದ್ರಕ್ಕೆ ಭೇಟಿ ನೀಡಿ."\n'
            "- If the question is off-topic, say: "
            '"ನಾನು ಶ್ರಮ ಸಹಾಯಕ — ಕಟ್ಟಡ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು ಮತ್ತು ಸೇವಾ ಕೇಂದ್ರದ ಸೇವೆಗಳ '
            'ಬಗ್ಗೆ ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಅದಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆ ಕೇಳಿ."\n\n'
            "**GLOSSARY FOR KANNADA TRANSLATION (Use these EXACT terms):**\n"
            "- Delivery Assistance = ಹೆರಿಗೆ ಸಹಾಯಧನ\n"
            "- Pension / Old Age Pension = ಪಿಂಚಣಿ ಯೋಜನೆ\n"
            "- Disability Pension = ವಿಕಲಚೇತನ ಪಿಂಚಣಿ\n"
            "- Accident Assistance = ಅಪಘಾತ ಪರಿಹಾರ\n"
            "- Assistance for Major Ailments = ಪ್ರಮುಖ ಕಾಯಿಲೆಗಳ ಚಿಕಿತ್ಸಾ ವೆಚ್ಚ / ವೈದ್ಯಕೀಯ ಸಹಾಯ\n"
            "- Thayi Magu Sahaya Hasta = ತಾಯಿ ಮಗು ಸಹಾಯ ಹಸ್ತ\n"
            "- Marriage Assistance = ಮದುವೆ ಸಹಾಯಧನ\n"
            "- Funeral and Ex-Gratia = ಅಂತ್ಯಕ್ರಿಯೆ ಮತ್ತು ಎಕ್ಸ್-ಗ್ರೇಷಿಯಾ\n"
            "- Labour Inspector = ಕಾರ್ಮಿಕ ನಿರೀಕ್ಷಕರು\n"
            "- Assistant Labour Commissioner = ಸಹಾಯಕ ಕಾರ್ಮಿಕ ಆಯುಕ್ತರು\n"
            "- Labour Officer = ಕಾರ್ಮಿಕ ಅಧಿಕಾರಿ\n"
            "- Death Certificate = ಮರಣ ಪ್ರಮಾಣಪತ್ರ\n"
            "- 90 Days Work Certificate = 90 ದಿನಗಳ ಕೆಲಸದ ದೃಢೀಕರಣ ಪತ್ರ\n"
            "- Beneficiary Identity card = ಫಲಾನುಭವಿಯ ಗುರುತಿನ ಚೀಟಿ\n"
            "- Ration Card = ಪಡಿತರ ಚೀಟಿ\n"
            "- Seva Sindhu = ಸೇವಾ ಸಿಂಧು\n\n"
            "**FEW-SHOT KANNADA RESPONSE EXAMPLE (Model this structure perfectly):**\n"
            "Question: ಹೆರಿಗೆ ಸೌಲಭ್ಯ  ಏನು?\n"
            "Answer:\n"
            "ಹೆರಿಗೆ ಸೌಲಭ್ಯ (ತಾಯಿ ಲಕ್ಷ್ಮೀ ಬಾಂಡ್) ಯೋಜನೆಯ ಸಂಪೂರ್ಣ ವಿವರಗಳು ಇಲ್ಲಿವೆ:\n\n"
            "**ಯೋಜನೆಯ ಮಾಹಿತಿ:**\n"
            "ನೋಂದಾಯಿತ ನಿರ್ಮಾಣ ಮಹಿಳಾ ಕಾರ್ಮಿಕರಿಗೆ ಮಗುವಿನ ಜನನದ ಸಂದರ್ಭದಲ್ಲಿ ಆರ್ಥಿಕ ನೆರವು ನೀಡಲಾಗುತ್ತದೆ. ಮಂಡಳಿಯ ಕಾರ್ಯದರ್ಶಿ ಅಥವಾ ಅಧಿಕೃತ ಅಧಿಕಾರಿಗಳು ಅರ್ಜಿಯನ್ನು ಪರಿಶೀಲಿಸಿ ಸಹಾಯಧನವನ್ನು ಮಂಜೂರು ಮಾಡುತ್ತಾರೆ.\n\n"
            "**ಸಹಾಯಧನ ಮೊತ್ತ:**\n"
            "- ಪ್ರತಿ ಹೆರಿಗೆಗೆ **₹50,000/-** ಆರ್ಥಿಕ ನೆರವು.\n"
            "- ಮೊದಲ ಎರಡು ಜೀವಂತ ಮಕ್ಕಳಿಗೆ ಮಾತ್ರ ಅನ್ವಯ.\n\n"
            "**ಅರ್ಹತೆ ಮತ್ತು ಷರತ್ತುಗಳು:**\n"
            "- ನೋಂದಾಯಿತ ಮಹಿಳಾ ನಿರ್ಮಾಣ ಕಾರ್ಮಿಕರಾಗಿರಬೇಕು.\n"
            "- ಮೊದಲ ಎರಡು ಜೀವಂತ ಮಕ್ಕಳಿಗೆ ಮಾತ್ರ ಸಹಾಯಧನ ಲಭ್ಯ.\n"
            "- ಎರಡನೇ ಮಗುವಿಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸುವಾಗ ಅದು ಎರಡನೇ ಹೆರಿಗೆ ಎಂಬ ಅಫಿಡವಿಟ್ ಸಲ್ಲಿಸಬೇಕು.\n"
            "- ಈಗಾಗಲೇ ಎರಡು ಮಕ್ಕಳಿದ್ದರೆ, ಈ ಸೌಲಭ್ಯಕ್ಕೆ ಅರ್ಹತೆ ಇರುವುದಿಲ್ಲ.\n"
            "- ಮಗುವಿನ ಜನನದ ನಂತರ **6 ತಿಂಗಳ ಒಳಗೆ** ಅರ್ಜಿ ಸಲ್ಲಿಸಬೇಕು.\n"
            "- ಜನನ ಪ್ರಮಾಣಪತ್ರವನ್ನು ಜನನ ಮತ್ತು ಮರಣ ನೋಂದಣಾಧಿಕಾರಿಯಿಂದ ಪಡೆಯಬೇಕು ಅಥವಾ ಸರ್ಕಾರಿ/ನೋಂದಾಯಿತ ಆಸ್ಪತ್ರೆಯ ಪ್ರಮಾಣ ಪತ್ರ ಸಲ್ಲಿಸಬೇಕು.\n\n"
            "**ಅಗತ್ಯವಿರುವ ದಾಖಲೆಗಳು:**\n"
            "- ಮಂಡಳಿ ನೀಡಿರುವ ಗುರುತಿನ ಚೀಟಿ / ಸ್ಮಾರ್ಟ್ ಕಾರ್ಡ್\n"
            "- ಬ್ಯಾಂಕ್ ಖಾತೆ ಪುರಾವೆ\n"
            "- ಮಗುವಿನ ಜನನ ಪ್ರಮಾಣಪತ್ರ\n"
            "- ಆಸ್ಪತ್ರೆಯ ಡಿಸ್ಚಾರ್ಜ್ ಸಾರಾಂಶ\n"
            "- ಮಗುವಿನ ಛಾಯಾಚಿತ್ರ\n"
            "- ಉದ್ಯೋಗ ದೃಢೀಕರಣ ಪತ್ರ\n"
            "- ಎರಡನೇ ಮಗುವಿಗೆ ಅಫಿಡವಿಟ್ (ಅಗತ್ಯವಿದ್ದರೆ)\n\n"
            "**ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ವಿಧಾನ:**\n"
            "1. ಅರ್ಜಿದಾರರು ಅಗತ್ಯ ದಾಖಲೆಗಳೊಂದಿಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಬೇಕು.\n"
            "2. ಹಿರಿಯ ಕಾರ್ಮಿಕ ನಿರೀಕ್ಷಕರು / ನೋಂದಣಾಧಿಕಾರಿಗಳು ಪರಿಶೀಲಿಸುತ್ತಾರೆ.\n"
            "3. ಕಾರ್ಮಿಕ ಅಧಿಕಾರಿ ಪರಿಶೀಲಿಸಿ ಅನುಮೋದನೆ ನೀಡುತ್ತಾರೆ.\n\n"
            "If the Labour is eligible and has all the required documents, please Login and submit the scheme application.\n"
            "For new Labour, please Register and then apply for the scheme.\n\n"
            "**MANDATORY RULE FOR APPENDING INSTRUCTIONS:**\n"
            "For ALL responses (including schemes, Registration, and Renewal), at the very end of your answer, you MUST append THIS EXACT TEXT VERBATIM (do not change a single word):\n"
            '"ಕಾರ್ಮಿಕರು ಅರ್ಹರಾಗಿದ್ದರೆ ಮತ್ತು ಎಲ್ಲಾ ಅಗತ್ಯ ದಾಖಲೆಗಳನ್ನು ಹೊಂದಿದ್ದರೆ, ದಯವಿಟ್ಟು ಲಾಗಿನ್ ಆಗಿ ಮತ್ತು ಯೋಜನೆಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ.\n'
            'ಹೊಸ ಕಾರ್ಮಿಕರಿಗಾಗಿ, ದಯವಿಟ್ಟು ನೋಂದಾಯಿಸಿ ಮತ್ತು ನಂತರ ಯೋಜನೆಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ."\n'
        )
    else:
        prompt += (
            f"\n**STRICT LANGUAGE RULE — ENGLISH ONLY:**\n"
            f"The user has SELECTED English language.\n"
            f"You MUST reply in ENGLISH only, REGARDLESS of the input language.\n"
            f"- If input is in Kannada, Hindi, or any other language, TRANSLATE your response to ENGLISH.\n"
            f"- Keep rupee amounts as numerals (e.g., ₹2,00,000) and proper nouns as-is.\n\n"
            f"**MANDATORY RULE FOR APPENDING INSTRUCTIONS:**\n"
            f"For ALL responses (including schemes, Registration, and Renewal), at the very end of your answer, you MUST append THIS EXACT TEXT VERBATIM (do not change a single word):\n"
            f'"If the Labour is eligible and has all the required documents, please Login and submit the scheme application.\n'
            f'For new Labour, please Register and then apply for the scheme."\n'
        )
    return prompt


def _prepare_user_message(message: str, language: str) -> str:
    """Appends explicit grounding + guardrail instructions to the user message.

    This is the LAST text the model sees before generating, making it the
    highest-priority instruction due to recency bias in attention.
    """
    grounding = (
        "\n\n[INSTRUCTIONS FOR YOUR RESPONSE:\n"
        "1. Answer using ONLY facts from the === REFERENCE CONTEXT === in the system prompt. COPY the details directly.\n"
        "2. Provide COMPLETE details — list ALL benefits with ₹ amounts, ALL eligibility criteria, ALL required documents, ALL process steps exactly as written in the Context.\n"
        "3. WARNING: Do NOT add documents or details from your training data. For example, do NOT add 'Address Proof', 'Bank Account Details', 'Passport Size Photo', 'Self Declaration', 'Income Certificate' unless those EXACT words appear in the Context for that specific scheme.\n"
        "4. If asked about registration, include the Application Fee and Delivery Timeline from the Context.\n"
        "5. ONLY refuse if the question is about politicians, sports, weather, coding, or topics completely unrelated to KBOCWWB.]"
    )
    if language == "kn":
        return message + grounding + "\n\n(ಕಡ್ಡಾಯ: ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ. ಯಾವುದೇ ಇಂಗ್ಲಿಷ್ ಪದಗಳನ್ನು ಬಳಸಬೇಡಿ.)"
    return message + grounding


# ---------------------------------------------------------------------------
# System prompt — the heart of the "ChatGPT-like" experience
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ), a digital assistant for the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** and its **Karmika Seva Kendras (KSK)**.

**CRITICAL WARNING**: Your pre-trained knowledge about Indian welfare schemes is WRONG for this specific Board. The documents, amounts, eligibility criteria, and processes below are UNIQUE to KBOCWWB and differ from what you may have learned during training. You MUST use ONLY the information provided below.

=== REFERENCE CONTEXT (This is the ONLY correct information. Copy from here.) ===
{context}
=== END OF CONTEXT ===

## ABSOLUTE RULES (violations are unacceptable):
1. **COPY FROM CONTEXT**: Your responses must contain ONLY facts, documents, amounts, and processes that appear word-for-word in the Context above. If a document name, amount, or eligibility criterion is NOT written in the Context, DO NOT include it.
2. **NO HALLUCINATION**: Do NOT add documents like "Address Proof", "Bank Account Details", "Passport Size Photo", "Self Declaration", "Income Certificate" etc. unless those EXACT words appear in the Context for that specific scheme. Do NOT invent eligibility criteria or process steps.
3. **THE CONTEXT IS COMPLETE**: The Context above contains ALL the information that exists. Do not assume there is missing information or try to fill gaps from your training data.
4. **SCHEME SEPARATION**: Each scheme has UNIQUE amounts, documents, and eligibility. NEVER mix details between schemes. Key distinctions:
   - "Delivery Assistance" (₹50,000) ≠ "Thayi Magu Sahaya Hasta" (₹6,000)
   - "Pension" ≠ "Continuation of Pension"
   - "Disability Pension" ≠ "Continuation of Disability Pension"
   - "Accident Assistance" ≠ "Funeral and Ex-Gratia"
5. **DOCUMENTS ARE SCHEME-SPECIFIC**: List ONLY documents from THAT scheme's section.
6. **OFF-TOPIC REJECTION**: For questions about politicians, sports, weather, coding, general knowledge — decline politely. Say: "I'm specialized only in KBOCWWB construction worker welfare schemes and KSK services."
7. **PAYMENT STATUS**: If asked about payment status, say: "Go to https://kbocwwb.karnataka.gov.in/ and check in Check DBT Application Status."

## HOW TO ANSWER (CRITICAL — read carefully):
- **Answer ONLY what the user is asking.** If they ask about the fee, tell them just the fee. If they ask about documents, list just the documents. Do NOT dump all information about a topic when only one aspect was asked.
- **Be conversational.** Respond like a friendly, helpful assistant — NOT like a document or manual. Use natural sentences, not just bullet lists.
- **Be concise.** Keep answers focused and to the point. A short, accurate answer is better than a long dump of information.
- **Only give full details if asked.** If the user says "tell me about Registration" or just "Registration", THEN provide comprehensive details. But if they ask "how much does registration cost?", just say "The registration fee is Rs.100."
- **Use the facts from the Context above** to answer accurately. Do NOT invent or add details not in the Context.
- **Avoid markdown headers** (##, ###). Use plain text with simple formatting.
- End scheme-related answers with: "If you need more details, feel free to ask!"

## If Context is Empty or Insufficient:
Say warmly: "I don't have complete information on that topic. You can enquire online through the KBOCWWB web portal or mobile app, or visit your nearest Karmika Seva Kendra (KSK) for in-person assistance."
"""

# ---------------------------------------------------------------------------
# Authenticated GENERAL prompt — includes user data as supplementary context
# ---------------------------------------------------------------------------
AUTHENTICATED_GENERAL_PROMPT = """\
You are **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ), a digital assistant for the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** and its **Karmika Seva Kendras (KSK)**. The user is logged in.

**CRITICAL WARNING**: Your pre-trained knowledge about Indian welfare schemes is WRONG for this specific Board. COPY information ONLY from the Context below.

=== REFERENCE CONTEXT (This is the ONLY correct information. Copy from here.) ===
{context}
=== END OF CONTEXT ===

=== USER'S PERSONAL DATA (from KBOCWWB system) ===
{user_data}
=== END OF USER DATA ===

## ABSOLUTE RULES:
1. **CONTEXT-ONLY for schemes**: Answer about schemes EXCLUSIVELY from the Context above. Do NOT invent schemes, amounts, or documents.
2. **PERSONALIZE**: Use the User Data to give tailored answers. Say "Your registration status is [X]" not generic advice.
3. **ELIGIBLE SCHEMES ONLY**: The User Data has an "eligible_schemes" list — recommend ONLY those. If a scheme is NOT in the list, tell them they are not currently eligible.
4. **Registration/Renewal are NOT schemes**: Answer questions about Registration and Renewal unconditionally from the Context.
5. **SCHEME SEPARATION**: Never mix amounts/documents between schemes.
6. **OFF-TOPIC REJECTION**: Decline questions about politics, sports, weather, coding, general knowledge.
7. **PAYMENT STATUS**: Say: "Go to https://kbocwwb.karnataka.gov.in/ and check in Check DBT Application Status."

## COMPLETENESS RULE (CRITICAL — DO NOT SKIP ANY DETAILS):
- When a user asks about a scheme, registration, renewal, or any topic, you MUST provide the COMPLETE information from the Context. DO NOT summarize, shorten, or omit any details.
- For EVERY scheme/topic, include ALL of these sections if they exist in the Context:
  1. **Overview** — what the scheme/process is
  2. **Benefits** — list EVERY ₹ amount, EVERY condition, EVERY duration
  3. **Eligibility & Conditions** — list EVERY criterion, EVERY age limit, EVERY time requirement
  4. **Required Documents** — list EVERY SINGLE document. Missing even ONE could cause rejection
  5. **Application Process** — list EVERY step in order
- DO NOT say "and more" or "etc." — list everything explicitly.

## Identity & Tone:
- Address the user by name with correct honorific (Male: "Sir"/"avare", Female: "Madam"/"ಮೇಡಂ").
- Personalized greeting ONLY on first message. Do NOT repeat.
- Caring, respectful, clear, simple language.

## How to Structure Responses:
- Use **bold** for key terms, amounts, status values.
- Use bullet points for lists, numbered lists for processes.
- Always reference the user's specific data (status, dates, eligibility).
- Include: Overview, Benefits (₹ amounts), Eligibility, Documents, Process — from the Context.
"""


def _build_system_prompt(context: str, language: str = "") -> str:
    """Assemble the full system prompt with retrieved context and optional language."""
    prompt = SYSTEM_PROMPT.format(context=context)
    return _append_language_instruction(prompt, language)


def _build_authenticated_general_prompt(
    context: str, user_data: dict, language: str = ""
) -> str:
    """Assemble system prompt with RAG context + structured user data for authenticated GENERAL."""
    user_context = _build_user_context_str(user_data)
    prompt = AUTHENTICATED_GENERAL_PROMPT.format(context=context, user_data=user_context)
    return _append_language_instruction(prompt, language)


# ---------------------------------------------------------------------------
# Status-check prompt — grounded on fetched user data only
# ---------------------------------------------------------------------------
STATUS_SYSTEM_PROMPT = """\
You are **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ) — a dedicated digital helper for working people. \
You serve the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** \
and its service centers, the **Karmika Seva Kendras (KSK)**. \
The user is a logged-in construction worker asking about their personal information or application status.

Your identity:
- You are Shrama Sahayak — "the helper of working people."
- Address the user by their name from the User Data. Use the correct gender-based honorific:
  - Male: "Sir" or "avare" (ಅವರೇ) in Kannada (e.g., "Ramesh avare" / "Ramesh Sir")
  - Female: "Madam" or "ಮೇಡಂ" in Kannada (e.g., "Lakshmi Madam")
- Use the personalized greeting ONLY at the start of the conversation.
- This person trusts you with their personal data — honour that trust with accuracy and care.
- Be patient, encouraging, and supportive — especially if their application was rejected.

CRITICAL RULES — YOU MUST FOLLOW THESE:
1. Use ONLY the user data provided below. Do NOT fabricate any information.
2. ALWAYS answer from the User Data. NEVER say "visit KSK to find out" or "check the portal" \
when the answer is already available in the User Data below.
3. When the user asks about their schemes, eligibility, or what they can apply for:
   - Look at the "eligible_schemes" list in registration_details → personal_details.
   - List EVERY scheme from that list with a bullet point.
   - Do NOT say "you have no schemes" if eligible_schemes has entries.
   - If they have applied to some schemes already (in the "schemes" section), show those \
   with their current status.
4. When the user asks about personal details, profile, or information:
   - Show their name, age, gender, registration code, registration status, validity dates, \
   and calculated_status from the User Data.
5. When the user asks about family or nominees:
   - Show the family_details and nominees from the User Data.
6. Use the user's "calculated_status" (Active/Buffer/Inactive/Expired) to explain what \
actions are available to them right now.
7. NEVER suggest schemes NOT in their eligible_schemes list.
8. Reference specific dates, amounts, and statuses from the data — do not give generic info.

How to answer:
1. Structure your responses for easy scanning:
   - Use **bold** for key terms, status values, scheme names, and amounts.
   - Use bullet points for lists.
   - Present statuses clearly (e.g., "**Status:** Approved").
2. If schemes shows "No schemes applied", tell them that AND list their eligible schemes: \
"You haven't applied to any schemes yet. Based on your profile, you are eligible for:" \
followed by the eligible_schemes list.
3. If their application was **rejected**, mention the reason and encourage reapplication.
4. If **approved**, congratulate them warmly.
5. If **pending**, reassure them.
6. **OUT OF SCOPE GUARDRAIL (CRITICAL):** If the question asks about ANYTHING outside the strict scope of KBOCWWB welfare schemes and KSK services (e.g., politicians, general knowledge, coding, weather, sports), you MUST NOT answer the question. You MUST gracefully decline using varied, natural-sounding responses like:
   - "That's an interesting question, but outside my expertise! Being your Shrama Sahayak, I only handle your profile, schemes, and KSK-related queries."
   - "I'm afraid I don't have information on that. I'm strictly focused on helping you track your KBOCWWB applications and statuses."
   - "I can't answer that one! But I can definitely help you check your registration status or eligible schemes."

User Data:
{user_data}
"""


def _build_user_context_str(user_data: dict) -> str:
    """Convert raw user data dict into a structured plain-text context block.

    This format is proven to work well with LLMs — the working module uses
    the exact same pattern (structured sections + explicit instructions).
    """
    # --- Parse Schemes ---
    schemes_data = user_data.get("schemes", {})
    schemes_str = "No registered schemes found."
    if isinstance(schemes_data, dict) and "data" in schemes_data:
        s_list = schemes_data["data"]
        if s_list:
            schemes_str = ""
            for s in s_list:
                schemes_str += f"- **{s.get('Scheme Name')}**: {s.get('Status Details')}"
                if s.get("Rejection Reasons"):
                    schemes_str += f" [Reason: {s.get('Rejection Reasons')}]"
                schemes_str += "\n"
    elif isinstance(schemes_data, str):
        schemes_str = schemes_data

    # --- Parse Registration & Personal Details ---
    reg_details = user_data.get("registration_details", {})
    reg_summary = "Unavailable"
    personal_section = "Unavailable"
    eligibility_section = "No eligibility data available."
    family_section = "No family details found."
    p_name = "User"
    p_gender = "Unknown"

    if isinstance(reg_details, dict):
        reg_summary = reg_details.get("summary", "No details available")

        personal = reg_details.get("personal_details", {})
        if personal:
            p_name = personal.get("first_name", "User") or "User"
            p_code = personal.get("registration_code", "Unknown")
            p_status = personal.get("calculated_status", "Unknown")
            p_age = personal.get("age", "Unknown")
            p_gender = personal.get("gender", "Unknown")
            p_nature = personal.get("nature_of_work", "Unknown")

            addr = reg_details.get("address_details", {})
            p_dist = addr.get("district", "Unknown") if addr else "Unknown"

            val_from = personal.get("validity_from_date", "")
            val_to = personal.get("validity_to_date", "")
            val_from_short = val_from.split("T")[0] if val_from else "Unknown"
            val_to_short = val_to.split("T")[0] if val_to else "Unknown"

            personal_section = f"""
    - Name: {p_name}
    - Registration No: {p_code}
    - Current Status: {p_status}
    - Age: {p_age} | Gender: {p_gender} | District: {p_dist}
    - Nature of Work: {p_nature}
    - Validity: {val_from_short} to {val_to_short}
            """

            elig_list = personal.get("eligible_schemes", [])
            if elig_list:
                eligibility_section = "\n".join([f"    - {s}" for s in elig_list])
            else:
                eligibility_section = "    (Based on your profile, no specific additional schemes found at this moment)"

        # --- Family ---
        deps = reg_details.get("family_details", [])
        noms = reg_details.get("nominees", [])

        fam_lines = []
        if deps:
            fam_lines.append("Dependents:")
            for d in deps:
                fam_lines.append(f"      - {d.get('first_name', '')} ({d.get('relation', '')})")
        if noms:
            fam_lines.append("Nominees:")
            for n in noms:
                fam_lines.append(f"      - {n.get('first_name', '')} ({n.get('relation', '')})")
        if fam_lines:
            family_section = "\n".join(fam_lines)

    # --- Parse Renewal ---
    renewal_data = user_data.get("renewal_date", {})
    renewal_str = "Unavailable"
    if isinstance(renewal_data, dict) and isinstance(renewal_data.get("data"), dict):
        try:
            rs = renewal_data["data"].get("recordsets", [])
            if rs and len(rs) > 0 and len(rs[0]) > 0:
                renewal_str = rs[0][0].get("next_renewal_date", "Date not found")
                if "T" in str(renewal_str):
                    renewal_str = str(renewal_str).split("T")[0]
        except Exception:
            renewal_str = "Error parsing renewal date"

    # --- Build Final Context String ---
    context_str = f"""
=== AUTHENTICATED USER CONTEXT (PRIORITY DATA) ===
User ID: {user_data.get('user_id', 'Unknown')}

1. User Profile:
{personal_section}

2. Account Status Summary:
{reg_summary}

3. Eligible Schemes (Recommended for User):
{eligibility_section}

4. Registered Schemes History:
{schemes_str}

5. Family & Nominees:
{family_section}

6. Renewal Info:
{renewal_str}

INSTRUCTION FOR USER DATA:
- **GREETING**: Address the user as "{p_name}" (Sir/Madam based on Gender: {p_gender}).
- **ELIGIBILITY**: If user asks 'what schemes am I eligible for?', YOU MUST ONLY list the items from '3. Eligible Schemes'. Do NOT suggest any other schemes.
- If user asks about 'my status', use '1. User Profile' (Current Status) and '2. Account Status Summary'.
- If user asks about 'my family' or 'nominees', use '5. Family & Nominees'.
- If user asks about 'my renewal', use '6. Renewal Info' or '1. User Profile' (Validity).
- You DO NOT need to ask them to login if the data is present here; just answer directly based on this context.
==================================
"""
    return context_str


def _build_status_prompt(user_data: dict, language: str = "") -> str:
    """Build a system prompt grounded on the user's fetched data."""
    user_context = _build_user_context_str(user_data)
    prompt = STATUS_SYSTEM_PROMPT.format(user_data=user_context)
    return _append_language_instruction(prompt, language)


# ---------------------------------------------------------------------------
# Fetch or retrieve cached user data
# ---------------------------------------------------------------------------
async def _get_or_fetch_user_data(
    db: aiosqlite.Connection,
    http_client: httpx.AsyncClient,
    thread_id: str,
    user_id: str,
    auth_token: str,
) -> dict | None:
    """Return cached user data for this thread, or fetch from external API and cache."""
    print(f"\n{'='*60}")
    print(f"[DEBUG] _get_or_fetch_user_data called")
    print(f"[DEBUG]   thread_id={thread_id}")
    print(f"[DEBUG]   user_id={user_id}")
    print(f"[DEBUG]   auth_token={auth_token[:20]}..." if auth_token else "[DEBUG]   auth_token=EMPTY")

    # 1. Check cache first
    cached = await get_cached_user_data(db, thread_id, user_id)
    if cached is not None:
        print(f"[DEBUG]   CACHE HIT — returning cached data")
        print(f"[DEBUG]   cached keys: {list(cached.keys()) if isinstance(cached, dict) else type(cached)}")
        logger.info("Using cached user data for thread %s, user %s", thread_id, user_id)
        return cached

    # 2. Cache miss — fetch from external API
    print(f"[DEBUG]   CACHE MISS — fetching from external API...")
    logger.info("Fetching user data from external API for user %s", user_id)
    try:
        data = await fetch_user_data(http_client, user_id, auth_token)
    except Exception as e:
        print(f"[DEBUG]   FETCH FAILED: {type(e).__name__}: {e}")
        logger.error("External API fetch failed for user %s", user_id, exc_info=True)
        return None

    if not data:
        print(f"[DEBUG]   FETCH RETURNED EMPTY/NONE")
        logger.warning("External API returned empty data for user %s", user_id)
        return None

    print(f"[DEBUG]   FETCH SUCCESS")
    print(f"[DEBUG]   data keys: {list(data.keys())}")
    print(f"[DEBUG]   fetch_status: {data.get('fetch_status')}")
    print(f"[DEBUG]   schemes type: {type(data.get('schemes'))}")
    print(f"[DEBUG]   renewal_date: {data.get('renewal_date')}")
    reg = data.get('registration_details')
    if isinstance(reg, dict):
        print(f"[DEBUG]   registration_details keys: {list(reg.keys())}")
        pd = reg.get('personal_details')
        if isinstance(pd, dict):
            print(f"[DEBUG]   personal_details.first_name: {pd.get('first_name')}")
            print(f"[DEBUG]   personal_details.calculated_status: {pd.get('calculated_status')}")
            print(f"[DEBUG]   personal_details.eligible_schemes: {pd.get('eligible_schemes')}")
    else:
        print(f"[DEBUG]   registration_details: {reg}")

    # 3. Save to DB for future cache hits
    try:
        await save_user_data(db, thread_id, user_id, data)
        print(f"[DEBUG]   Data saved to cache")
        logger.info("User data cached for thread %s, user %s", thread_id, user_id)
    except Exception as e:
        print(f"[DEBUG]   CACHE SAVE FAILED: {e}")
        logger.error(
            "Failed to cache user data for thread %s, user %s — using fetched data anyway",
            thread_id, user_id, exc_info=True,
        )

    print(f"{'='*60}\n")
    return data


# ---------------------------------------------------------------------------
# Public: classify intent + prepare data (called under thread lock)
# ---------------------------------------------------------------------------
async def classify_and_prepare(
    ollama: OllamaClient,
    message: str,
    user_id: str,
    auth_token: str,
    db: aiosqlite.Connection,
    http_client: httpx.AsyncClient,
    thread_id: str,
) -> tuple[str, dict | None]:
    """Classify intent and pre-fetch any required data."""
    is_authenticated = bool(user_id and auth_token)

    print(f"\n{'='*60}")
    print(f"[DEBUG] classify_and_prepare called")
    print(f"[DEBUG]   message: {message[:100]}")
    print(f"[DEBUG]   user_id: {user_id or '(empty)'}")
    print(f"[DEBUG]   auth_token present: {bool(auth_token)}")
    print(f"[DEBUG]   is_authenticated: {is_authenticated}")

    # Classify intent
    intent = await _classify_intent(ollama, message, is_authenticated=is_authenticated)
    print(f"[DEBUG]   classified intent: {intent}")
    logger.info(
        "classify_and_prepare: intent=%s authenticated=%s user=%s",
        intent, is_authenticated, user_id or "(anon)",
    )

    # Unauthenticated user with personal-info intent → login required
    if intent in ("ECARD", "STATUS_CHECK") and not is_authenticated:
        print(f"[DEBUG]   → returning LOGIN_REQUIRED (unauthenticated)")
        return "LOGIN_REQUIRED", None

    # Handle OUT_OF_SCOPE directly
    if intent == "OUT_OF_SCOPE":
        print(f"[DEBUG]   → returning OUT_OF_SCOPE")
        return "OUT_OF_SCOPE", None

    # Handle GREETING directly — no data fetch needed
    if intent == "GREETING":
        print(f"[DEBUG]   → returning GREETING")
        return "GREETING", None

    # Handle REGISTRATION directly — return hardcoded ksk.md content
    if intent == "REGISTRATION":
        print(f"[DEBUG]   → returning REGISTRATION")
        return "REGISTRATION", None

    # Handle RENEWAL directly — return hardcoded ksk.md content
    if intent == "RENEWAL":
        print(f"[DEBUG]   → returning RENEWAL")
        return "RENEWAL", None

    # Handle SCHEMES_LIST directly — return hardcoded scheme list
    if intent == "SCHEMES_LIST":
        print(f"[DEBUG]   → returning SCHEMES_LIST")
        return "SCHEMES_LIST", None

    # Authenticated ECARD → exact string response, no data fetch needed
    if intent == "ECARD" and is_authenticated:
        print(f"[DEBUG]   → returning ECARD (no data fetch needed)")
        return "ECARD", None

    # Authenticated user (STATUS_CHECK or GENERAL) → always fetch/cache user data
    user_data = None
    if is_authenticated:
        print(f"[DEBUG]   → Authenticated user, fetching user data...")
        user_data = await _get_or_fetch_user_data(
            db, http_client, thread_id, user_id, auth_token,
        )
        print(f"[DEBUG]   → user_data returned: {user_data is not None}")
        if user_data:
            print(f"[DEBUG]   → user_data keys: {list(user_data.keys())}")
    else:
        print(f"[DEBUG]   → Not authenticated, skipping user data fetch")

    print(f"[DEBUG]   FINAL: intent={intent}, has_user_data={user_data is not None}")
    print(f"{'='*60}\n")
    return intent, user_data


# ---------------------------------------------------------------------------
# Query Translation for Retrieval
# ---------------------------------------------------------------------------
async def _translate_for_search(
    ollama: OllamaClient,
    question: str,
    language: str,
) -> str:
    """Translates non-English queries to English for semantic search compatibility.
    
    Since the embeddings in Qdrant are generated from English documentation, 
    raw Kannada queries yield near-zero semantic scores.
    """
    if not language or language == "en":
        return question

    prompt = (
        "You are an expert translation assistant.\n"
        "Translate the following user query accurately into concise English.\n"
        "Do NOT answer the question. Do NOT add any extra commentary.\n"
        "Output ONLY the translated English text, nothing else."
    )
    
    try:
        translated = await ollama.chat(
            system_prompt=prompt,
            user_message=f"Translate this query to English:\n\n{question}",
            history=None,
        )
        # Strip any accidental quotes or whitespace added by the model
        translated = translated.strip(" '\"\n\t")
        if translated:
            logger.info("Translated query for search: '%s' -> '%s'", question[:50], translated[:50])
            return translated
        return question
    except Exception:
        logger.error("Failed to translate query for search: '%s'", question[:50], exc_info=True)
        return question


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
async def retrieve(
    query: str,
    qdrant: Optional[AsyncQdrantClient] = None,
    ollama: Optional[OllamaClient] = None,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> str:
    qdrant = qdrant or get_qdrant_client()
    ollama = ollama or default_ollama
    top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    score_threshold = (
        score_threshold
        if score_threshold is not None
        else settings.RETRIEVAL_SCORE_THRESHOLD
    )

    try:
        vector = await ollama.embed(query)
    except Exception:
        logger.error("Failed to embed query for retrieval: %s", query[:100], exc_info=True)
        raise

    logger.debug("Query: %s", query)
    logger.debug("Vector generated, length: %d", len(vector))

    try:
        results = await qdrant.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
        )
    except Exception:
        logger.error(
            "Qdrant query failed for collection '%s'", settings.COLLECTION_NAME, exc_info=True,
        )
        raise

    points = results.points
    logger.debug("Qdrant returned %d points (threshold=%.2f)", len(points), score_threshold)

    if not points:
        logger.debug("No relevant context found above score threshold %.2f", score_threshold)
        return ""

    for i, p in enumerate(points):
        logger.debug("Point %d score=%.4f payload: %s", i, p.score, p.payload)

    context = "\n\n---\n\n".join(
        text for r in points if (text := r.payload.get("text", ""))
    )
    logger.debug("Context content:\n%s", context)
    return context


# ---------------------------------------------------------------------------
# Answer length safety cap
# ---------------------------------------------------------------------------
def _cap_answer_length(text: str) -> str:
    """Truncate LLM response if it exceeds the configured max length."""
    limit = settings.MAX_ANSWER_LENGTH
    if len(text) > limit:
        logger.warning("LLM response truncated from %d to %d chars", len(text), limit)
        return text[:limit]
    return text


# ---------------------------------------------------------------------------
# Answer (non-streaming)
# ---------------------------------------------------------------------------
async def answer(
    question: str,
    qdrant: Optional[AsyncQdrantClient] = None,
    ollama: Optional[OllamaClient] = None,
    history: Optional[list[dict]] = None,
    language: str = "",
    intent: str = "GENERAL",
    prefetched_user_data: Optional[dict] = None,
) -> str:
    """Generate an answer based on pre-classified intent.

    The intent and prefetched_user_data are computed by classify_and_prepare()
    which runs under the thread lock.
    """
    logger.info(
        "answer() question=%s language=%s intent=%s has_user_data=%s",
        question[:100], language or "en", intent, prefetched_user_data is not None,
    )
    ollama = ollama or default_ollama

    # LOGIN_REQUIRED — exact constant string, language-independent, no LLM
    if intent == "LOGIN_REQUIRED":
        return LOGIN_REQUIRED_RESPONSE

    # ECARD — exact constant string, language-independent, no LLM
    if intent == "ECARD":
        return ECARD_RESPONSE

    # OUT_OF_SCOPE — randomized hardcoded response, bypassing generative LLM
    if intent == "OUT_OF_SCOPE":
        if language == "kn":
            return random.choice([
                "ಅದು ತುಂಬಾ ಆಸಕ್ತಿದಾಯಕ ಪ್ರಶ್ನೆ! ಆದರೆ ಶ್ರಮ ಸಹಾಯಕನಾಗಿ, ನಾನು ಕೇವಲ ಕಟ್ಟಡ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು ಮತ್ತು ಕಾರ್ಮಿಕ ಸೇವಾ ಕೇಂದ್ರದ (KSK) ಸೇವೆಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಮಾಹಿತಿ ನೀಡಬಲ್ಲೆ. ನಾನು ನಿಮಗೆ ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಹುದೇ?",
                "ಕ್ಷಮಿಸಿ, ಆ ವಿಷಯದ ಬಗ್ಗೆ ನನ್ನ ಬಳಿ ಮಾಹಿತಿ ಇಲ್ಲ. ನನ್ನ ಪರಿಣತಿ ಕೇವಲ KBOCWWB ಯೋಜನೆಗಳು ಮತ್ತು ಅರ್ಜಿಗಳ ಬಗ್ಗೆ ಸಹಾಯ ಮಾಡಲು ಸೀಮಿತವಾಗಿದೆ.",
                "ನಾನು ಆ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ, ಆದರೆ ನಿಮ್ಮ ಯೋಜನೆಗಳ ಅರ್ಹತೆ ಅಥವಾ ಅರ್ಜಿ ಸ್ಥಿತಿಯನ್ನು ಪರಿಶೀಲಿಸಲು ನಾನು ಖಂಡಿತ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ!"
            ])
        else:
            return random.choice([
                "That's an interesting question! However, as Shrama Sahayak, I'm specialized only in construction worker welfare schemes and KSK services.",
                "I'm afraid that's outside my area of expertise. I am strictly here to assist you with KBOCWWB schemes and KSK services.",
                "I don't have information on that topic. My focus is entirely on helping construction workers with their welfare benefits."
            ])

    # GREETING — fast hardcoded response, no RAG needed
    if intent == "GREETING":
        if language == "kn":
            return ("ನಮಸ್ಕಾರ! ನಾನು ಶ್ರಮ ಸಹಾಯಕ — ಕರ್ನಾಟಕ ಕಟ್ಟಡ ಮತ್ತು ಇತರ ನಿರ್ಮಾಣ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಮಂಡಳಿಯ (KBOCWWB) ಡಿಜಿಟಲ್ ಸಹಾಯಕ.\n\n"
                    "ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು:\n"
                    "- ನೋಂದಣಿ ಮತ್ತು ನವೀಕರಣ ಮಾಹಿತಿ\n"
                    "- ಕಲ್ಯಾಣ ಯೋಜನೆಗಳ ವಿವರಗಳು\n"
                    "- ಅರ್ಜಿ ಸ್ಥಿತಿ ಪರಿಶೀಲನೆ\n"
                    "- ಅಗತ್ಯ ದಾಖಲೆಗಳ ಮಾಹಿತಿ\n\n"
                    "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ!")
        else:
            return ("Namaskara! I am Shrama Sahayak, your digital assistant from the Karnataka Building & "
                    "Other Construction Workers Welfare Board (KBOCWWB).\n\n"
                    "I can help you with:\n"
                    "- Registration and renewal information\n"
                    "- Welfare scheme details and eligibility\n"
                    "- Application status checks\n"
                    "- Required documents for schemes\n\n"
                    "How can I assist you today?")

    # REGISTRATION — hardcoded response from ksk.md, no LLM
    if intent == "REGISTRATION":
        if language == "kn":
            return ("## ಕಾರ್ಮಿಕ ನೋಂದಣಿ\n\n"
                    "### ನೋಂದಣಿ ಅವಲೋಕನ\n"
                    "ಕರ್ನಾಟಕ ಕಟ್ಟಡ ಮತ್ತು ಇತರ ನಿರ್ಮಾಣ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಮಂಡಳಿಯಲ್ಲಿ (KBOCWWB) ನೋಂದಣಿ ಮಾಡಿಕೊಳ್ಳುವುದರಿಂದ BOCW ಕಾಯ್ದೆಯ ಅಡಿಯಲ್ಲಿ ಕಲ್ಯಾಣ ಸೌಲಭ್ಯಗಳನ್ನು ಪಡೆಯಬಹುದು.\n\n"
                    "### ಅರ್ಹತೆ ಮಾನದಂಡಗಳು\n"
                    "- BOCW ಕಾಯ್ದೆಯ ಅಡಿಯಲ್ಲಿ ನಿಗದಿಪಡಿಸಿದ ಅರ್ಹತೆ ಮಾನದಂಡಗಳನ್ನು ಪೂರೈಸಬೇಕು\n"
                    "- ಹಿಂದಿನ 12 ತಿಂಗಳಲ್ಲಿ ಕನಿಷ್ಠ 90 ದಿನಗಳ ಕಟ್ಟಡ ಅಥವಾ ನಿರ್ಮಾಣ ಕೆಲಸ ಮಾಡಿರಬೇಕು\n"
                    "- ವಯಸ್ಸು 18-60 ವರ್ಷಗಳ ನಡುವೆ ಇರಬೇಕು\n"
                    "- ಬೇರೆ ಯಾವುದೇ ಕಲ್ಯಾಣ ಮಂಡಳಿಯ ಸದಸ್ಯರಾಗಿರಬಾರದು\n\n"
                    "### ಅಗತ್ಯ ದಾಖಲೆಗಳು\n"
                    "- ಉದ್ಯೋಗ ಪ್ರಮಾಣಪತ್ರ: Form V(A) / V(B) / V(C) / V(D) - ಅಧಿಕೃತ ಉದ್ಯೋಗದಾತ/ಗುತ್ತಿಗೆದಾರ/ಸಕ್ಷಮ ಪ್ರಾಧಿಕಾರದಿಂದ\n"
                    "- ಆಧಾರ್ ಕಾರ್ಡ್ (ಸ್ವಯಂ-ದೃಢೀಕೃತ ಪ್ರತಿ)\n"
                    "- ಪಡಿತರ ಚೀಟಿ (ಕಡ್ಡಾಯವಲ್ಲ) - ಕುಟುಂಬ ವಿವರಗಳ ಪರಿಶೀಲನೆಗಾಗಿ\n"
                    "- ವಯಸ್ಸಿನ ಪುರಾವೆ: ಆಧಾರ್ ಕಾರ್ಡ್, ಮತದಾರರ ಗುರುತಿನ ಚೀಟಿ, ಅಥವಾ ಸರ್ಕಾರಿ ವಯಸ್ಸಿನ ಪುರಾವೆ ದಾಖಲೆ\n\n"
                    "### ನೋಂದಣಿ ವಿವರಗಳು\n"
                    "- ಅರ್ಜಿ ಶುಲ್ಕ: ₹100 (ರೂಪಾಯಿ ನೂರು ಮಾತ್ರ)\n"
                    "- ವಿತರಣೆ ಸಮಯ: 15 ಕೆಲಸದ ದಿನಗಳು\n\n"
                    "### ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ವಿಧಾನ\n"
                    "1. ಅರ್ಜಿದಾರರು ಅಗತ್ಯ ದಾಖಲೆಗಳೊಂದಿಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಬೇಕು\n"
                    "2. ನೋಂದಣಿ ಪ್ರಾಧಿಕಾರ (ಕಾರ್ಮಿಕ ನಿರೀಕ್ಷಕ/ಹಿರಿಯ ಕಾರ್ಮಿಕ ನಿರೀಕ್ಷಕ) ಪರಿಶೀಲಿಸುತ್ತಾರೆ\n"
                    "3. ಉದ್ಯೋಗ ಪ್ರಮಾಣಪತ್ರ ಮತ್ತು ಅರ್ಹತೆಯ ಪರಿಶೀಲನೆ\n"
                    "4. ಸಕ್ಷಮ ಪ್ರಾಧಿಕಾರದಿಂದ ಅನುಮೋದನೆ/ತಿರಸ್ಕರಣೆ\n"
                    "5. ಫಲಾನುಭವಿ ನೋಂದಣಿ (ಲೇಬರ್ ಕಾರ್ಡ್) ವಿತರಣೆ\n\n"
                    "ಕಾರ್ಮಿಕರು ಅರ್ಹರಾಗಿದ್ದರೆ ಮತ್ತು ಎಲ್ಲಾ ಅಗತ್ಯ ದಾಖಲೆಗಳನ್ನು ಹೊಂದಿದ್ದರೆ, ದಯವಿಟ್ಟು ಲಾಗಿನ್ ಆಗಿ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ.")
        else:
            return ("## Scheme: Worker Registration\n\n"
                    "### Registration Overview\n"
                    "Registration under the Karnataka Building and Other Construction Workers Welfare Board enables eligible construction workers to avail welfare benefits under the BOCW Act.\n\n"
                    "### Eligibility Criteria\n"
                    "- The applicant must satisfy the eligibility criteria prescribed under the BOCW Act.\n"
                    "- Must have worked for a minimum of 90 days in building or other construction work during the preceding 12 months.\n"
                    "- Age must fall within the prescribed limit (generally 18-60 years).\n"
                    "- Must not be a member of any other Welfare Board.\n\n"
                    "### Required Documents\n"
                    "The applicant must submit the following documents:\n"
                    "- **Employment Certificate**: In Form V(A) / V(B) / V(C) / V(D). Issued by authorized employer / contractor / competent authority.\n"
                    "- **Aadhaar Card**: (Self-attested copy)\n"
                    "- **Ration Card**: (Non-Mandatory) For family details verification.\n"
                    "- **Age Proof**: (Any one of the following): Aadhaar Card, Voter ID Card, or Any Government-issued age proof document.\n\n"
                    "### Registration Details\n"
                    "- **Application Fee**: Rs.100 (Rupees One Hundred only)\n"
                    "- **Delivery Timeline**: 15 Working Days (subject to document verification and field validation)\n\n"
                    "### Procedure for Applying\n"
                    "1. The applicant submits the duly filled application along with required documents.\n"
                    "2. Application is scrutinized by the Registering Authority (Labour Inspector / Senior Labour Inspector).\n"
                    "3. Verification of employment certificate and eligibility.\n"
                    "4. Approval / Rejection by the competent authority.\n"
                    "5. Issuance of Beneficiary Registration (Labour Card).\n\n"
                    "### 20-A, Employment Certificate for Continuation\n"
                    "As per Section 14 of the BOCW Act, the beneficiary must submit every year:\n"
                    "- Pay slip (Non-Mandatory) OR Copy of nominal muster roll as proof of employment\n"
                    "- AND Employment Certificate in Form V(A) / V(B) / V(C) / V(D)\n\n"
                    "The documents must establish engagement in construction work for a minimum of 90 days in the preceding 12 months.\n\n"
                    "If the Labour is eligible and has all the required documents, please Login and submit the scheme application.\n"
                    "For new Labour, please Register and then apply for the scheme.")

    # RENEWAL — hardcoded response from ksk.md, no LLM
    if intent == "RENEWAL":
        if language == "kn":
            return ("## ಕಾರ್ಮಿಕ ನವೀಕರಣ\n\n"
                    "### ನವೀಕರಣ ಅವಲೋಕನ\n"
                    "ಮಂಡಳಿಯ ಅಡಿಯಲ್ಲಿ ಕಲ್ಯಾಣ ಸೌಲಭ್ಯಗಳನ್ನು ಮುಂದುವರಿಸಲು ನೋಂದಣಿ ನವೀಕರಣ ಅಗತ್ಯ.\n\n"
                    "### ನವೀಕರಣಕ್ಕೆ ಅರ್ಹತೆ\n"
                    "- 90 ದಿನಗಳ ಕೆಲಸದ ಅವಶ್ಯಕತೆ: ಕಳೆದ 12 ತಿಂಗಳಲ್ಲಿ ಕನಿಷ್ಠ 90 ದಿನಗಳ ಕೆಲಸ ಮಾಡಿರಬೇಕು\n"
                    "- ಮಾನ್ಯ 90 ದಿನಗಳ ಕೆಲಸದ ಪ್ರಮಾಣಪತ್ರ ಕಡ್ಡಾಯ\n"
                    "- ಪ್ರಮಾಣಪತ್ರವನ್ನು ಸಕ್ಷಮ ಪ್ರಾಧಿಕಾರ (ಬಿಲ್ಡರ್/ಗುತ್ತಿಗೆದಾರ/ಎಂಜಿನಿಯರ್/ಸ್ಥಳೀಯ ಪ್ರಾಧಿಕಾರ) ನೀಡಬೇಕು\n\n"
                    "### ನವೀಕರಣ ಯಾವಾಗ ಅರ್ಜಿ ಸಲ್ಲಿಸಬಹುದು\n"
                    "- ನೋಂದಣಿ ಅವಧಿ ಮುಗಿದ ನಂತರ ಮಾತ್ರ ನವೀಕರಣಕ್ಕೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಬಹುದು\n"
                    "- ಬಫರ್ ಅವಧಿ: ಅವಧಿ ಮುಗಿದ ದಿನಾಂಕದಿಂದ 365 ದಿನಗಳು (1 ವರ್ಷ)\n"
                    "- 365 ದಿನಗಳ ಒಳಗೆ ನವೀಕರಣಕ್ಕೆ ಅರ್ಜಿ ಸಲ್ಲಿಸದಿದ್ದರೆ, ಹೊಸ ನೋಂದಣಿಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಬೇಕು\n\n"
                    "ಕಾರ್ಮಿಕರು ಅರ್ಹರಾಗಿದ್ದರೆ ಮತ್ತು ಎಲ್ಲಾ ಅಗತ್ಯ ದಾಖಲೆಗಳನ್ನು ಹೊಂದಿದ್ದರೆ, ದಯವಿಟ್ಟು ಲಾಗಿನ್ ಆಗಿ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ.")
        else:
            return ("## Scheme: Worker Renewal\n\n"
                    "### Renewal Overview\n"
                    "Renewal of registration is required to continue availing welfare benefits under the Board.\n\n"
                    "### Eligibility for Renewal\n"
                    "A registered construction worker can apply for Renewal subject to the following conditions:\n"
                    "- **90 Days Work Requirement:** The worker must have worked at least 90 days in the last 12 months.\n"
                    "- A valid 90 Days Work Certificate is mandatory.\n"
                    "- The certificate must be issued by a competent authority (Builder / Contractor / Engineer / Local Authority).\n"
                    "- Without a valid 90 Days Work Certificate, renewal cannot be approved.\n\n"
                    "### When Can Renewal Be Applied\n"
                    "**After Registration Expiry:**\n"
                    "- Renewal can be applied only after the registration has expired.\n"
                    "- Renewal is applicable for previously registered (Active) workers whose registration validity has lapsed.\n\n"
                    "**Buffer Period (Grace Period):**\n"
                    "- After the registration expiry, the worker enters a buffer period of 365 days (1 year) from the date of expiry.\n"
                    "- During this buffer period, the worker is eligible to apply for Renewal.\n\n"
                    "### Important Rule\n"
                    "- If the worker does not apply for renewal within 365 days from the expiry date, Renewal is not permitted.\n"
                    "- The worker must apply for New Registration again.\n\n"
                    "### Please Check\n"
                    "- Can only apply after Expiry date.\n"
                    "- Buffer period calculation (Expiry Date + 365 days).\n"
                    "- 90 Days Work Certificate (last 12 months).\n"
                    "- Check Duplicate active membership.\n"
                    "- Aadhaar-based authentication.\n\n"
                    "If the Labour is eligible and has all the required documents, please Login and submit the scheme application.\n"
                    "For new Labour, please Register and then apply for the scheme.")

    # SCHEMES_LIST — hardcoded list of all schemes, no LLM
    if intent == "SCHEMES_LIST":
        if language == "kn":
            return ("## KBOCWWB ಅಡಿಯಲ್ಲಿ ಲಭ್ಯವಿರುವ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು\n\n"
                    "1. **ಅಪಘಾತ ಸೌಲಭ್ಯ** - ಮರಣಕ್ಕೆ ₹8 ಲಕ್ಷ, ಶಾಶ್ವತ ಅಂಗವಿಕಲತೆಗೆ ₹2 ಲಕ್ಷ\n"
                    "2. **ಪ್ರಮುಖ ಕಾಯಿಲೆಗಳಿಗೆ ಸಹಾಯ (ಕಾರ್ಮಿಕ ಚಿಕಿತ್ಸಾ ಭಾಗ್ಯ)** - ₹2,00,000 ವರೆಗೆ\n"
                    "3. **ತಾಯಿ ಮಗು ಸಹಾಯ ಹಸ್ತ** - ₹6,000 (₹500/ತಿಂಗಳು)\n"
                    "4. **ಹೆರಿಗೆ ಸಹಾಯ (ತಾಯಿ ಲಕ್ಷ್ಮೀ ಬಾಂಡ್)** - ₹50,000 ಪ್ರತಿ ಹೆರಿಗೆಗೆ\n"
                    "5. **ವೈದ್ಯಕೀಯ ಸಹಾಯ** - ₹300/ದಿನ, ಗರಿಷ್ಠ ₹20,000\n"
                    "6. **ಅಂಗವಿಕಲ ಪಿಂಚಣಿ** - ₹2,000/ತಿಂಗಳು + ₹2,00,000 ವರೆಗೆ\n"
                    "7. **ಅಂಗವಿಕಲ ಪಿಂಚಣಿ ಮುಂದುವರಿಕೆ** - ವಾರ್ಷಿಕ ಜೀವಂತ ಪ್ರಮಾಣಪತ್ರ ಅಗತ್ಯ\n"
                    "8. **ಪಿಂಚಣಿ (ವೃದ್ಧಾಪ್ಯ)** - ₹3,000/ತಿಂಗಳು ವರೆಗೆ\n"
                    "9. **ಪಿಂಚಣಿ ಮುಂದುವರಿಕೆ** - ವಾರ್ಷಿಕ ಜೀವಂತ ಪ್ರಮಾಣಪತ್ರ ಅಗತ್ಯ\n"
                    "10. **ಅಂತ್ಯಕ್ರಿಯೆ ಮತ್ತು ಸಹಾಯಧನ** - ₹1,46,000\n"
                    "11. **ಮದುವೆ ಸಹಾಯ** - ₹60,000\n\n"
                    "ಯಾವುದೇ ನಿರ್ದಿಷ್ಟ ಯೋಜನೆಯ ಬಗ್ಗೆ ವಿವರಗಳಿಗಾಗಿ, ದಯವಿಟ್ಟು ಅದರ ಹೆಸರಿನಿಂದ ಕೇಳಿ.")
        else:
            return ("## Available Welfare Schemes under KBOCWWB\n\n"
                    "The following welfare schemes are available for registered construction workers:\n\n"
                    "1. **Accident Benefits** - Up to Rs.8 Lakh for death, Rs.2 Lakh for permanent total disablement, Rs.1 Lakh for partial disablement\n"
                    "2. **Assistance for Major Ailments (Karmika Chikitsa Bhagya)** - Up to Rs.2,00,000 for treatment of major ailments\n"
                    "3. **Thayi Magu Sahaya Hasta (Nutritional Support)** - Rs.6,000 (Rs.500/month) for pre-school education and nutritional support\n"
                    "4. **Delivery Assistance (Tayi Lakshmi Bond)** - Rs.50,000 per delivery (first two living children only)\n"
                    "5. **Medical Assistance** - Rs.300 per day of hospitalization, maximum Rs.20,000 (minimum 48 hours required)\n"
                    "6. **Disability Pension and Ex-Gratia** - Rs.2,000/month pension + up to Rs.2,00,000 ex-gratia\n"
                    "7. **Continuation of Disability Pension** - Annual Living Certificate required\n"
                    "8. **Pension (Old Age Pension)** - Up to Rs.3,000/month for workers who completed 60 years\n"
                    "9. **Continuation of Pension** - Annual continuation, Living Certificate required every December\n"
                    "10. **Funeral and Ex-Gratia** - Rs.1,46,000 for funeral expenses and ex-gratia to nominee\n"
                    "11. **Marriage Assistance** - Rs.60,000 for marriage of worker or dependent children (maximum twice per family)\n\n"
                    "**Note:** Most schemes require a valid 90 Days Work Certificate and active registration.\n"
                    "For detailed information about any specific scheme, please ask about it by name.\n\n"
                    "If the Labour is eligible and has all the required documents, please Login and submit the scheme application.\n"
                    "For new Labour, please Register and then apply for the scheme.")

    # STATUS_CHECK — LLM grounded on pre-fetched user data + conversation history
    if intent == "STATUS_CHECK":
        print(f"[DEBUG] answer() STATUS_CHECK path")
        print(f"[DEBUG]   prefetched_user_data is None: {prefetched_user_data is None}")
        if not prefetched_user_data:
            print(f"[DEBUG]   ERROR: No user data for STATUS_CHECK!")
            logger.error("STATUS_CHECK intent but no prefetched user data")
            return "Unable to fetch your information at this time. Please try again later."
        print(f"[DEBUG]   user_data keys: {list(prefetched_user_data.keys())}")
        print(f"[DEBUG]   FULL USER DATA JSON:")
        print(json.dumps(prefetched_user_data, indent=2, ensure_ascii=False, default=str))
        print(f"[DEBUG]   Building status prompt...")
        system_prompt = _build_status_prompt(prefetched_user_data, language)
        print(f"[DEBUG]   Status prompt length: {len(system_prompt)} chars")
        print(f"[DEBUG]   First 500 chars of prompt: {system_prompt[:500]}")
        truncated_history = None
        if history:
            truncated_history = history[-settings.AUTHENTICATED_HISTORY_MESSAGES:]
            logger.debug(
                "STATUS_CHECK: using %d of %d history messages",
                len(truncated_history), len(history),
            )
        result = await ollama.chat(
            system_prompt=system_prompt,
            user_message=_prepare_user_message(question, language),
            history=truncated_history,
        )
        print(f"[DEBUG]   LLM result (first 200): {result[:200]}")
        return _cap_answer_length(result)


    # GENERAL — detect topic and use focused context for LLM (hybrid AI).
    # Topic detection from message + conversation history.
    topic_context = _get_topic_context(question, history=history)

    # Then try direct section match (exact ksk.md content, no LLM).
    direct_section = _find_direct_section(question)
    if direct_section:
        print(f"[DEBUG]   Direct section match found, returning ksk.md content directly (no LLM)")
        footer = ("\n\nIf the Labour is eligible and has all the required documents, please Login and submit "
                  "the scheme application. For new Labour, please Register and then apply for the scheme.")
        return _cap_answer_length(direct_section + footer)

    # Use focused topic context (~2KB) if available, otherwise full knowledge base (33KB)
    context = topic_context if topic_context else _FULL_KNOWLEDGE_BASE
    print(f"[DEBUG]   Using {'focused topic' if topic_context else 'full knowledge base'} context ({len(context)} chars)")

    if prefetched_user_data:
        # Authenticated GENERAL: RAG context + user data
        system_prompt = _build_authenticated_general_prompt(
            context, prefetched_user_data, language,
        )
        max_hist = settings.AUTHENTICATED_HISTORY_MESSAGES
    else:
        # Unauthenticated GENERAL: RAG context only
        system_prompt = _build_system_prompt(context, language)
        max_hist = settings.MAX_HISTORY_MESSAGES

    truncated_history = None
    if history:
        truncated_history = history[-max_hist:]
        logger.debug(
            "GENERAL: using %d of %d history messages (max=%d)",
            len(truncated_history), len(history), max_hist,
        )

    result = await ollama.chat(
        system_prompt=system_prompt,
        user_message=_prepare_user_message(question, language),
        history=truncated_history,
    )
    return _cap_answer_length(result)


# ---------------------------------------------------------------------------
# Answer (streaming)
# ---------------------------------------------------------------------------
async def answer_stream(
    question: str,
    qdrant: Optional[AsyncQdrantClient] = None,
    ollama: Optional[OllamaClient] = None,
    history: Optional[list[dict]] = None,
    language: str = "",
    intent: str = "GENERAL",
    prefetched_user_data: Optional[dict] = None,
) -> AsyncIterator[str]:
    """Streaming variant — yields text chunks as they arrive from Ollama.

    The intent and prefetched_user_data are computed by classify_and_prepare()
    which runs under the thread lock.
    """
    logger.info(
        "answer_stream() question=%s language=%s intent=%s has_user_data=%s",
        question[:100], language or "en", intent, prefetched_user_data is not None,
    )
    ollama = ollama or default_ollama

    # LOGIN_REQUIRED — exact constant string, language-independent, no LLM
    if intent == "LOGIN_REQUIRED":
        yield LOGIN_REQUIRED_RESPONSE
        return

    # ECARD — exact constant string, language-independent, no LLM
    if intent == "ECARD":
        yield ECARD_RESPONSE
        return

    # OUT_OF_SCOPE — randomized hardcoded response, bypassing generative LLM
    if intent == "OUT_OF_SCOPE":
        if language == "kn":
            yield random.choice([
                "ಅದು ತುಂಬಾ ಆಸಕ್ತಿದಾಯಕ ಪ್ರಶ್ನೆ! ಆದರೆ ಶ್ರಮ ಸಹಾಯಕನಾಗಿ, ನಾನು ಕೇವಲ ಕಟ್ಟಡ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು ಮತ್ತು ಕಾರ್ಮಿಕ ಸೇವಾ ಕೇಂದ್ರದ (KSK) ಸೇವೆಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಮಾಹಿತಿ ನೀಡಬಲ್ಲೆ. ನಾನು ನಿಮಗೆ ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಹುದೇ?",
                "ಕ್ಷಮಿಸಿ, ಆ ವಿಷಯದ ಬಗ್ಗೆ ನನ್ನ ಬಳಿ ಮಾಹಿತಿ ಇಲ್ಲ. ನನ್ನ ಪರಿಣತಿ ಕೇವಲ KBOCWWB ಯೋಜನೆಗಳು ಮತ್ತು ಅರ್ಜಿಗಳ ಬಗ್ಗೆ ಸಹಾಯಲು ಸೀಮಿತವಾಗಿದೆ.",
                "ನಾನು ಆ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ, ಆದರೆ ನಿಮ್ಮ ಯೋಜನೆಗಳ ಅರ್ಹತೆ ಅಥವಾ ಅರ್ಜಿ ಸ್ಥಿತಿಯನ್ನು ಪರಿಶೀಲಿಸಲು ನಾನು ಖಂಡಿತ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ!"
            ])
        else:
            yield random.choice([
                "That's an interesting question! However, as Shrama Sahayak, I'm specialized only in construction worker welfare schemes and KSK services.",
                "I'm afraid that's outside my area of expertise. I am strictly here to assist you with KBOCWWB schemes and KSK services.",
                "I don't have information on that topic. My focus is entirely on helping construction workers with their welfare benefits."
            ])
        return

    # GREETING — fast hardcoded response, no RAG needed
    if intent == "GREETING":
        if language == "kn":
            yield ("ನಮಸ್ಕಾರ! ನಾನು ಶ್ರಮ ಸಹಾಯಕ — ಕರ್ನಾಟಕ ಕಟ್ಟಡ ಮತ್ತು ಇತರ ನಿರ್ಮಾಣ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಮಂಡಳಿಯ (KBOCWWB) ಡಿಜಿಟಲ್ ಸಹಾಯಕ.\n\n"
                    "ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು:\n"
                    "- ನೋಂದಣಿ ಮತ್ತು ನವೀಕರಣ ಮಾಹಿತಿ\n"
                    "- ಕಲ್ಯಾಣ ಯೋಜನೆಗಳ ವಿವರಗಳು\n"
                    "- ಅರ್ಜಿ ಸ್ಥಿತಿ ಪರಿಶೀಲನೆ\n"
                    "- ಅಗತ್ಯ ದಾಖಲೆಗಳ ಮಾಹಿತಿ\n\n"
                    "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ!")
        else:
            yield ("Namaskara! I am Shrama Sahayak, your digital assistant from the Karnataka Building & "
                    "Other Construction Workers Welfare Board (KBOCWWB).\n\n"
                    "I can help you with:\n"
                    "- Registration and renewal information\n"
                    "- Welfare scheme details and eligibility\n"
                    "- Application status checks\n"
                    "- Required documents for schemes\n\n"
                    "How can I assist you today?")
        return

    # REGISTRATION, RENEWAL, SCHEMES_LIST — reuse answer() responses
    if intent in ("REGISTRATION", "RENEWAL", "SCHEMES_LIST"):
        # Call answer() which has the full hardcoded responses
        result = await answer(
            question=question, qdrant=qdrant, ollama=ollama,
            history=history, language=language, intent=intent,
            prefetched_user_data=prefetched_user_data,
        )
        yield result
        return

    # STATUS_CHECK — LLM grounded on pre-fetched user data + conversation history
    if intent == "STATUS_CHECK":
        if not prefetched_user_data:
            yield "Unable to fetch your information at this time. Please try again later."
            return
        system_prompt = _build_status_prompt(prefetched_user_data, language)
        truncated_history = None
        if history:
            truncated_history = history[-settings.AUTHENTICATED_HISTORY_MESSAGES:]
        async for chunk in ollama.chat_stream(
            system_prompt=system_prompt,
            user_message=_prepare_user_message(question, language),
            history=truncated_history,
        ):
            yield chunk
        return


    # GENERAL — detect topic and use focused context for LLM (hybrid AI).
    topic_context = _get_topic_context(question, history=history)

    # Then try direct section match (exact ksk.md content, no LLM).
    direct_section = _find_direct_section(question)
    if direct_section:
        print(f"[DEBUG]   Direct section match found, returning ksk.md content directly (no LLM)")
        footer = ("\n\nIf the Labour is eligible and has all the required documents, please Login and submit "
                  "the scheme application. For new Labour, please Register and then apply for the scheme.")
        yield direct_section + footer
        return

    # Use focused topic context (~2KB) if available, otherwise full knowledge base (33KB)
    context = topic_context if topic_context else _FULL_KNOWLEDGE_BASE
    print(f"[DEBUG]   Using {'focused topic' if topic_context else 'full knowledge base'} context ({len(context)} chars)")

    if prefetched_user_data:
        # Authenticated GENERAL: RAG context + user data
        system_prompt = _build_authenticated_general_prompt(
            context, prefetched_user_data, language,
        )
        max_hist = settings.AUTHENTICATED_HISTORY_MESSAGES
    else:
        # Unauthenticated GENERAL: RAG context only
        system_prompt = _build_system_prompt(context, language)
        max_hist = settings.MAX_HISTORY_MESSAGES

    truncated_history = None
    if history:
        truncated_history = history[-max_hist:]

    async for chunk in ollama.chat_stream(
        system_prompt=system_prompt,
        user_message=_prepare_user_message(question, language),
        history=truncated_history,
    ):
        yield chunk
