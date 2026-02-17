import json
import logging
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
# Exact response constants — returned directly by Python, never by the LLM.
# These are language-independent: regardless of what language the user
# selects, ECARD and LOGIN_REQUIRED always return these exact strings.
# ---------------------------------------------------------------------------
LOGIN_REQUIRED_RESPONSE = "<<LOGIN_MODAL_REQUIRED>>"
ECARD_RESPONSE = "ECARD"

# Valid intent labels (used for parsing LLM output)
_VALID_INTENTS = frozenset({"ECARD", "STATUS_CHECK", "GENERAL"})

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
    # English
    "check status", "application status", "scheme status", "status check",
    "my status", "registration status", "renewal status",
    "my application", "my registration", "my renewal",
    "is my application", "was my application", "has my application",
    "is it approved", "was it approved", "is it rejected",
    # Kannada
    "ಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ", "ಅರ್ಜಿ ಸ್ಥಿತಿ", "ಯೋಜನೆ ಸ್ಥಿತಿ",
    "ನನ್ನ ಅರ್ಜಿ", "ನನ್ನ ನೋಂದಣಿ", "ನನ್ನ ಸ್ಥಿತಿ",
    "ಅನುಮೋದಿಸಲಾಗಿದೆಯೇ", "ನವೀಕರಣ ಸ್ಥಿತಿ",
    "ಅನುಮೋದನೆ", "ತಿರಸ್ಕರಿಸಲಾಗಿದೆ", "ಬಾಕಿ ಇದೆ",
]

# Pre-normalize keywords at import time for consistent matching
_ECARD_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _ECARD_KEYWORDS
]
_STATUS_CHECK_KEYWORDS_NORM: list[str] = [
    unicodedata.normalize("NFC", kw) for kw in _STATUS_CHECK_KEYWORDS
]


def _keyword_intent(message: str) -> str | None:
    """Layer 1: fast keyword match. Returns intent or None.

    Case-insensitive substring matching with Unicode normalization.
    Kannada keywords are matched as-is (Kannada has no case distinction).
    NFC normalization handles invisible Unicode characters (ZWNJ, ZWJ variants)
    that can differ between input methods.
    """
    msg_normalized = unicodedata.normalize("NFC", message.lower())
    # Check ECARD first — more specific keywords, less likely to false-positive
    if any(kw in msg_normalized for kw in _ECARD_KEYWORDS_NORM):
        return "ECARD"
    if any(kw in msg_normalized for kw in _STATUS_CHECK_KEYWORDS_NORM):
        return "STATUS_CHECK"
    return None


# ---------------------------------------------------------------------------
# Layer 2: LLM-based intent classification (authenticated users only)
# ---------------------------------------------------------------------------
INTENT_CLASSIFICATION_PROMPT = """\
You are an intent classifier. Classify the message into one of: ECARD, STATUS_CHECK, GENERAL

ECARD: User wants their e-card, labour card, ID card — to view, download, or print it.
STATUS_CHECK: User wants to know the status of their application, scheme, registration, or renewal — approved, rejected, pending.
GENERAL: Everything else — questions about schemes, eligibility, documents, how to apply, greetings, etc.

Examples:
- "download my ecard" → ECARD
- "ನನ್ನ ಕಾರ್ಡ್ ತೋರಿಸಿ" → ECARD
- "ಇ-ಕಾರ್ಡ್ ಡೌನ್\u200cಲೋಡ್" → ECARD
- "show me my labour card" → ECARD
- "ಕಾರ್ಡ್ ಪ್ರಿಂಟ್ ಮಾಡಿ" → ECARD
- "what is my application status" → STATUS_CHECK
- "ನನ್ನ ಅರ್ಜಿ ಸ್ಥಿತಿ ಏನು" → STATUS_CHECK
- "ನನ್ನ ನೋಂದಣಿ ಅನುಮೋದಿಸಲಾಗಿದೆಯೇ?" → STATUS_CHECK
- "is my renewal approved?" → STATUS_CHECK
- "ನನ್ನ ಯೋಜನೆ ಅನುಮೋದನೆ ಆಗಿದೆಯಾ?" → STATUS_CHECK
- "what schemes can I apply for" → GENERAL
- "ಯಾವ ಯೋಜನೆಗಳು ಲಭ್ಯವಿದೆ?" → GENERAL
- "how to register" → GENERAL
- "ನೋಂದಣಿ ಹೇಗೆ ಮಾಡುವುದು" → GENERAL
- "hello" → GENERAL

Respond with exactly one word: ECARD, STATUS_CHECK, or GENERAL"""


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

    # Unauthenticated: keywords only, no LLM classification → GENERAL
    if not is_authenticated:
        logger.debug("Unauthenticated user, no keyword match → GENERAL")
        return "GENERAL"

    # Layer 2: LLM classification (authenticated users only)
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
            "\nIMPORTANT — LANGUAGE INSTRUCTION:\n"
            "You MUST respond ENTIRELY in Kannada (ಕನ್ನಡ). Every single word of your "
            "response must be in the Kannada script.\n"
            "- Write naturally and fluently in Kannada as a native speaker would.\n"
            "- Do NOT transliterate English into Kannada script. Use proper Kannada words "
            "and grammar.\n"
            "- Keep rupee amounts as numerals (e.g., ₹2,00,000).\n"
            "- Keep official scheme names in English if they are commonly known that way, "
            "but explain them in Kannada.\n"
            "- Refer to yourself as ಶ್ರಮ ಸಹಾಯಕ (Shrama Sahayak) ONLY if you are introducing yourself for the first time or if the user asks.\n"
            "- If the user has already greeted you, do NOT repeat your name or say 'Namaskara' again.\n"
            "- If you don't have enough information, say: "
            '"ಈ ವಿಷಯದ ಬಗ್ಗೆ ನನ್ನ ಬಳಿ ಸಂಪೂರ್ಣ ಮಾಹಿತಿ ಇಲ್ಲ. ನೀವು KBOCWWB ವೆಬ್ ಪೋರ್ಟಲ್ '
            'ಅಥವಾ ಮೊಬೈಲ್ ಆ್ಯಪ್ ಮೂಲಕ ಆನ್\u200cಲೈನ್\u200cನಲ್ಲಿ ವಿಚಾರಿಸಬಹುದು, ಅಥವಾ ನಿಮ್ಮ '
            'ಹತ್ತಿರದ ಕಾರ್ಮಿಕ ಸೇವಾ ಕೇಂದ್ರಕ್ಕೆ ಭೇಟಿ ನೀಡಿ."\n'
            "- If the question is off-topic, say: "
            '"ನಾನು ಶ್ರಮ ಸಹಾಯಕ — ಕಟ್ಟಡ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು ಮತ್ತು KSK ಸೇವೆಗಳ '
            'ಬಗ್ಗೆ ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಅದಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆ ಕೇಳಿ."\n'
        )
    else:
        prompt += (
            f"\nIMPORTANT: Respond entirely in {lang_name}. Translate all information "
            f"naturally into {lang_name} while keeping rupee amounts as numerals and "
            f"proper nouns as-is.\n"
        )
    return prompt


def _prepare_user_message(message: str, language: str) -> str:
    """Appends an explicit language instruction to the user message."""
    if language == "kn":
        return message + "\n\n(IMPORTANT: Answer entirely in Kannada (Kannada script) / ದಯವಿಟ್ಟು ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ)"
    return message


# ---------------------------------------------------------------------------
# System prompt — the heart of the "ChatGPT-like" experience
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ) — a dedicated digital helper for working people. \
You serve the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** \
and its service centers, the **Karmika Seva Kendras (KSK)**.

Your identity:
- You are Shrama Sahayak — "the helper of working people."
- Introduce yourself by this name ONLY if this is the start of the conversation or if the user asks who you are.
- Do NOT repeat your name or say "Namaskara" in every message.
- You are genuinely caring, respectful, and knowledgeable. Treat every user like a valued guest.
- Speak in clear, simple language. Many users may not be highly educated, so avoid jargon.
- Be patient, encouraging, and supportive. These workers deserve dignity and helpful guidance.
- Always end on a positive, actionable note.

How to answer:
1. Use ONLY the information provided in the Context below. Do not guess or invent facts.
2. Structure your responses for easy scanning:
   - Use **bold** for key terms, scheme names, and amounts.
   - For multi-step processes (like how to apply), use numbered lists (1, 2, 3...).
   - For lists of documents, benefits, or schemes, use bullet points.
   - When explaining eligibility, use a clear "**Who can apply:**" section.
3. Include specific amounts (in ₹ with Indian formatting, e.g., ₹2,00,000) and deadlines \
whenever the context provides them.
4. If the context does not contain enough information to fully answer, say warmly: \
"I don't have complete information on that topic right now. You can:\n\
- Apply or enquire **online** through the KBOCWWB web portal or mobile app\n\
- Visit your nearest **Karmika Seva Kendra (KSK)** for in-person assistance\n\
- Call the helpline for guidance\n\
They'll be happy to help you!"
5. If the question is outside the scope of KSK/KBOCWWB (for example, unrelated topics like \
cooking, sports, politics), gently guide the user back: \
"That's an interesting question! However, I'm Shrama Sahayak — specially trained to help \
with construction worker welfare schemes and KSK services. Here are some things I can \
help you with:\n\
- How to apply for welfare schemes online or at a KSK center?\n\
- How to register as a construction worker under KBOCWWB?\n\
- What documents are needed for registration or scheme applications?\n\
- What benefits and financial assistance are available?\n\
- How to check your application or renewal status?\n\
Feel free to ask me any of these!"

Response quality guidelines:
- Start with a warm, direct answer to the question. Do NOT greet unless it's the first message.
- Follow up with relevant details (eligibility, documents needed, amounts, deadlines).
- Always provide a clear **Next Steps** section at the end with actionable guidance \
(e.g., "You can apply online through the KBOCWWB web portal or mobile app, or visit \
your nearest KSK center where staff will guide you through the process!").
- Be comprehensive but scannable — users should quickly find the information they need.
- Keep responses well-organized. Do not pad with unnecessary filler.

Context:
{context}
"""

# ---------------------------------------------------------------------------
# Authenticated GENERAL prompt — includes user data as supplementary context
# ---------------------------------------------------------------------------
AUTHENTICATED_GENERAL_PROMPT = """\
You are **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ) — a dedicated digital helper for working people. \
You serve the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** \
and its service centers, the **Karmika Seva Kendras (KSK)**. \
The user is logged in and you have access to their personal data.

Your identity:
- You are Shrama Sahayak — "the helper of working people."
- If the user's name is available in the User Data below, address them by name to make \
the conversation personal and warm (e.g., "Namaskara Ramesh ji!" or "Ramesh avare" in Kannada) \
ONLY at the start of the conversation. Do NOT repeat this in every message.
- You are genuinely caring, respectful, and knowledgeable. This person has trusted you \
with their personal information — honour that trust.
- Speak in clear, simple language. Many users may not be highly educated, so avoid jargon.
- Be patient, encouraging, and supportive. These workers deserve dignity and helpful guidance.
- Always end on a positive, actionable note.

How to answer:
1. Use ONLY the information provided in the Context and User Data below. Do not guess or invent facts.
2. Structure your responses for easy scanning:
   - Use **bold** for key terms, scheme names, and amounts.
   - For multi-step processes (like how to apply), use numbered lists (1, 2, 3...).
   - For lists of documents, benefits, or schemes, use bullet points.
   - When explaining eligibility, use a clear "**Who can apply:**" section.
3. Include specific amounts (in ₹ with Indian formatting, e.g., ₹2,00,000) and deadlines \
whenever the context provides them.
4. When answering, personalize the response using the user's data where relevant \
(e.g., "Based on your registration, you are eligible for..." or "Your renewal is due on...").
5. If the context does not contain enough information to fully answer, say warmly: \
"I don't have complete information on that topic right now. You can:\n\
- Apply or enquire **online** through the KBOCWWB web portal or mobile app\n\
- Visit your nearest **Karmika Seva Kendra (KSK)** for in-person assistance\n\
- Call the helpline for guidance\n\
They'll be happy to help you!"
6. If the question is outside the scope of KSK/KBOCWWB (for example, unrelated topics like \
cooking, sports, politics), gently guide the user back: \
"That's an interesting question! However, I'm your Shrama Sahayak — specially trained to \
help with construction worker welfare schemes and KSK services. Here are some things I can \
help you with:\n\
- What welfare schemes are available for you?\n\
- How to apply for schemes online or at a KSK center?\n\
- How to check your application or renewal status?\n\
- What benefits and financial assistance are you eligible for?\n\
Feel free to ask me any of these!"

Response quality guidelines:
- Start with a direct answer. Use a personalized greeting ONLY if it is the first message.
- Follow up with relevant details (eligibility, documents needed, amounts, deadlines).
- Always provide a clear **Next Steps** section at the end with actionable guidance \
(e.g., "You can apply online through the KBOCWWB web portal or mobile app, or visit \
your nearest KSK center where staff will guide you through the process!").
- Be comprehensive but scannable — users should quickly find the information they need.
- Keep responses well-organized. Do not pad with unnecessary filler.

Context:
{context}

User's Personal Data (from KBOCWWB system — use ONLY if relevant to their question):
{user_data}
"""


def _build_system_prompt(context: str, language: str = "") -> str:
    """Assemble the full system prompt with retrieved context and optional language."""
    prompt = SYSTEM_PROMPT.format(context=context)
    return _append_language_instruction(prompt, language)


def _build_authenticated_general_prompt(
    context: str, user_data: dict, language: str = ""
) -> str:
    """Assemble system prompt with RAG context + user data for authenticated GENERAL."""
    user_data_str = json.dumps(user_data, indent=2, ensure_ascii=False, default=str)
    prompt = AUTHENTICATED_GENERAL_PROMPT.format(context=context, user_data=user_data_str)
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
- Address the user by name if available (e.g., "Namaskara Ramesh ji!") ONLY at the start of the conversation.
- Do NOT repeat your name or greeting in every message.
- This person trusts you with their personal data — honour that trust with accuracy and care.
- Be patient, encouraging, and supportive — especially if their application was rejected \
or is pending. Offer hope and suggest concrete next steps.
- Always end on a positive, actionable note.

How to answer:
1. Use ONLY the user data provided below. Do NOT fabricate any information.
2. Structure your responses for easy scanning:
   - Use **bold** for key terms, status values, scheme names, and amounts.
   - Use bullet points for lists. Include ₹ amounts and dates where available.
   - Present application statuses clearly (e.g., "**Status:** Approved" or "**Status:** Pending").
3. If a section says "No schemes applied" or data is missing, say so honestly but kindly, \
and suggest schemes they might be eligible for.
4. If their application was **rejected**, mention the reason (if available) and encourage them: \
"Don't worry — you can reapply. You can do this **online** through the KBOCWWB web portal \
or mobile app, or visit your nearest **Karmika Seva Kendra (KSK)** where staff will guide \
you through the reapplication process."
5. If their application is **approved**, congratulate them warmly and let them know next steps \
for receiving the benefit.
6. If their application is **pending**, reassure them that it is being processed and suggest \
they can check back or track the status online through the web portal or mobile app.
7. If the question is outside the scope of KSK/KBOCWWB, gently guide them back: \
"I'm your Shrama Sahayak, here to help with your KBOCWWB welfare information. Here are \
some things I can help you with:\n\
- Check your scheme application status\n\
- View your registration and renewal details\n\
- Understand what benefits you're eligible for\n\
- How to apply for new schemes online or at a KSK center\n\
Would you like to know about any of these?"

User Data:
{user_data}
"""


def _build_status_prompt(user_data: dict, language: str = "") -> str:
    """Build a system prompt grounded on the user's fetched data."""
    user_data_str = json.dumps(user_data, indent=2, ensure_ascii=False, default=str)
    prompt = STATUS_SYSTEM_PROMPT.format(user_data=user_data_str)
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
    """Return cached user data for this thread, or fetch from external API and cache.

    This function is designed to be called under the thread lock to prevent
    concurrent fetches for the same thread. The returned dict is the exact
    data that was fetched (or cached) — it is NOT re-read from DB after write.
    """
    # 1. Check cache first
    cached = await get_cached_user_data(db, thread_id, user_id)
    if cached is not None:
        logger.info("Using cached user data for thread %s, user %s", thread_id, user_id)
        return cached

    # 2. Cache miss — fetch from external API
    logger.info("Fetching user data from external API for user %s", user_id)
    try:
        data = await fetch_user_data(http_client, user_id, auth_token)
    except Exception:
        logger.error("External API fetch failed for user %s", user_id, exc_info=True)
        return None

    if not data:
        logger.warning("External API returned empty data for user %s", user_id)
        return None

    # 3. Save to DB for future cache hits
    try:
        await save_user_data(db, thread_id, user_id, data)
        logger.info("User data cached for thread %s, user %s", thread_id, user_id)
    except Exception:
        logger.error(
            "Failed to cache user data for thread %s, user %s — using fetched data anyway",
            thread_id, user_id, exc_info=True,
        )

    # 4. Return the fetched data directly (not a re-read from DB)
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
    """Classify intent and pre-fetch any required data.

    MUST be called under the thread lock to prevent race conditions on
    cache reads/writes.

    Classification strategy:
    - Unauthenticated: keyword matching only (no LLM call).
      Personal-intent keywords → LOGIN_REQUIRED; else → GENERAL.
    - Authenticated: keyword matching first, LLM fallback for ambiguous messages.
      Always fetches user data for non-ECARD intents.

    Returns:
        (intent, user_data_or_none) where intent is one of:
        "LOGIN_REQUIRED", "ECARD", "STATUS_CHECK", "GENERAL"
    """
    is_authenticated = bool(user_id and auth_token)

    # Classify intent (auth-aware: keywords only for unauth, keywords+LLM for auth)
    intent = await _classify_intent(ollama, message, is_authenticated=is_authenticated)
    logger.info(
        "classify_and_prepare: intent=%s authenticated=%s user=%s",
        intent, is_authenticated, user_id or "(anon)",
    )

    # Unauthenticated user with personal-info intent → login required
    if intent in ("ECARD", "STATUS_CHECK") and not is_authenticated:
        return "LOGIN_REQUIRED", None

    # Authenticated ECARD → exact string response, no data fetch needed
    if intent == "ECARD" and is_authenticated:
        return "ECARD", None

    # Authenticated user (STATUS_CHECK or GENERAL) → always fetch/cache user data
    user_data = None
    if is_authenticated:
        user_data = await _get_or_fetch_user_data(
            db, http_client, thread_id, user_id, auth_token,
        )

    return intent, user_data


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

    # STATUS_CHECK — LLM grounded on pre-fetched user data + conversation history
    if intent == "STATUS_CHECK":
        if not prefetched_user_data:
            logger.error("STATUS_CHECK intent but no prefetched user data")
            return "Unable to fetch your information at this time. Please try again later."
        system_prompt = _build_status_prompt(prefetched_user_data, language)
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
        return _cap_answer_length(result)

    # GENERAL — RAG pipeline, optionally enriched with user data
    context = await retrieve(question, qdrant=qdrant, ollama=ollama)

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

    # GENERAL — RAG pipeline, optionally enriched with user data
    context = await retrieve(question, qdrant=qdrant, ollama=ollama)

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
