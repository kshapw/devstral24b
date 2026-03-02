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
    "poem", "story", "write a", "code", "programming", "python",
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


def _keyword_intent(message: str) -> str | None:
    """Layer 1: fast keyword match. Returns intent or None.

    Case-insensitive substring matching with Unicode normalization.
    Kannada keywords are matched as-is (Kannada has no case distinction).
    NFC normalization handles invisible Unicode characters (ZWNJ, ZWJ variants)
    that can differ between input methods.
    """
    msg_normalized = unicodedata.normalize("NFC", message.lower()).strip()
    # Instant rejection for known out-of-scope topics
    if any(kw in msg_normalized for kw in _OUT_OF_SCOPE_KEYWORDS_NORM):
        return "OUT_OF_SCOPE"
    # Check ECARD first — more specific keywords, less likely to false-positive
    if any(kw in msg_normalized for kw in _ECARD_KEYWORDS_NORM):
        return "ECARD"
    if any(kw in msg_normalized for kw in _STATUS_CHECK_KEYWORDS_NORM):
        return "STATUS_CHECK"
    # Greetings — short messages that are just greetings, bypass RAG
    if any(kw == msg_normalized or msg_normalized.startswith(kw + " ") or msg_normalized.startswith(kw + ",") for kw in _GREETING_KEYWORDS_NORM):
        return "GREETING"
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
        "1. Answer this question using the information from the === REFERENCE CONTEXT === provided in the system prompt.\n"
        "2. Do NOT use your pre-trained knowledge to invent scheme names, benefit amounts, or document requirements that are NOT in the Context.\n"
        "3. If this question is about KBOCWWB topics (registration, renewal, schemes, welfare) BUT the Context has limited info, "
        "provide what you CAN from the Context and suggest visiting the KBOCWWB portal or nearest KSK for more details.\n"
        "4. ONLY refuse to answer if the question is about politicians, sports, weather, coding, or topics completely unrelated to KBOCWWB.]"
    )
    if language == "kn":
        return message + grounding + "\n\n(ಕಡ್ಡಾಯ: ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ. ಯಾವುದೇ ಇಂಗ್ಲಿಷ್ ಪದಗಳನ್ನು ಬಳಸಬೇಡಿ.)"
    return message + grounding


# ---------------------------------------------------------------------------
# System prompt — the heart of the "ChatGPT-like" experience
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ), a digital assistant for the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** and its **Karmika Seva Kendras (KSK)**.

=== REFERENCE CONTEXT (Your ONLY source of truth) ===
{context}
=== END OF CONTEXT ===

## ABSOLUTE RULES (violations are unacceptable):
1. **CONTEXT-ONLY**: Answer EXCLUSIVELY from the Context above. If information is NOT in the Context, say "I don't have complete information on that." DO NOT use your pre-trained knowledge to fill gaps.
2. **NO HALLUCINATION**: Do NOT invent schemes, amounts, documents, dates, or eligibility criteria. Every fact must be traceable to the Context. The ONLY schemes that exist are those explicitly named in the Context.
3. **SCHEME SEPARATION**: Each scheme has UNIQUE amounts, documents, and eligibility. NEVER mix details between schemes. Key distinctions:
   - "Delivery Assistance" (₹50,000) ≠ "Thayi Magu Sahaya Hasta" (₹6,000)
   - "Pension" ≠ "Continuation of Pension"
   - "Disability Pension" ≠ "Continuation of Disability Pension"
   - "Accident Assistance" ≠ "Funeral and Ex-Gratia"
4. **DOCUMENTS ARE SCHEME-SPECIFIC**: List ONLY documents from THAT scheme's section.
5. **OFF-TOPIC REJECTION**: For questions about politicians, sports, weather, coding, general knowledge — decline politely. Say: "I'm specialized only in KBOCWWB construction worker welfare schemes and KSK services."
6. **PAYMENT STATUS**: If asked about payment status, say: "Go to https://kbocwwb.karnataka.gov.in/ and check in Check DBT Application Status."

## Identity & Tone:
- You are Shrama Sahayak — caring, respectful, knowledgeable.
- Introduce yourself ONLY on the first message. Do NOT repeat greetings.
- Use clear, simple language. Avoid jargon.

## How to Structure Responses:
- Use **bold** for scheme names, amounts, and key terms.
- Use numbered lists for processes, bullet points for documents/eligibility.
- When answering about a scheme include: Overview, Benefits (₹ amounts), Eligibility, Required Documents, and Application Process — as found in the Context.
- If asking about "Registration" → use chunks labeled "Scheme: Worker Registration".
- If asking about "Renewal" → use chunks labeled "Scheme: Worker Renewal".
- End with: "If the Labour is eligible and has all the required documents, please Login and submit the scheme application. For new Labour, please Register and then apply for the scheme."

## If Context is Empty or Insufficient:
Say warmly: "I don't have complete information on that topic. You can enquire online through the KBOCWWB web portal or mobile app, or visit your nearest Karmika Seva Kendra (KSK) for in-person assistance."
"""

# ---------------------------------------------------------------------------
# Authenticated GENERAL prompt — includes user data as supplementary context
# ---------------------------------------------------------------------------
AUTHENTICATED_GENERAL_PROMPT = """\
You are **Shrama Sahayak** (ಶ್ರಮ ಸಹಾಯಕ), a digital assistant for the **Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB)** and its **Karmika Seva Kendras (KSK)**. The user is logged in.

=== REFERENCE CONTEXT (Your ONLY source of truth for scheme details) ===
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

    # GENERAL — RAG pipeline, optionally enriched with user data
    search_query = await _translate_for_search(ollama, question, language)
    context = await retrieve(search_query, qdrant=qdrant, ollama=ollama)

    # Empty context fallback — prevent hallucination when no relevant docs found
    if not context.strip():
        context = "[The retrieval system did not find highly relevant documents for this query. However, this appears to be a valid KBOCWWB-related question. Please answer based on whatever Context IS available in the system prompt, and if truly insufficient, suggest the user visit the KBOCWWB web portal (https://kbocwwb.karnataka.gov.in/) or their nearest Karmika Seva Kendra (KSK) for detailed assistance.]"

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
    search_query = await _translate_for_search(ollama, question, language)
    context = await retrieve(search_query, qdrant=qdrant, ollama=ollama)

    # Empty context fallback — prevent hallucination when no relevant docs found
    if not context.strip():
        context = "[The retrieval system did not find highly relevant documents for this query. However, this appears to be a valid KBOCWWB-related question. Please answer based on whatever Context IS available in the system prompt, and if truly insufficient, suggest the user visit the KBOCWWB web portal (https://kbocwwb.karnataka.gov.in/) or their nearest Karmika Seva Kendra (KSK) for detailed assistance.]"

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
