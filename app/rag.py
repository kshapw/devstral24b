import logging
from typing import AsyncIterator, Optional

from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.ollama_client import OllamaClient, default_ollama
from app.qdrant_service import get_qdrant_client

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# System prompt — the heart of the "ChatGPT-like" experience
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a friendly and knowledgeable assistant for Karmika Seva Kendra (KSK), \
a service center of the Karnataka Building & Other Construction Workers Welfare Board (KBOCWWB).

Your role:
- Help construction workers and their families understand welfare schemes, registration, \
and claim procedures.
- Speak in clear, simple language. Many users may not be highly educated, so avoid jargon.
- Be warm, patient, and encouraging. These workers deserve respectful, supportive guidance.

How to answer:
1. Use ONLY the information provided in the Context below. Do not guess or invent facts.
2. When listing schemes, benefits, eligibility, or documents, use bullet points or numbered lists \
for easy reading.
3. Include specific amounts (in rupees) and deadlines whenever the context provides them.
4. If the context does not contain enough information to fully answer, say honestly: \
"I don't have complete information on that topic. I recommend visiting your nearest \
Karmika Seva Kendra or calling the helpline for more details."
5. If the question is completely outside the scope of KSK/KBOCWWB (for example, unrelated \
topics like cooking, sports, politics), politely say: \
"I'm here to help with questions about construction worker welfare schemes and KSK services. \
Could you ask me something related to that?"

Formatting guidelines:
- Start with a direct answer to the question.
- Follow up with relevant details (eligibility, documents needed, amounts, deadlines).
- End with a helpful note or next step when appropriate (e.g., "You can apply at your nearest KSK center").
- Keep responses concise but complete. Do not pad with unnecessary filler.

Context:
{context}
"""


def _build_system_prompt(context: str, language: str = "") -> str:
    """Assemble the full system prompt with retrieved context and optional language."""
    prompt = SYSTEM_PROMPT.format(context=context)
    if language:
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
                "- If you don't have enough information, say: "
                '"ಈ ವಿಷಯದ ಬಗ್ಗೆ ನನ್ನ ಬಳಿ ಸಂಪೂರ್ಣ ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹತ್ತಿರದ '
                'ಕಾರ್ಮಿಕ ಸೇವಾ ಕೇಂದ್ರಕ್ಕೆ ಭೇಟಿ ನೀಡಿ ಅಥವಾ ಸಹಾಯವಾಣಿಗೆ ಕರೆ ಮಾಡಿ."\n'
                "- If the question is off-topic, say: "
                '"ನಾನು ಕಟ್ಟಡ ಕಾರ್ಮಿಕರ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳು ಮತ್ತು KSK ಸೇವೆಗಳ ಬಗ್ಗೆ ಸಹಾಯ '
                'ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಅದಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆ ಕೇಳಿ."\n'
            )
        else:
            prompt += (
                f"\nIMPORTANT: Respond entirely in {lang_name}. Translate all information "
                f"naturally into {lang_name} while keeping rupee amounts as numerals and "
                f"proper nouns as-is.\n"
            )
    return prompt


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
        logger.info("No relevant context found above score threshold %.2f", score_threshold)
        return ""

    for i, p in enumerate(points):
        logger.debug("Point %d score=%.4f payload: %s", i, p.score, p.payload)

    context = "\n\n---\n\n".join([r.payload["text"] for r in points])
    logger.debug("Context content:\n%s", context)
    return context


# ---------------------------------------------------------------------------
# Answer (non-streaming)
# ---------------------------------------------------------------------------
async def answer(
    question: str,
    qdrant: Optional[AsyncQdrantClient] = None,
    ollama: Optional[OllamaClient] = None,
    history: Optional[list[dict]] = None,
    language: str = "",
) -> str:
    logger.info(
        "RAG answer() question=%s language=%s",
        question[:100], language or "en",
    )
    context = await retrieve(question, qdrant=qdrant, ollama=ollama)
    system_prompt = _build_system_prompt(context, language)

    ollama = ollama or default_ollama

    truncated_history = None
    if history:
        truncated_history = history[-settings.MAX_HISTORY_MESSAGES:]
        logger.debug(
            "Using %d of %d history messages",
            len(truncated_history),
            len(history),
        )

    return await ollama.chat(
        system_prompt=system_prompt,
        user_message=question,
        history=truncated_history,
    )


# ---------------------------------------------------------------------------
# Answer (streaming)
# ---------------------------------------------------------------------------
async def answer_stream(
    question: str,
    qdrant: Optional[AsyncQdrantClient] = None,
    ollama: Optional[OllamaClient] = None,
    history: Optional[list[dict]] = None,
    language: str = "",
) -> AsyncIterator[str]:
    """Streaming variant — yields text chunks as they arrive from Ollama."""
    logger.info(
        "RAG answer_stream() question=%s language=%s",
        question[:100], language or "en",
    )
    context = await retrieve(question, qdrant=qdrant, ollama=ollama)
    system_prompt = _build_system_prompt(context, language)

    ollama = ollama or default_ollama

    truncated_history = None
    if history:
        truncated_history = history[-settings.MAX_HISTORY_MESSAGES:]

    async for chunk in ollama.chat_stream(
        system_prompt=system_prompt,
        user_message=question,
        history=truncated_history,
    ):
        yield chunk
