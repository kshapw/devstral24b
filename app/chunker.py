import logging

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import settings

logger = logging.getLogger(__name__)

headers = [
    ("#", "section"),
    ("##", "scheme"),
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_markdown(text: str) -> list[str]:
    try:
        docs = md_splitter.split_text(text)
    except Exception:
        logger.error("Failed to split markdown text (len=%d)", len(text), exc_info=True)
        raise
    logger.info("Split markdown into %d raw document sections", len(docs))

    chunks: list[str] = []
    for d in docs:
        metadata = d.metadata
        content = d.page_content

        # Build context string from metadata
        context_parts: list[str] = []
        if "section" in metadata:
            context_parts.append(f"Section: {metadata['section']}")
        if "scheme" in metadata:
            context_parts.append(f"Scheme: {metadata['scheme']}")

        if context_parts:
            full_content = "\n".join(context_parts) + "\n\n" + content
        else:
            full_content = content

        # If this chunk is too large, sub-split it
        if len(full_content) > settings.CHUNK_SIZE:
            sub_chunks = size_splitter.split_text(full_content)
            logger.debug(
                "Sub-split oversized chunk (%d chars) into %d pieces",
                len(full_content),
                len(sub_chunks),
            )
            # Prepend metadata context to each sub-chunk so retrieval
            # still knows which section/scheme the text belongs to
            prefix = "\n".join(context_parts) + "\n\n" if context_parts else ""
            for sc in sub_chunks:
                if not sc.startswith(prefix.strip()):
                    chunks.append(prefix + sc)
                else:
                    chunks.append(sc)
        else:
            chunks.append(full_content)

    logger.info("Produced %d chunks with metadata context (size-limited)", len(chunks))
    return chunks
