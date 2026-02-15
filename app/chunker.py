from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [
    ("#", "section"),
    ("##", "scheme"),
    ("###", "subsection")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

def chunk_markdown(text):
    docs = splitter.split_text(text)
    chunks = []
    for d in docs:
        metadata = d.metadata
        content = d.page_content
        
        # Build context string from metadata
        context_parts = []
        if "section" in metadata:
            context_parts.append(f"Section: {metadata['section']}")
        if "scheme" in metadata:
            context_parts.append(f"Scheme: {metadata['scheme']}")
        if "subsection" in metadata:
            context_parts.append(f"Subsection: {metadata['subsection']}")
            
        if context_parts:
            # Prepend context to content
            full_content = "\n".join(context_parts) + "\n\n" + content
        else:
            full_content = content
            
        chunks.append(full_content)
        
    return chunks
