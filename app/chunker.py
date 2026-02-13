from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [
    ("#", "section"),
    ("##", "scheme"),
    ("###", "subsection")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

def chunk_markdown(text):
    docs = splitter.split_text(text)
    return [d.page_content for d in docs]
