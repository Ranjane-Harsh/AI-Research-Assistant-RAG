import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config
import re

REF_PATTERN = re.compile(r"^\s*(\d+\.?\s*)?(REFERENCES|BIBLIOGRAPHY|LITERATURE CITED)\b", re.IGNORECASE)

def block_header_footer(block_rect: pymupdf.Rect, page_height: float, page_num: int) -> bool:
    y0, y1 = block_rect.y0, block_rect.y1

    if page_num > 1 and y1 < (page_height * 0.06):
        return True

    if y0 > (page_height * 0.94):
        return True
    
    return False

def load_pdf(filepath: str) -> list[Document]:
    doc = pymupdf.open(filepath)
    total_pages = len(doc)
    pages = []
    stop_ingestion = False

    for page_num, page in enumerate(doc, start=1):
        if stop_ingestion:
            break
        page_height = page.rect.height
        blocks = page.get_text("blocks", sort = True)
        page_text_pieces = []

        for b in blocks:
           # block tuple format: (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 is text; 1 is an image
            if b[6] != 0:
                continue
            block_text = b[4].strip()
            if not block_text:
                continue

            block_rect = pymupdf.Rect(b[:4])

            if block_header_footer(block_rect, page_height, page_num):
                continue

            if page_num > (total_pages * 0.5) and REF_PATTERN.search(block_text):
                print(f"[Ingestion] Reference cutoff triggered on Page {page_num}. Halting further extraction")

                stop_ingestion = True
                break

            page_text_pieces.append(block_text)

        cleaned_page_text = "\n\n".join(page_text_pieces).strip()
        if cleaned_page_text:
            pages.append(
                Document(page_content=cleaned_page_text, metadata = {"source": filepath, "page": page_num})
            )

    doc.close()
    return pages

def chunk_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size = config.CHUNK_SIZE, chunk_overlap = config.CHUNK_OVERLAP, separators= ["\n\n", "\n", ". ", " ", ""])

    return splitter.split_documents(pages)

def ingest_pdf(filepath: str) -> list[Document]:
    pages = load_pdf(filepath)
    chunks = chunk_documents(pages)
    return chunks