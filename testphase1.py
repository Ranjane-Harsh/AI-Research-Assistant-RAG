from core.ingestion import ingest_pdf
from core.vector_store import add_documents, list_papers, similarity_search
from core.retriever import retrieve
import config

pdf_path = r"D:\Coding\Projects\Academic Literature RAG Assistant\data\papers\Evolution-of-Robotics-in-Medical-Surgeries-and-Health-Care-Systems-A-Review.pdf"

chunks = ingest_pdf(pdf_path)

print(f"Total chunks produced: {len(chunks)}")
print("\n--- First chunk (text) ---")
print(chunks[0].page_content[:300])
print("\n--- First chunk (metadata) ---")
print(chunks[0].metadata)

print("Step 2: Saving chunks to ChromaDB...")
print("  (First run downloads the embedding model ~440MB — this takes a few minutes)")
add_documents(chunks)
print("  ✅ Saved to ChromaDB\n")

print("Step 3: Listing indexed papers...")
papers = list_papers()
print(f"  ✅ Papers in index: {papers}\n")

print("Step 4: Running a semantic similarity search...")
query = "What was the PUMA 560 robot used for in surgery?"
results = similarity_search(query=query, top_k= config.INITIAL_TOP_K)

print("==========================================================")
print(f"QUERY: '{query}'")
print("==========================================================\n")
# 1. Inspect Stage 1 (Before Re-Ranking)
print("--- STAGE 1: Top 5 Candidates from ChromaDB (Bi-Encoder) ---")
for rank, doc in enumerate(results, start=1):
    page = doc.metadata.get("page", "?")
    has_puma = "PUMA 560" in doc.page_content
    marker = " ⭐ [CONTAINS 'PUMA 560']" if has_puma else ""
    snippet = doc.page_content.replace("\n", " ")[:120]
    print(f"Rank {rank} | Page {page:2} | {snippet}...{marker}")
print("\n" + "=" * 58 + "\n")
# 2. Inspect Stage 2 (After Cross-Encoder Re-Ranking)
print("--- STAGE 2: Top 3 Re-Ranked Results (Cross-Encoder) ---")
final_results = retrieve(query, top_k=config.FINAL_TOP_K)
for rank, doc in enumerate(final_results, start=1):
    page = doc.metadata.get("page", "?")
    score = doc.metadata.get("rerank_score", 0.0)
    has_puma = "PUMA 560" in doc.page_content
    marker = " ⭐ [CONTAINS 'PUMA 560']" if has_puma else ""
    snippet = doc.page_content.replace("\n", " ")[:140]
    print(f"Rank {rank} | Score: {score:+.4f} | Page {page:2} | {snippet}...{marker}")
print("\n--- Diagnostic Check ---")
if final_results and "PUMA 560" in final_results[0].page_content:
    print("🎯 PHENOMENAL SUCCESS! Rank #1 is the exact passage describing PUMA 560!")
    print("   The Cross-Encoder successfully identified and promoted the exact entity.")
else:
    print("Check ranks: PUMA 560 should be promoted significantly higher than in Stage 1.")