import streamlit as st
import os
from core.ingestion import ingest_pdf
from core.vector_store import add_documents, list_papers, delete_paper
import config

def render():
    st.title("📄 Document Library & Ingestion")
    st.markdown("Upload Academic Research Papers (PDFs) to parse, chunk, and index them into Chroma DB.")

    st.subheader("Upload Research Paper")
    uploaded_file = st.file_uploader("Select an Academic PDF", type=["pdf"])

    if uploaded_file is not None:
        col1, col2 = st.columns([3,1])
        col1.write(f"**Selected File:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

        if col2.button("🚀 Ingest & Index", use_container_width=True):
            os.makedirs(config.PAPERS_DIR, exist_ok=True)
            save_path = os.path.join(config.PAPERS_DIR, uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Extracting text (layout-aware), stripping header and footers, and embedding chunks..."):

                try:
                    chunks = ingest_pdf(save_path)
                    add_documents(chunks=chunks)
                    st.success(f"✅ Sucessfully indexed {len(chunks)} chunks from `{uploaded_file.name}` !")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    st.divider()

    st.subheader("Indexed Literature in ChromaDB")
    indexed_papers = list_papers()

    if not indexed_papers:
        st.info("No research papers currently indexed in ChromaDB. Upload one above.")
    else:
        for paper_path in indexed_papers:
            filename = os.path.basename(paper_path)
            card = st.container(border=True)
            c1, c2 = card.columns([4,1])
            c1.markdown(f"📄 **{filename}**")
            c1.caption(f"Path: `{paper_path}`")

            if c2.button("🗑️ Delete", key=f"del_{paper_path}", use_container_width=True):
                delete_paper(paper_path)
                st.toast(f"Deleted {filename} from index.")
                st.rerun()

