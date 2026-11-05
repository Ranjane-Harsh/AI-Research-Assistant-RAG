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