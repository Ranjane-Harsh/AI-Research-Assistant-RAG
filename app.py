import streamlit as st
from ui import upload_page, query_page, eval_page

st.set_page_config(
    page_title="Academic Literature RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
st.sidebar.caption("Two-Stage Retrieval RAG System")

selected_page = st.sidebar.radio(
    "Go to:",
    ["🔍 Ask a Question", "📄 Document Library", "📊 Evaluation Dashboard"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Architecture Stack: **\n"
    "- **Bi-Encoder:** BAAI/bge-base-en-v1.5\n"
    "- **Re-Ranker:** BAAI/bge-reranker-base\n"
    "- **Vector DB:** ChromaDB (Persistent)\n"
    "- **LLM:** LLaMA 3.1 8B (via Groq)"
)

if selected_page == "🔍 Ask a Question":
    query_page.render()
elif selected_page == "📄 Document Library":
    upload_page.render()
elif selected_page == "📊 Evaluation Dashboard":
    eval_page.render()