import streamlit as st
import os
from core.generator import generate_answer
import config

def render():
    st.title("🔍 Academic Research Query")
    st.markdown("Ask precise questions across your indexed papers. Answers are strictly grounded with verifiable inline citations.")

    with st.form(key="query_form"):
        user_query = st.text_input(
            "Enter your research question:", 
            placeholder="e.g. What was the PUMA 560 robot used for in Surgery?"
        )
        submit_btn = st.form_submit_button("Generate Answer", use_container_width=True)

        if submit_btn and user_query.strip():
            with st.spinner("Executing Two-Stage Retrieval (Bi-Encoder + Cross-Encoder Transformer) & Sythesizing answer via Groq ..."):
                try:
                    result = generate_answer(user_query.strip())
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    return

            st.subheader("💡 Sythesized Answer")
            st.markdown(result["answer"])

            st.divider()

            st.subheader("📎Retrieved Evidence and Re-ranking confidence")
            sources = result.get("sources", [])

            if not sources:
                st.warning("No context passages were retrieved.")
            else:
                with st.expander(f"Inspect {len(sources)} Evidence Passages used (Ranked by Cross-Encoder)", expanded=True):
                    for i, doc in enumerate(sources, start=1):
                        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                        page = doc.metadata.get("page", "?")
                        score = doc.metadata.get("rerank_score", None)

                        st.markdown(f"#### Rank #{i} - `{source_name}` (Page {page})")
                        if score is not None:
                            st.caption(f"**Cross-Encoder Relevance Score:** `{score:+.4f}`")

                        st.info(doc.page_content)
