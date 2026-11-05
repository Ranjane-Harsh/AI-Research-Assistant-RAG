import streamlit as st

def render():
    st.title("📊 RAGAS Evaluation Dashboard")
    st.info("The automated quantitative evaluation module.")
    st.markdown(
        """
        ### Planned Metrics:
        - **Faithfulness:** Measures whether the answer is strictly derived from the context (hallucination detector).
        - **Answer Relevancy:** Measures how directly the response addresses the query.
        - **Context Recall:** Evaluates whether our two-stage retriever found all necessary evidence.
        - **Context Precision:** Measures the signal-to-noise ratio in the retrieved passages.
    
"""
    )