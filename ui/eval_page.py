# ui/eval_page.py
import streamlit as st
import pandas as pd
import os
import tempfile
from evaluation.evaluator import run_ragas_evaluation


DEFAULT_GOLDEN_DATASET = "evaluation/golden_dataset.csv"


def render():
    st.title("📊 RAGAS Evaluation Dashboard")
    st.markdown(
        "Quantitatively evaluate our RAG pipeline against a curated golden dataset "
        "using the **RAG Triad** (Faithfulness, Relevancy, Recall, Precision)."
    )

    # --- Section 1: Theory Drawer ---
    with st.expander("ℹ️ How are these metrics calculated?", expanded=False):
        st.markdown("""
        - **Faithfulness:** Measures factual consistency. Are all claims in the answer supported by the retrieved context? (Hallucination detection).
        - **Answer Relevancy:** Measures whether the answer directly addresses the query without fluff or divergence.
        - **Context Recall:** Did our two-stage retriever find all the necessary facts required by the ground-truth reference?
        - **Context Precision:** Signal-to-noise ratio: did the most relevant chunks get ranked higher by the Cross-Encoder?
        """)

    # --- Section 2: Dataset Selector ---
    st.subheader("1. Select Evaluation Dataset")
    col1, col2 = st.columns([2, 1])

    dataset_choice = col1.radio(
        "Choose benchmark data source:",
        ["Use Default Golden Dataset (5 Questions)", "Upload Custom CSV"],
        horizontal=True
    )

    dataset_path = None

    if dataset_choice == "Use Default Golden Dataset (5 Questions)":
        if os.path.exists(DEFAULT_GOLDEN_DATASET):
            dataset_path = DEFAULT_GOLDEN_DATASET
            preview_df = pd.read_csv(dataset_path)
            st.caption(f"Loaded `{DEFAULT_GOLDEN_DATASET}` ({len(preview_df)} Q&A pairs)")
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.error(f"Default dataset `{DEFAULT_GOLDEN_DATASET}` not found.")
    else:
        uploaded_csv = st.file_uploader("Upload golden dataset (.csv with 'question' and 'ground_truth')", type=["csv"])
        if uploaded_csv:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_csv.getvalue())
                dataset_path = tmp.name
            preview_df = pd.read_csv(dataset_path)
            st.dataframe(preview_df, use_container_width=True)

    # --- Section 3: Run Evaluation ---
    st.divider()
    st.subheader("2. Execute Quantitative Evaluation")

    if dataset_path and st.button("🚀 Run RAGAS Benchmark", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio, message):
            progress_bar.progress(ratio)
            status_text.text(message)

        try:
            results_df = run_ragas_evaluation(dataset_path, progress_callback=update_progress)
            st.session_state["eval_results"] = results_df
            st.success("✅ Evaluation successfully executed!")
        except Exception as e:
            st.error(f"❌ Evaluation error: {e}")

    # --- Section 4: Display Results ---
    if "eval_results" in st.session_state:
        results_df = st.session_state["eval_results"]

        st.divider()
        st.subheader("3. Benchmark Results")

        # Metric summary cards
        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        available_cols = [c for c in metric_cols if c in results_df.columns]
        avg_scores = results_df[available_cols].mean()

        m_cols = st.columns(len(available_cols))
        for col_widget, col_name in zip(m_cols, available_cols):
            score_val = avg_scores[col_name]
            col_widget.metric(
                label=col_name.replace("_", " ").title(),
                value=f"{score_val:.2f}",
                delta=f"{(score_val - 0.70):+.2f} vs Target (0.70)" if score_val >= 0.70 else f"{score_val - 0.70:.2f}"
            )

        # Bar chart comparison
        st.subheader("Metric Distribution")
        st.bar_chart(avg_scores)

        # Full detailed dataframe
        st.subheader("Detailed Question-by-Question Breakdown")
        st.dataframe(results_df, use_container_width=True)

        # Diagnostic Failure Analysis
        st.subheader("🔍 Automated Diagnostics")
        low_faith = results_df[results_df["faithfulness"] < 0.80] if "faithfulness" in results_df.columns else []
        low_recall = results_df[results_df["context_recall"] < 0.70] if "context_recall" in results_df.columns else []

        if len(low_faith) == 0 and len(low_recall) == 0:
            st.success("🎯 Outstanding Performance: No hallucinations detected and all context recall thresholds satisfied!")
        else:
            if len(low_faith) > 0:
                st.warning(f"⚠️ {len(low_faith)} question(s) had Faithfulness < 0.80 (Potential ungrounded claims).")
            if len(low_recall) > 0:
                st.warning(f"⚠️ {len(low_recall)} question(s) had Context Recall < 0.70 (Retriever missed ground-truth facts).")