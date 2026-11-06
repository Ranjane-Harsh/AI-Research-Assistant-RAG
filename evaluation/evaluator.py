import os
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from core.generator import generate_answer
from core.embeddings import get_embedding_model
import config


def run_ragas_evaluation(csv_path: str, progress_callback=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("The Golden Dataset must contain 'question' and 'ground_truth' columns.")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    total_rows = len(df)
    print(f" [Evaluation] Executing RAG pipeline for {total_rows} questions ...")

    for idx, row in df.iterrows():
        q = str(row["question"]).strip()
        gt = str(row["ground_truth"]).strip()

        if progress_callback:
            progress_callback((idx) / total_rows, f"Generating answer for Question {idx + 1} / {total_rows} ...")

        rag_output = generate_answer(q)
        retrieved_text_chunks = [doc.page_content for doc in rag_output["sources"]]

        questions.append(q)
        answers.append(rag_output["answer"])
        contexts.append(retrieved_text_chunks)
        ground_truths.append(gt)

    if progress_callback:
        progress_callback(0.9, "Running RAGAS metric scoring via Groq Judge...")

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Wrap Groq LLM with bypass_n=True because Groq does not support n > 1
    base_llm = ChatGroq(
        api_key=config.GROQ_API_KEY,
        model_name=config.GROQ_MODEL,
        temperature=0.0,
    )
    evaluator_llm = LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings=get_embedding_model())

    # Limit concurrency to 2 workers to avoid overwhelming Groq API rate limits and timeouts
    run_config = RunConfig(
        timeout=180,
        max_workers=2,
        max_retries=5,
    )

    print(" [Evaluation] Scoring metrics (Faithfulness, Relevancy, Recall, Precision)...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    results_df = results.to_pandas()

    # Ragas renames input columns to user_input, response, etc.
    # Map them back so downstream code accessing 'question', 'answer', etc. works smoothly
    rename_map = {
        "user_input": "question",
        "response": "answer",
        "reference": "ground_truth",
        "retrieved_contexts": "contexts",
    }
    results_df.rename(columns=rename_map, inplace=True)

    os.makedirs("evaluation/results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation/results/eval_run_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f" [Evaluation] Saved run results to: {output_file}")

    if progress_callback:
        progress_callback(1.0, "Evaluation Complete!")

    return results_df
