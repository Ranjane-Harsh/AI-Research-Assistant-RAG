import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from core.retriever import retrieve
import config

SYSTEM_PROMPT = """ 
You are an expert academic research assistant specializing in synthesizing scientific literature.

Your task is to answer user's question using ONLY the facts directly mentioned in the provided Context.

CRITICAL INSTRUCTIONS:
1. Strict Grounding: Every statement in your answer must be supported by the Context. Do NOT assume, extrapolate or use outside knowledge.
2. Inline Citation: Whenever you cite a fact, immediately follow it with an inline citation in the exact format: [Source: <filename>, Page <page>].
3. Refusal Condition: If the provided context, does not contain enough information to answer the question, clearly state: "I could not find an answer to this question in the provided literature".
4. Tone and Style: Be concise, rigorous, and direct. Avoid coversational filler.

Context:
{context}

"""
def format_context(docs: list[Document]) -> str:
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("rerank_score", None)
        score_str = f" (Relevance Score: {score:+.4f})" if score is not None else ""

        block = (
            f"--- Document [{i}] | Source: {source} | Page: {page}{score_str} ---\n"
            f"{doc.page_content.strip()}"
        )
        formatted_chunks.append(block)

    return "\n\n".join(formatted_chunks)

def generate_answer(query: str, top_k: int = None) -> dict:
    docs = retrieve(query=query, top_k=top_k)

    if not docs:
        return{
            "query": query,
            "answer": "No relevant literature could be retrived from the index.",
            "sources": []
        }

    context_str = format_context(docs=docs)

    llm = ChatGroq(
        api_key=config.GROQ_API_KEY,
        model_name= config.GROQ_MODEL,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    reponse_text = chain.invoke({
        "context": context_str,
        "question": query
    })

    return {
        "query": query,
        "answer": reponse_text.strip(),
        "sources": docs
    }