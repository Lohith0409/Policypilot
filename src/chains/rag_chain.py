"""
The core RAG chain: retrieve relevant chunks, then ask the LLM to answer
using ONLY those chunks.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, LLM_MODEL
from src.retrieval.retriever import get_retriever


SYSTEM_PROMPT = """You are PolicyPilot, an assistant that answers questions \
using ONLY the policy document excerpts provided as context below.

Rules you must follow:
- Answer using ONLY the information in the context. Never use outside knowledge.
- If the context does not contain enough information to answer the question, \
respond with EXACTLY this sentence and nothing else: \
"I couldn't find this in the policy documents provided."
- For every factual claim you make, cite the source in this exact format: \
(Source: filename, page X)
- IMPORTANT: Policy documents often contain multiple distinct provisions that \
apply to different scenarios (for example, different rules for different time \
periods, employee types, or conditions). If the retrieved context contains \
more than one provision that could be relevant to the question, do NOT pick \
just one and present it as the complete answer. Instead, identify each \
distinct provision separately, state which specific scenario or condition \
each one applies to, and clearly note if the question itself is ambiguous \
as to which provision applies.
- Be concise and direct. Do not add disclaimers beyond what's asked.

Context:
{context}
"""


def format_docs_for_prompt(docs: list[Document]) -> str:
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0,
    )


def answer_question(question: str, session_id: str | None = None) -> dict:
    """Run the full RAG flow for one question: retrieve, then generate.

    top_k=6 (up from 4): a deliberate fix after discovering that ambiguous
    questions could retrieve only ONE of several relevant clauses at k=4,
    causing the model to confidently answer with an incomplete picture.
    Retrieving more candidates gives the model a better chance of seeing
    ALL relevant provisions, not just the closest match.
    """
    retriever = get_retriever(top_k=6, session_id=session_id)
    source_docs = retriever.invoke(question)
    context = format_docs_for_prompt(source_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "source_docs": source_docs,
    }