import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


# ─────────────────────────────────────────────────────────────
# Mistral LLM
# ─────────────────────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is not set. "
            "Please add it to your .env file."
        )

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )


# ─────────────────────────────────────────────────────────────
# Format retrieved documents
# ─────────────────────────────────────────────────────────────

def format_docs(docs):
    if not docs:
        return "No relevant information was found."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ─────────────────────────────────────────────────────────────
# RAG Prompt
# ─────────────────────────────────────────────────────────────

def get_rag_prompt():

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Rules:
- Use only the provided transcript context.
- Do not invent information.
- Be concise and precise.
- If quoting someone, make it clear that it is a quote.
- Do not use outside knowledge.

Context from meeting transcript:

{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )


# ─────────────────────────────────────────────────────────────
# Build RAG chain for a new transcript
# ─────────────────────────────────────────────────────────────

def build_rag_chain(transcript: str):

    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcript is empty. Cannot build RAG chain."
        )

    print("Building vector store...")

    # Create vector store from transcript
    vector_store = build_vector_store(transcript)

    print("Vector store created.")

    # Create retriever
    retriever = get_retriever(
        vector_store,
        k=4,
    )

    print("Retriever created.")

    # Load Mistral
    llm = get_llm()

    # Prompt
    prompt = get_rag_prompt()

    # ─────────────────────────────────────────────────────────
    # LCEL RAG pipeline
    # ─────────────────────────────────────────────────────────

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain ready.")

    return rag_chain


# ─────────────────────────────────────────────────────────────
# Load an existing RAG chain
# ─────────────────────────────────────────────────────────────

def load_rag_chain():

    print("Loading existing vector store...")

    vector_store = load_vector_store()

    if vector_store is None:
        raise ValueError(
            "No vector store found. "
            "Build the RAG chain with a transcript first."
        )

    # Create retriever from loaded vector store
    retriever = get_retriever(
        vector_store,
        k=4,
    )

    print("Retriever loaded.")

    # Load Mistral
    llm = get_llm()

    # Prompt
    prompt = get_rag_prompt()

    # RAG chain
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Existing RAG chain loaded.")

    return rag_chain


# ─────────────────────────────────────────────────────────────
# Ask a question
# ─────────────────────────────────────────────────────────────

def ask_question(rag_chain, question: str) -> str:

    if not rag_chain:
        raise ValueError(
            "RAG chain is not initialized."
        )

    if not question or not question.strip():
        raise ValueError(
            "Please enter a question."
        )

    question = question.strip()

    print(f"\nQuestion: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer