import os

from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    # Prompt for summarizing individual transcript chunks
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely.",
            ),
            (
                "human",
                "{text}",
            ),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    # Split transcript into chunks
    chunks = split_transcript(transcript)

    # Summarize each chunk
    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Combine all partial summaries
    combined = "\n\n".join(chunk_summaries)

    # Prompt for final summary
    combine_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. "
                "Combine these partial summaries into one "
                "professional meeting summary using bullet points.",
            ),
            (
                "human",
                "{text}",
            ),
        ]
    )

    combine_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combine_prompt
        | llm
        | StrOutputParser()
    )

    return combine_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    # Prompt for generating meeting title
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a short professional meeting title "
                "(maximum 8 words). Return only the title.",
            ),
            (
                "human",
                "{text}",
            ),
        ]
    )

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])