#Actionableitems , decision , questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 


def get_llm():
    return ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2)


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. From this video transcript, "
        "extract actionable takeaways — things the viewer is told or "
        "recommended to do, try, avoid, or follow up on. For each provide:\n"
        "- Action/recommendation\n"
        "- Who it's for or who suggested it (if mentioned, else 'General audience')\n"
        "- Any timeframe or condition mentioned (if none, write 'Not specified')\n\n"
        "Format as a numbered list. If the transcript contains no actionable "
        "takeaways, say exactly: 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. From this video transcript, "
        "extract the key points, conclusions, or claims the speaker "
        "asserts or settles on. Format as a numbered list. "
        "If none are found, say exactly: 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From this video transcript, extract any questions raised but not "
        "answered, topics mentioned as needing further explanation, or "
        "areas the speaker says they'll cover later / didn't get to. "
        "Format as a numbered list. If none are found, say exactly: "
        "'No open questions found.'"
    )
    return chain.invoke(transcript)