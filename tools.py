from crewai.tools import tool
from typing import List
import json
from duckduckgo_search import DDGS

@tool("Document_Summarization_Tool")
def document_summarization_tool(document_text: str) -> str:
    """
    Returns raw document text for the agent to summarize.
    The agent's LLM will handle summarization.
    """
    return document_text


@tool("Keyword_Extraction_Tool")
def keyword_extraction_tool(document_text: str) -> str:
    """
    Returns raw document text for keyword extraction.
    """
    return document_text


@tool("Document_Query_Answering_Tool")
def document_query_answering_tool(query: str, documents: List[str]) -> str:
    """
    Returns combined document context and query.
    """
    context = "\n\n".join(documents)
    return f"Context:\n{context}\n\nQuery:\n{query}"


@tool("Internet_Information_Fetching_Tool")
def internet_search_tool(query: str) -> str:
    """
    Fetches real-time information using DuckDuckGo.
    """
    results = DDGS().text(query, max_results=5)

    if not results:
        return json.dumps({
            "response": "No results found",
            "source": "DuckDuckGo"
        })

    formatted_results = [
        {
            "title": r["title"],
            "snippet": r["body"],
            "url": r["href"]
        }
        for r in results
    ]

    return json.dumps({
        "response": formatted_results,
        "source": "DuckDuckGo"
    })


@tool("Agent_Selection_Decision_Tool")
def agent_selection_tool(user_input: str) -> dict:
    """
    Returns a structured decision skeleton.
    The Manager Agent's LLM will finalize the reasoning.
    """
    return {
        "user_input": user_input,
        "hint": "Decide which agents are needed based on task type"
    }
