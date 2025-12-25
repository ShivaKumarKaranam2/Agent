from crewai import Task
from agents import (
    document_summarizer_keywords_agent,
    query_responder_agent,
    internet_connected_agent,
    manager_agent
)
from tools import (
    document_summarization_tool,
    keyword_extraction_tool,
    document_query_answering_tool,
    internet_search_tool,
    agent_selection_tool
)
from models import SummarizerOutput, ResponderOutput, InternetOutput, ManagerDecision

document_summarizer_keywords_task = Task(
    description="Summarize the document and extract relevant keywords. Document Content: {document_content}. If 'document_content' is empty or unavailable, explicitly state 'No document provided for summarization.'",
    expected_output="A JSON object with 'document' (summary) and 'keywords' (list of strings).",
    agent=document_summarizer_keywords_agent,
    tools=[document_summarization_tool, keyword_extraction_tool],
    output_pydantic=SummarizerOutput,
    )

query_responder_task = Task(
        description="Respond to the user query: '{query}' based strictly on the document summary and keywords. If the answer is NOT found in the document content, you must set the 'response' field to exactly 'NOT_FOUND_IN_DOCUMENT' (without quotes). Do not attempt to make up an answer.",
        expected_output="A JSON object with 'query' and 'response'.",
        agent=query_responder_agent,
        tools=[document_query_answering_tool],
        output_pydantic=ResponderOutput,
    )

internet_connected_task = Task(
        description="Search the internet for information related to: '{query}' to provide a comprehensive answer.",
        expected_output="A JSON object with 'query', 'response', and 'source'.",
        agent=internet_connected_agent,
        tools=[internet_search_tool],
        output_pydantic=InternetOutput,
    )

manager_planning_task = Task(
    description=(
        "Analyze the user's input to determine the execution plan. "
        "User Input: Query='{query}', Document Present='{has_document}'. "
        "Strictly follow these rules:\n"
        "1. IF Document ONLY: Select 'Document Summarizer and Keyword Extractor'. Execution Mode: 'parallel'.\n"
        "2. IF Document AND Query: ALWAYS select 'Query Responder Agent' AND 'Internet Connected Agent'. Execution Mode: 'parallel'.\n"
        "3. IF Query ONLY (no doc): Select 'Internet Connected Agent'. Execution Mode: 'hierarchical'.\n"
        "Output a STRICT JSON object matching ManagerDecision schema."
    ),
    expected_output="A JSON object defining the execution plan.",
    tools=[agent_selection_tool],
    agent=manager_agent,
    output_pydantic=ManagerDecision,
)
