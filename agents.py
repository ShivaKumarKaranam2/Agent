from crewai import Agent, LLM
from tools import document_summarization_tool, keyword_extraction_tool, document_query_answering_tool, internet_search_tool, agent_selection_tool 

from dotenv import load_dotenv
load_dotenv()  # Loads GROQ_API_KEY safely

import os
llm = LLM(
    model="openai/mistralai/mistral-large-3-675b-instruct-2512",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.15,
    max_tokens=2048
)

# # Debug safety check (optional but recommended)
# assert isinstance(llm, LLM), "llm must be an instance of crewai.LLM"


document_summarizer_keywords_agent = Agent(
    role="Document Summarizer and Keyword Extractor",
    goal="Summarize the provided document efficiently and extract key topics and keywords. Keywords should be in the for of list like[keyword1, keyword2, keyword3]",
    backstory=(
        "You are an expert content analyst and summarizer. "
        "You specialize in processing complex text documents, distilling them into "
        "clear, concise summaries, and identifying the most critical keywords and themes. "
        "You always provide your output in a strict JSON format matching the SummarizerOutput schema."
    ),
    llm=llm,
    tools=[document_summarization_tool, keyword_extraction_tool],
    verbose=True,
    memory=False,
    allow_delegation=False
)

query_responder_agent = Agent(
    role="Query Responder Agent",
    goal="Respond to user queries based strictly on provided documents",
    backstory=(
        "You are an expert at answering questions using only supplied document content. "
        "You do not use external knowledge. "
        "You always provide your output in a strict JSON format matching the ResponderOutput schema."
    ),
    
    tools=[document_query_answering_tool],
    verbose=True,
    memory=False,
    llm=llm,
    allow_delegation=False
)

internet_connected_agent = Agent(
    role="Internet Connected Agent",
    goal="Search the internet for up-to-date information and return verified answers",
    backstory=(
        "You are a skilled online researcher who fetches real-time information "
        "from reliable sources and reports it clearly. "
        "You always provide your output in a strict JSON format matching the InternetOutput schema."
    ),
    llm=llm,
    tools=[internet_search_tool],
    verbose=True,
    memory=False,
    allow_delegation=False
)


manager_agent = Agent(
    role="Manager Agent",
    goal="Analyze user input and delegate tasks to the appropriate specialized agents",
    backstory=(
        "You are a senior AI orchestrator responsible for selecting the correct agents, "
        "deciding execution order, and consolidating final responses. "
        "Your final output must be a consolidated JSON report matching the ManagerOutput schema."
    ),
    llm=llm,
    tools=[agent_selection_tool],
    verbose=True,
    memory=False,
    allow_delegation=True
)
