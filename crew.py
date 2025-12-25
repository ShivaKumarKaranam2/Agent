from crewai import Crew, Process
from tasks import manager_task, document_summarizer_keywords_task,query_responder_task,internet_connected_task
from agents import manager_agent, document_summarizer_keywords_agent, query_responder_agent, internet_connected_agent, llm

crew = Crew(
    name="Manager Agent",
    agents=[manager_agent, document_summarizer_keywords_agent, query_responder_agent, internet_connected_agent],
    tasks=[manager_task, document_summarizer_keywords_task, query_responder_task, internet_connected_task],
    process=Process.hierarchical, 
    manager_llm=llm,
    memory=False,
    cache=True,
    max_rpm=5 # Reduced to throttle requests
)

if __name__ == "__main__":
    # Inputs for the tasks
    result = crew.kickoff(inputs=inputs)
    print(result)
