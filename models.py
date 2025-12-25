from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

class SummarizerOutput(BaseModel):
    document: str
    keywords: List[str]

class ResponderOutput(BaseModel):
    document_content: str
    query: str
    response: str

class InternetOutput(BaseModel):
    query: str
    response: str
    source: str

class ManagerDecisionDetail(BaseModel):
    document_present: bool
    query_present: bool
    execution_mode: Literal["parallel", "sequential", "hierarchical"]
    fallback_to_internet: bool
    reason: str

class ManagerDecision(BaseModel):
    decision: ManagerDecisionDetail
    selected_agents: List[str]

# We might not need this if we aren't using a final consolidated JSON from the manager for the UI
# But user requested "Manager output MUST follow this structure" for the Manager Agent.
# The `selected_agents` list implies the Manager *outputs* this plan, and then we execute.
