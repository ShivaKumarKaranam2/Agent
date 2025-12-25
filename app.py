import streamlit as st
import pypdf
from crewai import Crew, Process
from models import ManagerDecision, SummarizerOutput, ResponderOutput, InternetOutput
from agents import (
    manager_agent,
    document_summarizer_keywords_agent,
    query_responder_agent,
    internet_connected_agent,llm
)
from tasks import (
    manager_planning_task,
    document_summarizer_keywords_task,
    query_responder_task,
    internet_connected_task
)

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Manager Agent Crew", page_icon="🤖")
st.title("🤖 Prioritized Agent Workflow")

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = []
if "uploaded_doc_text" not in st.session_state:
    st.session_state.uploaded_doc_text = ""
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ---------------- HELPERS ----------------
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = pypdf.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() for page in reader.pages)
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        return f"Error extracting text: {e}"
    return ""

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Input")

    uploaded_file = st.file_uploader(
        "Upload Document (TXT / PDF)", type=["txt", "pdf"]
    )

    if uploaded_file and uploaded_file.name != st.session_state.uploaded_file_name:
        with st.spinner("Processing document..."):
            st.session_state.uploaded_doc_text = extract_text_from_file(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("Document stored!")

    query = st.text_input("Query (optional)", placeholder="Ask a question about the document...")
    run_button = st.button("Run System", type="primary")

# ---------------- EXECUTION ----------------
if run_button:
    has_doc = bool(st.session_state.uploaded_doc_text)
    has_query = bool(query)

    if not has_doc and not has_query:
        st.error("Please upload a document or enter a query.")
        st.stop()

    # -------- PHASE 1: MANAGER PLANNING --------
    with st.status("Manager Agent: Planning...", expanded=True):
        planning_crew = Crew(
            agents=[manager_agent],
            tasks=[manager_planning_task],
            process=Process.hierarchical,
            manager_llm=llm,
            memory =False,
            cache=True,
            max_rpm=5 
        )

        plan_result = planning_crew.kickoff(
            inputs={
                "query": query or "None",
                "has_document": str(has_doc)
            }
        )

        decision: ManagerDecision = plan_result.pydantic
        st.success("Planning complete")

    # -------- PHASE 2: EXECUTION --------
    agent_map = {
        "Summarizer": (document_summarizer_keywords_agent, document_summarizer_keywords_task),
        "Responder": (query_responder_agent, query_responder_task),
        "Internet": (internet_connected_agent, internet_connected_task),
    }

    active_agents, active_tasks = [], []

    for name in decision.selected_agents:
        for key in agent_map:
            if key in name:
                agent, task = agent_map[key]
                active_agents.append(agent)
                active_tasks.append(task)

    worker_crew = Crew(
        agents=active_agents,
        tasks=active_tasks,
        process=Process.sequential,
        verbose=True
    )

    with st.spinner("Executing agents..."):
        output = worker_crew.kickoff(
            inputs={
                "query": query,
                "document_content": st.session_state.uploaded_doc_text
            }
        )

    st.session_state.results.append({
        "plan": decision,
        "output": output,
        "fallback_output": None
    })

    # -------- FALLBACK CHECK --------
    # Check if Responder failed
    needs_fallback = False
    if hasattr(output, "tasks_output"):
        for task_out in output.tasks_output:
            if "Responder" in str(task_out.agent) and hasattr(task_out, "pydantic"):
                if isinstance(task_out.pydantic, ResponderOutput):
                    # Robust check (case insensitive substring)
                    resp_text = task_out.pydantic.response.lower()
                    if "not_found" in resp_text or "no_document" in resp_text:
                        needs_fallback = True
                        break

    if needs_fallback:
        with st.spinner("Searching internet..."):
            internet_crew = Crew(
                agents=[internet_connected_agent],
                tasks=[internet_connected_task],
                process=Process.sequential
            )
            fallback_result = internet_crew.kickoff(inputs={"query": query})
            
            # Update the last result with fallback
            st.session_state.results[-1]["fallback_output"] = fallback_result
            
            # Update the plan in history to show this agent was eventually used
            # We must be careful if Pydantic model is immutable, but defaults are usually mutable.
            # If immutable, we'd need to reconstruct. Let's try direct append first.
            if "Internet Connected Agent" not in decision.selected_agents:
                decision.selected_agents.append("Internet Connected Agent")

    st.success("Execution completed")

# ---------------- HISTORY DISPLAY ----------------
if st.session_state.results:
    st.divider()
    for i, res in enumerate(reversed(st.session_state.results), start=1):
        with st.expander(f"Run {i}", expanded=(i == 1)):
            st.markdown("### 🧠 Manager Decision")
            st.json(res["plan"].model_dump())

            st.markdown("### 🤖 Agent Outputs")
            if hasattr(res["output"], "tasks_output"):
                for task_out in res["output"].tasks_output:
                    
                    # Try to parse Pydantic output
                    try:
                        pydantic_obj = task_out.pydantic
                        
                        # --- Summarizer Output ---
                        if isinstance(pydantic_obj, SummarizerOutput):
                            with st.container(border=True):
                                st.subheader("📄 Document Summary")
                                st.markdown(pydantic_obj.document)
                                st.markdown("**Keywords:**")
                                st.markdown(", ".join([f"`{k}`" for k in pydantic_obj.keywords]))

                        # --- Responder Output ---
                        elif isinstance(pydantic_obj, ResponderOutput):
                            if "not_found" in pydantic_obj.response.lower() or "no_document" in pydantic_obj.response.lower():
                                continue
                                
                            with st.container(border=True):
                                st.subheader("🙋‍♂️ Query Response")
                                st.info(f"**Q:** {pydantic_obj.query}")
                                st.success(f"**A:** {pydantic_obj.response}")

                        # --- Internet Output ---
                        elif isinstance(pydantic_obj, InternetOutput):
                            with st.container(border=True):
                                st.subheader("🌐 Internet Search Result")
                                st.info(f"**Q:** {pydantic_obj.query}")
                                st.write(f"**A:** {pydantic_obj.response}")
                                st.caption(f"**Source:** {pydantic_obj.source}")

                        # --- Fallback Pydantic ---
                        else:
                            st.subheader(f"Agent: {task_out.agent}")
                            st.json(pydantic_obj.model_dump())

                    except Exception:
                        st.subheader(f"Agent: {task_out.agent}")
                        st.info(task_out.raw)
            
            # --- Fallback Output Display ---
            if res.get("fallback_output"):
                st.markdown("---")
                st.markdown("### 🤖 Agent: Internet Connected Agent (Fallback)")
                if hasattr(res["fallback_output"], "pydantic"):
                    fb_pydantic = res["fallback_output"].pydantic
                    if isinstance(fb_pydantic, InternetOutput):
                        with st.container(border=True):
                                st.info(f"**Q:** {fb_pydantic.query}")
                                st.write(f"**A:** {fb_pydantic.response}")
                                st.caption(f"**Source:** {fb_pydantic.source}")
                    else:
                         st.json(fb_pydantic.model_dump())
