import streamlit as st
import os
import sys
from pathlib import Path
import time
from crewai import Crew, Task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.writer import writer_agent
from tools.arxiv_rag import (
    fetch_arxiv_papers,
    build_vectorstore,
    build_rag_chain,
    agent_llm,
    build_vectorstore_from_pdf_bytes,
)

# Load OpenAI key from environment
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="AutoResearcher.ai", page_icon="📚", layout="centered")

st.title("🤖 AutoResearcher.ai - Agentic RAG Research Assistant")
st.markdown("Generate academic research reports using AI agents, RAG, and LLMs.")

# Preflight: API key check to avoid opaque LLM failures
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
generate_disabled = False
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in environment. Add it to your .env or system environment variables.")
    generate_disabled = True

# Normalize Crew output to a Markdown string
def crew_output_to_markdown(result_obj) -> str:
    try:
        if isinstance(result_obj, str):
            return result_obj
        # Common string-like attributes in CrewAI outputs
        for attr_name in [
            "raw",
            "output",
            "final_output",
            "result_text",
            "text",
        ]:
            if hasattr(result_obj, attr_name):
                attr_value = getattr(result_obj, attr_name)
                if isinstance(attr_value, str):
                    return attr_value
        # Methods that may produce strings
        if hasattr(result_obj, "to_string"):
            try:
                return result_obj.to_string()
            except Exception:
                pass
        if hasattr(result_obj, "json"):
            try:
                import json  # local import to avoid top-level dependency during app startup
                return json.dumps(result_obj.json(), ensure_ascii=False, indent=2)
            except Exception:
                pass
        # Fallback
        return str(result_obj)
    except Exception:
        return str(result_obj)

tab1, tab2 = st.tabs(["ArXiv Topic", "Upload PDF"])

with tab1:
    # Input topic
    topic = st.text_input("Enter a Research Topic")

    if st.button("🧠 Generate Report", disabled=generate_disabled, key="btn_topic"):
        with st.spinner("Fetching papers, planning subtopics, running agents..."):
            try:
                papers = fetch_arxiv_papers(topic)
                if not papers:
                    st.error("No papers found. Try a different topic.")
                else:
                    vectorstore = build_vectorstore(papers)
                    rag_chain = build_rag_chain(vectorstore)

                    # Agents
                    planner = planner_agent(agent_llm)
                    researcher = researcher_agent(rag_chain, agent_llm)
                    writer = writer_agent(agent_llm)

                    # Tasks
                    task_plan = Task(
                        description=f"Create 3-5 subtopics from the main topic: '{topic}'.",
                        expected_output="List of subtopics",
                        agent=planner
                    )

                    task_research = Task(
                        description="For each subtopic, use the RAG chain to summarize relevant research papers.",
                        expected_output="List of summarized citations per subtopic",
                        agent=researcher
                    )

                    task_write = Task(
                        description=f"Write a full research report on: '{topic}' with citations in Markdown format.",
                        expected_output="A well-structured Markdown report with citations.",
                        agent=writer
                    )

                    # Crew
                    crew = Crew(
                        agents=[planner, researcher, writer],
                        tasks=[task_plan, task_research, task_write],
                        verbose=True,
                        llm=agent_llm
                    )

                    result = crew.kickoff()
                    report_md = crew_output_to_markdown(result)

                    st.success("✅ Report generated!")
                    st.markdown(report_md)

                    # Download directly from memory (no file saved locally)
                    st.download_button(
                        label="📄 Download Markdown Report",
                        data=report_md.encode("utf-8"),
                        file_name="AutoResearcher_Report.md",
                        mime="text/markdown",
                    )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with tab2:
    uploaded_pdf = st.file_uploader("Upload a research paper (PDF)", type=["pdf"]) 
    user_question = st.text_input("Ask a question about the uploaded paper")
    if st.button("🔎 Ask", disabled=generate_disabled or not uploaded_pdf, key="btn_pdf_ask"):
        if not uploaded_pdf:
            st.error("Please upload a PDF first.")
        elif not user_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Building context from PDF and answering..."):
                try:
                    pdf_bytes = uploaded_pdf.read()
                    vectorstore_pdf = build_vectorstore_from_pdf_bytes(pdf_bytes)
                    rag_chain_pdf = build_rag_chain(vectorstore_pdf)
                    answer = rag_chain_pdf.run(user_question)
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

