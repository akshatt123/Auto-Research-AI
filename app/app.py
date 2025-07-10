import streamlit as st
import os
import time
from crewai import Crew, Task
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.writer import writer_agent
from tools.arxiv_rag import fetch_arxiv_papers, build_vectorstore, build_rag_chain

# Load OpenAI key from environment
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="AutoResearcher.ai", page_icon="📚", layout="centered")

st.title("🤖 AutoResearcher.ai - Agentic RAG Research Assistant")
st.markdown("Generate academic research reports using AI agents, RAG, and LLMs.")

# Input topic
topic = st.text_input("Enter a Research Topic", value="Open-source LLM Alignment Techniques")

if st.button("🧠 Generate Report"):
    with st.spinner("Fetching papers, planning subtopics, running agents..."):
        try:
            papers = fetch_arxiv_papers(topic)
            if not papers:
                st.error("No papers found. Try a different topic.")
            else:
                vectorstore = build_vectorstore(papers)
                rag_chain = build_rag_chain(vectorstore)

                # Agents
                planner = planner_agent()
                researcher = researcher_agent(rag_chain)
                writer = writer_agent()

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
                    verbose=True
                )

                result = crew.kickoff()

                # Save markdown file
                output_path = f"report_{int(time.time())}.md"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result)

                st.success("✅ Report generated!")

                # Download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📄 Download Markdown Report",
                        data=file,
                        file_name="AutoResearcher_Report.md",
                        mime="text/markdown"
                    )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
