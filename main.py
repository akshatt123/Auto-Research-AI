from dotenv import load_dotenv
load_dotenv()

from crewai import Crew, Task

from tools.arxiv_rag import fetch_arxiv_papers, build_vectorstore, build_rag_chain
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.writer import writer_agent

def main():
    topic = "Open-source LLM Alignment Techniques"

    #Fetch and embed papers
    papers = fetch_arxiv_papers(topic)
    vectorstore = build_vectorstore(papers)
    rag_chain = build_rag_chain(vectorstore)

    # Initialize agents
    planner = planner_agent()
    researcher = researcher_agent(rag_chain)
    writer = writer_agent()

    # Define tasks
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

    # Run crew
    crew = Crew(
        agents=[planner, researcher, writer],
        tasks=[task_plan, task_research, task_write],
        verbose=True
    )

    crew.kickoff()

if __name__ == "__main__":
    main()
