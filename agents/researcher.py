from crewai import Agent
from langchain.tools import Tool as LangchainTool

def researcher_agent(rag_chain):
    rag_tool = LangchainTool.from_function(
        name="RAG Research Tool",
        func=rag_chain.run,
        description="Retrieves academic knowledge using vector similarity and LLM",
    )

    return Agent(
        role="Scientific Researcher",
        goal="Retrieve top research summaries using RAG and generate contextual insights.",
        backstory="LLM agent with access to a vector-based RAG system using ArXiv papers.",
        tools=[rag_tool], 
        verbose=True
    )
