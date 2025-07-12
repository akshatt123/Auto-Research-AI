from crewai import Agent
from crewai.tools import tool

def researcher_agent(rag_chain,llm):
    @tool("RAG Research Tool")
    def rag_tool(topic: str) -> str:
        """
        Uses RAG to retrieve and generate summarized answers for a given research topic.
        """
        return rag_chain.run(topic)

    return Agent(
        role="Scientific Researcher",
        goal="Retrieve top research summaries using RAG and generate contextual insights.",
        backstory="LLM agent with access to a vector-based RAG system using ArXiv papers.",
        tools=[rag_tool],
        verbose=True,
        llm=llm
    )
