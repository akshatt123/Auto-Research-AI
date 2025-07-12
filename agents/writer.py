from crewai import Agent

def writer_agent(llm):
    return Agent(
        role="Technical Writer",
        goal="Generate a well-formatted markdown report using citations.",
        backstory="An AI agent trained to write academic research reports.",
        verbose=True,
        llm=llm
    )
