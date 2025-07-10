from crewai import Agent

def planner_agent():
    return Agent(
        role="Research Planner",
        goal="Break down the research task into 3-5 subtopics for deeper exploration.",
        backstory="Expert in academic research and content planning.",
        verbose=True
    )
