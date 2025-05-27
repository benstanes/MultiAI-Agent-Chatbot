from typing import Annotated, Sequence, List, Literal 
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults 
from langgraph.types import Command 
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent 
from IPython.display import Image, display 
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from registry import getLlm
from Tools import WebSearchTool


def research_node(state: MessagesState) -> Command[Literal["supervisor"]]:

    """
        Research_Agent node that gathers information using Tavily search.
        Takes the current task state, performs relevant research,
        and returns findings for supervisor.
    """
    
    research_agent = create_react_agent(
        getLlm("gpt-4o-mini"),  
        tools=[WebSearchTool.tavily_search],  
        state_modifier= """
                    You are an expert research assistant.
                    Your task is to:
                        1. Use web search tools to find relevant and reliable information based on the user's query.
                        2. Carefully review the top 3 search results.
                        3. Extract the most important facts, evidence, or insights.
                        4. Summarize the findings in a concise, neutral, and well-organized paragraph.
                    Do not include your opinions or answer the query directly. Only focus on gathering accurate and relevant information from reliable sources.
                     """
    )

    result = research_agent.invoke(state)

    print(f"--- Workflow Transition: Researcher → Supervisor ---")

    return Command(
        update={
            "messages": [ 
                HumanMessage(
                    content=result["messages"][-1].content,  
                    name="Research_Agent"  
                )
            ]
        },
        goto="supervisor", 
    )

