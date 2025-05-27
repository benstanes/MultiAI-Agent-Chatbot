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

def chat_styler_node(state: MessagesState) -> Command[Literal["__end__"]]:

    """
    Chat_Styler Node gets the answer which is accepted by the supervisor, and rephrase it into a friendly second person tone.
    """
    
    research_agent = create_react_agent(
        getLlm("gpt-4o-mini"),  
        tools=[],  
        state_modifier= """
                    You are expert user interactor.
                    Your task is to:
                        1. Get the Last Answer accepted by the supervisor
                        2. Rephrase the Last answer to answer the User in a Second person tone.
                        3. Your answer should be in friendly tone.
                     """
    )

    result = research_agent.invoke(state)

    print(f"--- Workflow Transition: Chat_Styler → FINISH ---")

    return Command(
        update={
            "messages": [ 
                AIMessage(
                    content=result["messages"][-1].content,  
                    name="Chat_Styler"  
                )
            ]
        },
        goto="FINISH", 
    )