from Agents.BaseAgent import BaseLLMAgent
from langchain_openai import ChatOpenAI
from registry import getLlm
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

def reasoning_node(state: MessagesState) -> Command[Literal["supervisor"]]:

    """
        Reasoning_Agent node that Reads the Summary from the web search and answer the User query.
        Takes the original user input and Search summary from Reaserch Agent and tries to answer the question along with reason, Pass the answer to the supervisor
    """
   
    system_prompt = (
       """You are an expert analyst and critical thinker.
            Your task is to:
            1. Read and understand the user original query.
            2. Analyze the summary of the research provided by the Researcher Agent.
            3. Use this information to generate a clear, accurate, and helpful response to the query.
            Focus on logical reasoning, clarity, and completeness. If there are uncertainties, mention them. Your goal is to provide the best possible answer based on the summary.
            """
    )
    llm = getLlm("gpt-4o-mini")

    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"]  
   
    enhanced_query = llm.invoke(messages)

    print(f"--- Workflow Transition: Prompt Reasoner → Supervisor ---")

    return Command(
        update={
            "messages": [  
                AIMessage(
                    content=enhanced_query.content, 
                    name="Reasoning_Agent"  
                )
            ]
        },
        goto="supervisor", 
    )

