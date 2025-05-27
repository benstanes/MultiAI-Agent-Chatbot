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



class Supervisor(BaseModel):
    next: Literal["Research_Agent", "Reasoning_Agent", "Chat_Styler"] = Field(
        description="Determines which Agent to activate next in the workflow sequence: "
                    "'Research_Agent' when additional facts, context, or data collection is necessary, "
                    "'Reasoning_Agent' when Analyzing collected information and synthesizes a final answer, "
                    "'Chat_Styler' When answer anddresses the core intent of the question ,even if not perfectly and answer need to rephrased before giving to user"
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )
    

def supervisor_node(state: MessagesState) -> Command[Literal["Research_Agent", "Reasoning_Agent","Chat_Styler"]]:

    system_prompt = ('''
                 
You are the Supervisor Agent responsible for orchestrating a multi-agent system to collaboratively answer user queries.

Your responsibilities:
1. Receive the user's query and any ongoing context or intermediate results from other agents.
2. Decide which agent among the available agents should handle the next step to best progress towards answering the query.
3. Call the selected agent with a clear, concise prompt or task.
4. Collect the agent's output and analyze it to determine whether:
   - The query is fully answered (then end the process), or
   - Further processing by another agent is required.
5. Route the workflow accordingly until completion.

Available agents:
- Research_Agentt: Performs in-depth web searches and gathers reliable information summaries.
- Reasoning_Agent: Analyzes collected information and synthesizes a final answer.
- Chat_Styler: Rephrases the Answer to be friendly and in 2nd person tone.

When routing:
- Always summarize and structure outputs from agents clearly.
- Keep track of which agents have already been involved to avoid repetition.
- If output is ambiguous or incomplete, decide whether to call another agent or request clarification.
- If the Answer is correct and valid, Give the task of rephrasing and giving answer to the user to Chat_Styler Agent.

Maintain clear and logical control flow at all times.
    ''')
    llm = getLlm("gpt-4o-mini")
    
    
    formatted_messages = []
    for msg in state["messages"]:
        if isinstance(msg, tuple):
            formatted_messages.append({"role": msg[0], "content": msg[1]})
        else:
            formatted_messages.append(msg)
    messages = [{"role": "system", "content": system_prompt}] + formatted_messages


    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")
    
    return Command(
        update={
            "messages": [
                AIMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,  
    )