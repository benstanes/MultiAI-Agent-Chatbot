from langchain.agents import initialize_agent, AgentType
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool


noop_tool = Tool.from_function(
    name="NoOp",
    func=lambda x: "No operation",
    description="A dummy tool that does nothing"
)

class BaseLLMAgent:
    
    def __init__(self, name: str, llm: BaseLanguageModel, tools=[], system_prompt: str = None):
        self.name = name
        self.system_prompt = system_prompt or f"You are {name}."
        if(len(tools)==0):
            tools=[noop_tool]
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "system_message": self.system_prompt
            }
        )

    def run(self, input_text: str) -> str:
        return self.agent.run(input_text)