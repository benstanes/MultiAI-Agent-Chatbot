from langchain_community.tools.tavily_search import TavilySearchResults 
import os
from langchain.tools import Tool
from dotenv import load_dotenv

# load_dotenv()


# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_search = TavilySearchResults(max_results=2)


