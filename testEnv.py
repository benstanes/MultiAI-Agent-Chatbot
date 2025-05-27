import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in the environment variables.")

client = OpenAI(api_key=openai_api_key)


try:
    response = llm.invoke("Hey Is the connection working? to gpt -4o-mini")
    print("Response from GPT-4o-mini:")
    print(response.content)
except Exception as e:
    print("Error communicating with OpenAI:", e)
