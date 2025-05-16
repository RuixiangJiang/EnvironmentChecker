from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60,  # 增加超时时间
    max_retries=3
)

response = llm.invoke([
    SystemMessage(content="You are a pirate-themed coding assistant."),
    HumanMessage(content="How do I check if a Python object is an instance of a class?")
])

print(response.content)