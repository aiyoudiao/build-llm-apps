import os
from langchain_community.chat_models import ChatTongyi
from langchain.agents import load_tools
import dashscope
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

print(f"Testing ChatTongyi with deepseek-v3 and tools...")
try:
    llm = ChatTongyi(model_name="deepseek-v3", dashscope_api_key=api_key)
    tools = load_tools(["llm-math"], llm=llm)
    print("Tools loaded.")
    
    # Test llm-math tool
    math_tool = tools[0]
    print(f"Testing tool: {math_tool.name}")
    res = math_tool.run("What is 100 divided by 4?")
    print(f"Tool result: {res}")

except Exception as e:
    print(f"Error: {e}")
