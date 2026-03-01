import os
from langchain_community.chat_models import ChatTongyi
import dashscope
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

print(f"Testing ChatTongyi with deepseek-v3...")
try:
    llm = ChatTongyi(model_name="deepseek-v3", dashscope_api_key=api_key)
    # Try simple invoke
    print("Invoking...")
    resp = llm.invoke("Hello")
    print(f"Response: {resp}")
    
    # Try stream
    print("Streaming...")
    for chunk in llm.stream("Hello"):
        print(f"Chunk: {chunk}")

except Exception as e:
    print(f"Error: {e}")
