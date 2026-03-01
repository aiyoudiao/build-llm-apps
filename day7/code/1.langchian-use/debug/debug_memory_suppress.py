import os
import logging
# Suppress the specific warning from langchain_core
logging.getLogger("langchain_core.tracers.context").setLevel(logging.ERROR)

from typing import Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('DASHSCOPE_API_KEY')
import dashscope
dashscope.api_key = api_key

llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helper."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("Invoking...")
try:
    res = conversation_chain.invoke(
        {"input": "Hi"},
        config={"configurable": {"session_id": "test_1"}}
    )
    print(res.content)
    
    res = conversation_chain.invoke(
        {"input": "What did I say?"},
        config={"configurable": {"session_id": "test_1"}}
    )
    print(res.content)
except Exception as e:
    print(f"Error: {e}")
