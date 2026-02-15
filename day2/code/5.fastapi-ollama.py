from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
import uvicorn

app = FastAPI()

# 请求模型
class Query(BaseModel):
    prompt: str
    model: str = "deepseek-r1:1.5b"
    stream: bool = False
    temperature: float = 0.7

@app.post("/api/chat")
async def chat(query: Query):
    if query.stream:
        return StreamingResponse(query_ollama_stream(query.prompt, query.model), media_type="text/event-stream")
    else:
        return query_ollama(query.prompt, query.model)

# 一次性返回结果
def query_ollama(prompt, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception("请求Error: " + response.text)
    return response.json()["response"]

# 流式返回结果
def query_ollama_stream(prompt, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    with requests.post(url, json=data, stream=True) as response:
        if response.status_code != 200:
            raise Exception("请求Error: " + response.text)
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "response" in obj:
                    yield obj["response"]
                if obj.get("done"):
                    break
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

