import requests
import json

# 一次性输出
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

response = query_ollama("帮我写一个二分查找法")
print(response)


# 分批次输出
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


for chunk in query_ollama_straem("帮我写一个二分查找法"):
    print(chunk, end="", flush=True)
