import requests
import json

def test_chat():
    """测试一次性输出"""
    url = "http://localhost:8000/api/chat"
    data = {
        "prompt": "帮我写一个二分查找法",
        "model": "deepseek-r1:1.5b",
        "stream": False
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            content = response.text
            try:
                content = json.loads(content)
            except:
                pass

            print(content)
        else:
            print(f"请求Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"请确保 5.fastapi-ollama.py 已运行: {e}")


def test_chat_stream():
    """测试流式输出"""
    url = "http://localhost:8000/api/chat"
    data = {
        "prompt": "帮我写一个二分查找法",
        "model": "deepseek-r1:1.5b",
        "stream": True
    }

    try:
        with requests.post(url, json=data, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_content(chunk_size=None):
                    if line:
                        print(line.decode("utf-8"), end="", flush=True)
            else:
                print(f"请求Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"请确保 5.fastapi-ollama.py 已运行: {e}")

print("测试一次性输出：")
test_chat()
print()
print("测试流式输出：")
test_chat_stream()
