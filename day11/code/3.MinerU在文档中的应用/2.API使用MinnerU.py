import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("MINERU_API_KEY")
url = "https://mineru.net/api/v4/extract/task"
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}
data = {
    "url": "https://vl-image.oss-cn-shanghai.aliyuncs.com/Qwen3-tech_report.pdf",
    "model_version": "vlm",
    'is_ocr':True,
    'enable_formula': False,
}

# 提交任务
res = requests.post(url,headers=header,json=data)
print(res.status_code)
print(res.json())
print(res.json()["data"])


# 获取任务进度
task_id = res.json()["data"]['task_id']
print(task_id)
url = f'https://mineru.net/api/v4/extract/task/{task_id}'
header = {
    'Content-Type':'application/json',
    "Authorization":f"Bearer {token}".format(token)
}

res = requests.get(url, headers=header)
print(res.status_code)
print(res.json())
print(res.json()["data"])
