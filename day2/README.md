## Day 2 总结

主要是在「怎么用大模型」这件事上，从最简单的云端调用，一路走到本地模型和自己的 API 服务。大概可以分成三条线：云端模型、提示词、以及本地部署 + 接口封装。

### 记忆卡片
- 云端调模型：dashscope + deepseek，熟悉 messages（system / user）和基本对话、情感分析。
- 提示词：把“角色 / 目标 / 规则 / 输出格式”说清楚，用 JSON 和 CoT 让结果更可控。
- 本地模型：modelscope 下载 DeepSeek，transformers 加载，选好设备，体验一次性生成和流式输出。
- Ollama：把本地模型变成统一的 HTTP 接口，支持一次性和流式两种调用方式。
- FastAPI + 客户端：封装 `/api/chat`，用 StreamingResponse 做流式返回，再用小脚本从“用户视角”测试整条链路。

### 理解
- 云端部分：用 dashscope 调 deepseek，先学会把话说清楚（messages 怎么写，system / user 怎么配），顺手做了聊天和情感分析两个小练习。
- 提示词部分：不再是“随便问问”，而是把任务拆开设计——谁在说话（角色），要干嘛（目标），有什么规则，最后想要什么格式的输出（比如 JSON），还试了 CoT 这种“请一步一步分析”的写法。
- 本地模型部分：把 DeepSeek 模型搬到本地，用 modelscope 下模型、用 transformers 加载，选好设备（cuda/mps/cpu），然后体验了一次性生成和流式输出的区别。
- Ollama 部分：把本地模型包成一个统一的 HTTP 接口，自己写了函数去调一次性结果和流式结果，感觉就像在调一个普通的服务。
- FastAPI 部分：再往外包一层，用 `/api/chat` 把 Ollama 暴露出去，前端或者别的服务只需要管这个接口；流式输出用 StreamingResponse 包了一下。
- 客户端部分：写了一个小脚本，负责“从用户角度”验证这个 API，分别测一次性输出和流式输出，也顺便熟悉了流式结果在客户端怎么读。

- “云端怎么调大模型” →  `1.推理型模型.py` 和 `2.提示词工程.py`。
- “本地怎么跑模型” →  `3.使用本地模型_transformers.py`。
- “怎么对外提供统一接口” →  `4.使用本地模型_ollama.py`、`5.fastapi-ollama.py` 和 `5.1.fastapi-ollama_客户端.py`，在脑子里过一遍：Ollama ←→ FastAPI ←→ 客户端。
