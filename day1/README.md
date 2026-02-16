## Day 1 总结

主要是在「让大模型真正帮我干活」这条线上打地基，从最简单的情感分析，一路到 function call、tool call、读图表格，以及联网搜索。

### 记忆卡片
- 情感分析：给 deepseek-v3 设定“情感分析大师”的角色，让它判断一句话是正向还是负向。
- 天气 function call：用 qwen-max 的 function call 机制，让模型主动决定何时调用 `get_mock_weather`，实现“问天气 → 调函数 → 组合回复”的闭环。
- 图片表格提取：用 qwen3-vl-plus 多模态模型，传一张表格图片 URL，让模型只用 JSON 把表格结构提取出来。
- 运维 tool call：用 qwen-turbo 的 tool calls，让模型根据告警决定要不要调用 `get_mock_status`，再综合监控数据给决策建议。
- 联网搜索（deepseek）：用 deepseek-r1 + `enable_search=True`，让模型自己上网查当前时间等实时信息。
- 联网搜索（OpenAI 兼容）：用 OpenAI SDK 兼容模式去调 qwen-turbo，同样打开 `enable_search`，顺便熟悉了一下“OpenAI 写法 + 阿里云后端”的组合。

### 理解
- 情感分析这块，让我熟悉了最基础的 messages 写法（system + user），也体验了一把“换个 system 角色，任务就完全变了”。
- 天气 function call 这个例子，让我真正体会到：大模型不仅能“回答”，还能“决定要不要调某个函数”，而且 arguments 是它自己填的。
- 图片表格提取，让我知道多模态模型可以直接“读图”，而且只要在提示词里限制“只输出 JSON”，就能直接拿来当结构化数据用。
- 运维 tool call 这块更像是一个小型“自动化运维助手”：模型看到告警 → 觉得需要更多信息 → 调监控接口 → 根据返回值给出处理建议，全程通过 messages + tool_calls 串起来。
- deepseek 联网搜索和 qwen-turbo 联网搜索，一边是 dashscope 原生调用，一边是 OpenAI 兼容模式调用，本质上都在练习“怎么在提示词里问清楚问题 + 开启实时搜索”。

- “情感分析 / 基本对话” → `1.情感分析.py`。
- “function call + 天气示例” → `2.天气function-call.py`。
- “读图片中的表格数据” → `3.提取图片表格数据.py`。
- “运维场景下的 tool call” → `4.运维事件处理tool-call.py`。
- “deepseek 的联网搜索” → `5.deepseek 联网搜索.py`。
- “OpenAI 兼容模式 + qwen 联网搜索” → `6.openai 联网搜索.py`。
