## Day 7 总结

主要是在「LangChain Agent」的深度应用与实战上，从基础的 Prompt Template 和 ReAct Agent，进阶到 LCEL（LangChain Expression Language）的声明式编排，最后通过一个真实的「网络故障诊断 Agent」展示了如何让大模型调用系统级工具解决实际问题。大概可以分成三条线：基础组件与记忆、LCEL 工具组合与迁移、以及生产级运维 Agent 实战。

### 记忆卡片
- **ReAct 范式**：Reasoning + Acting。模型通过 "思考-行动-观察" 的循环来解决复杂问题。它不仅仅是回答问题，而是能够使用工具获取外部信息。
- **LCEL (LangChain Expression Language)**：LangChain 的新标准，通过 `chain = prompt | llm | parser` 的声明式语法，让构建复杂的 AI 流水线变得像搭积木一样简单直观。
- **Agent 迁移**：从旧版的 `initialize_agent` 迁移到新版的 `create_react_agent`，强调了使用 `AgentExecutor` 来管理循环和容错。
- **Context Passing (上下文传递)**：在复杂的 Agent 任务中，通过在工具间传递一个共享的 JSON Context，解决了多步操作中信息丢失的问题，让 Agent 具备了"状态保持"能力。

### 理解
- **基础组件与记忆 (`1.langchian-use`)**：
  - `1.LLM和提示词模版.py`：最基础的 LangChain 用法，展示了 `PromptTemplate` 和 `LLM` 的组合，用于生成公司名称。
  - `2.LLM和Agent_tools-serpapi.py`：引入了 **Agent** 的概念。展示了如何让模型调用 `SerpAPI` 进行联网搜索，获取实时信息（如“历史上的今天”）。
  - `3.LLM和Agent_tools-llm-math.py`：展示了多工具组合。Agent 可以同时使用搜索工具和数学计算工具 (`llm-math`)，解决包含计算的复杂问题。
  - `4.保留多伦对话的记忆.py`：演示了如何给 Chain 加上“记忆”。使用 `RunnableWithMessageHistory` 和 `InMemoryChatMessageHistory`，让 AI 能够记住上下文，进行连续对话。
  - `5.模拟特斯拉智能助手.py`：一个定制化的 ReAct Agent 案例。展示了如何通过自定义 `Tool`（查询车型、查询公司信息）和自定义 Prompt，构建一个专注于特定领域的智能助手。

- **LCEL 工具组合与迁移 (`2.langchain-tool-composite`)**：
  - `0.声明式链式表达语法LCEL.py`：LCEL 的入门示例。展示了如何用管道符 `|` 串联 Prompt、LLM 和 OutputParser，构建一个"翻译-分析-回译"的处理流。
  - `1.多功能文本与数据处理Agent-基础自主.py`：展示了如何将旧版的“面向对象”工具类适配到新版的 LangChain 架构中。通过 `@tool` 装饰器和 `create_react_agent`，让 Agent 自主选择工具进行文本分析、数据转换等任务。
  - `2.多功能文本与数据处理Agent-手动调度.py`：展示了 LCEL 的另一种用法——**手动路由**。不依赖 LLM 进行决策，而是通过代码逻辑（Router Pattern）精确控制工具的调用，适合业务逻辑固定的场景。
  - `3.多功能文本与数据处理Agent-高级自主.py`：进一步增强了 Agent 的鲁棒性。引入了 `ReActJsonSingleInputOutputParser`，强制模型输出 JSON 格式的 Action，解决了多参数传递的准确性问题，并优化了情感分析算法。

- **网络故障诊断 Agent 实战 (`3.network-fault-diagnosis-agent`)**：
  - `1.网络故障诊断智能助手.py`：一个**真实执行**的 Agent。它不再是“纸上谈兵”，而是通过 `subprocess` 调用真实的系统命令（Ping, NSLookup, Ifconfig, Grep），能够诊断真实的网络连通性问题。包含了命令注入防护等安全机制。
  - `2.高级网络故障诊断智能助手.py`：架构升级版。引入了**上下文传递 (Context Passing)** 机制，所有工具共享并更新同一个 JSON 对象。这使得 Agent 能够进行复杂的多步诊断（如：先提取目标 -> 再 DNS 解析 -> 最后 Ping 解析出的 IP），形成了完整的证据链。
