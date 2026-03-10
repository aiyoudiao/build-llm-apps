# Build LLM Apps from Scratch

📍 从 0 到 1 搭建 LLM 应用。这是一个关于大语言模型应用开发的学习路径记录，涵盖了从基础调用到复杂的 Agent 系统、RAG 优化以及生产级应用实战。

---

## 📅 学习日志

### [Day 1: 让大模型真正帮我干活](./day1/README.md)
从最简单的情感分析开始，一路到 function call、tool call、读图表格，以及联网搜索。

> 🔗 **传送门**：[点击查看 Day 1 详细文档与代码](./day1/README.md)

- **情感分析**：设定角色进行文本情感判断。
- **Function Call**：让模型主动决定何时调用外部函数（如天气查询）。
- **多模态提取**：直接从图片中提取结构化表格数据。
- **Tool Call**：运维场景下的自动化决策与监控查询。
- **联网搜索**：分别使用 DeepSeek 和 OpenAI 兼容模式实现实时联网搜索。

### [Day 2: 模型调用与接口封装](./day2/README.md)
专注于“怎么用大模型”，从云端调用走到本地部署和 API 封装。

> 🔗 **传送门**：[点击查看 Day 2 详细文档与代码](./day2/README.md)

- **云端调用**：DashScope + DeepSeek 的基本对话与提示词设计。
- **提示词工程**：角色设定、目标明确、CoT（思维链）与 JSON 格式化输出。
- **本地部署**：使用 Transformers 加载本地模型，体验一次性与流式输出。
- **Ollama + FastAPI**：将本地模型封装为标准 HTTP 接口，并实现流式响应客户端。

### [Day 3: 生产力工具实战](./day3/README.md)
不再是纯粹的对话，而是构建实打实的数据处理与可视化应用。

> 🔗 **传送门**：[点击查看 Day 3 详细文档与代码](./day3/README.md)

- **Excel 自动化**：使用 Pandas 替代 VLOOKUP，自动合并与处理表格数据。
- **疫情数据大屏**：基于 Flask + ECharts 构建前后端分离的数据可视化应用。
- **病床使用率大屏**：使用 Streamlit + Plotly 快速构建交互式数据分析应用。
- **工具对比**：对比 Cursor、Trae、Lingma 在实际代码生成中的表现。

### [Day 4: 向量与推荐系统](./day4/README.md)
探索 NLP 的核心——向量（Embedding），从统计学方法到现代向量数据库。

> 🔗 **传送门**：[点击查看 Day 4 详细文档与代码](./day4/README.md)

- **TF-IDF 推荐**：基于统计学的传统文本推荐算法。
- **Word2Vec**：训练《西游记》等语料，理解词向量与语义相似度。
- **FAISS 向量数据库**：
  - 基础：IndexFlatL2 暴力搜索。
  - 进阶：IVF-PQ 倒排量化索引，处理海量数据的毫秒级检索。

### [Day 5: RAG 系统全栈优化](./day5/day5-1/README.md)
深入 RAG（检索增强生成）的各个环节，从基础构建到高级优化。

> 🔗 **传送门**：
> - [Day 5-1: 基础与多模态](./day5/day5-1/README.md)
> - [Day 5-2: 进阶与工程化](./day5/day5-2/README.md)

- **Day 5-1: 基础与多模态**
  - **Embedding 模型**：本地部署 BGE/GTE 模型。
  - **Chat PDF**：PDF 解析、切片、向量化与问答溯源。
  - **切片策略**：滑动窗口、语义切片、Markdown 层次切片等多种策略对比。
  - **多模态 RAG**：结合 OCR 与 CLIP 模型，实现以文搜图和图文问答。
- **Day 5-2: 进阶与工程化**
  - **Rerank 重排序**：使用 BGE-Reranker 提升检索精准度。
  - **召回率优化**：Multi-Query 策略，通过多角度提问扩大检索范围。
  - **Query 改写**：处理多轮对话中的指代消解与意图识别，智能路由联网搜索。
  - **知识库工程化**：自动生成 QA 对、知识沉淀、健康度检查与版本管理。

### [Day 6: Text-to-SQL 应用开发](./day6/README.md)
专注于让大模型操作数据库，从简单的 SQL 生成到智能 BI 助手。

> 🔗 **传送门**：[点击查看 Day 6 详细文档与代码](./day6/README.md)\
> 🔥 **Text-to-SQL 类项目落地**：[点击查看 Web3 Vanna Pro](https://github.com/aiyoudiao/web3-vanna-pro)

- **基础生成**：对比 Chat 模型与 Coder 模型（DeepSeek-Coder）的 SQL 生成效果。
- **SQL Agent**：利用 LangChain 让模型先查 Schema 再写 SQL，具备自动纠错能力。
- **Vanna 框架**：构建生产级 Text-to-SQL 应用，持久化存储 DDL 和业务知识，并通过 Flask 提供可视化查询界面。

### [Day 7: LangChain Agent 进阶与运维实战](./day7/README.md)
深入 LangChain Agent 的核心机制，并构建真实可执行的系统级 Agent。

> 🔗 **传送门**：[点击查看 Day 7 详细文档与代码](./day7/README.md)

- **基础组件**：Prompt Template、SerpAPI 联网、LLM-Math 计算。
- **LCEL 编排**：掌握 LangChain 新标准（声明式链式表达语法），对比自主 Agent 与手动路由（Router Pattern）。
- **工具组合**：将旧版工具类无缝迁移到新版架构，处理多参数复杂工具。
- **网络故障诊断 Agent**：
  - **真实执行**：调用 Ping, NSLookup, Ifconfig 等真实系统命令。
  - **Context Passing**：实现上下文传递机制，让 Agent 在多步诊断中保持状态，形成完整的证据链。

### [Day 8: Agent 框架与数据分析实战](./day8/README.md)
从手动实现 Function Calling 原理出发，进阶到使用 Qwen-Agent 框架构建标准化智能体，最终落地复杂的商业数据分析场景。

> 🔗 **传送门**：[点击查看 Day 8 详细文档与代码](./day8/README.md)

- **Function Calling 原理**：手动实现 LLM 与工具交互的完整闭环（思考-行动-观察）。
- **Qwen-Agent 框架**：
  - **单模/双模 Agent**：基于框架快速构建支持 WebUI 的智能体。
  - **MCP 协议**：混合调用本地 Python 工具与远程 Node.js 服务（高德地图）。
- **高级数据分析 Agent**：
  - **Text-to-SQL**：自然语言查库，自动生成 SQL。
  - **自动可视化**：智能识别数据维度，自动绘制柱状图/堆积图。
  - **归因分析**：利用线性回归量化客群贡献。
  - **关键因子挖掘**：利用决策树（CART）自动分析影响营收的核心因素（天气、节假日等）。

### [Day 9: Model Context Protocol (MCP) 与 Agent 协作](./day9/README.md)
深入探索「Model Context Protocol (MCP)」与「Agent-to-Agent (A2A)」智能体协作协议，打通智能体之间的“社交网络”。

> 🔗 **传送门**：[点击查看 Day 9 详细文档与代码](./day9/README.md)

- **MCP 服务端开发**：使用 `FastMCP` 快速构建本地服务。
  - **文件系统操作**：安全暴露本地桌面文件给 AI，实现统计与读取。
  - **内存知识库**：构建零依赖的开发规范查询服务，响应极快。
- **Agent 调用 MCP**：
  - **精准工具调度**：Qwen-Agent 作为 Client 自动调度本地 MCP 工具获取数据。
  - **混合架构 (Hybrid)**：同时挂载本地（高德地图 Node.js）与远程（ModelScope Bing Search & 网页 Fetch）MCP 服务，解决复杂生活问题。
- **Agent-to-Agent (A2A)**：基于标准协议的智能体协作。
  - **服务发现**：通过 `/.well-known/agent.json` 暴露能力名片，实现 Agent 互联。
  - **自动化决策**：构建“篮球活动决策代理”，自动发现并调用“天气代理”进行业务决策。

### [Day 10: 多架构智能体 (Multi-Architecture Agents)](./day10/README.md)
从反应式 (Reactive) 到深思熟虑 (Deliberative)，再到混合架构 (Hybrid) 的系统级智能体实战。

> 🔗 **传送门**：[点击查看 Day 10 详细文档与代码](./day10/README.md)

- **Reactive Agent (反应式)**：
  - **私募合规助手**：基于 LangChain 0.3.x / LangGraph 1.0+ 构建 ReAct Agent，精准查询法规，内置防幻觉机制。
- **Deliberative Agent (深思熟虑)**：
  - **深度投研智能体**：基于 Qwen-Agent / LangGraph 模拟分析师 SOP（感知→建模→推理→决策→报告），生成专业研报。
- **Hybrid Agent (混合架构)**：
  - **智能财富投顾**：构建三层架构系统，通过协调层自动判断意图，灵活切换“快思考”（查行情）与“慢思考”（做规划）模式。

---
*Last updated: 2026-03-11*
