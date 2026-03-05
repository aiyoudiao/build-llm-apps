## Day 8 总结

主要是在「Function Calling」原理与「Qwen-Agent」框架的深度应用与实战上，从手动实现 Function Calling 的底层循环，进阶到使用 Qwen-Agent 框架构建标准化智能体，最后通过一系列复杂的「门票与餐饮数据分析 Agent」展示了 Text-to-SQL、自动可视化以及基于机器学习（归因分析、决策树）的深度数据挖掘能力。大概可以分成三条线：Function Calling 原理、Qwen-Agent 框架应用、以及高级数据分析智能体实战。

### 记忆卡片
- **Function Calling (函数调用)**：大模型连接外部世界的桥梁。模型不再只输出文本，而是输出包含函数名和参数的结构化数据 (JSON)，由宿主程序执行后将结果回传给模型。
- **Qwen-Agent 框架**：阿里云推出的 Agent 开发框架。提供了 `Assistant`、`GroupChat` 等高级抽象，支持 `@register_tool` 装饰器注册工具，并内置了 WebUI 和 TUI 交互界面。
- **MCP (Model Context Protocol)**：一种连接 AI 模型与外部数据/工具的开放标准协议。允许 Agent 以标准化的方式调用本地或远程的服务（如 Node.js 编写的高德地图服务）。
- **Text-to-SQL**：将自然语言问题转化为 SQL 查询语句的技术，是数据分析 Agent 的核心能力之一。
- **归因分析 (Attribution Analysis)**：利用机器学习模型（如线性回归、决策树）来量化不同因素（如天气、活动、客群）对业务指标（如营收）的贡献度或影响权重。

### 理解
- **Function Calling 原理 (`1.function-calling`)**：
  - `1.LLM智能天气助手.py`：**手动实现**了 ReAct 循环。通过 `dashscope` SDK，演示了如何构造 `tools` schema，如何解析模型返回的 `tool_calls`，如何执行本地 Python 函数，以及如何将执行结果封装为 `tool` 角色消息回传给模型，完成一次完整的“思考-行动-观察-回答”闭环。
  - `2.LLM天气智能体原理解析.py`：对上述过程的详细拆解与注释，适合理解底层原理。

- **Qwen-Agent 框架应用 (`2.qwen-agent`)**：
  - `1.单模天气对话助手.py`：使用 `qwen-agent` 框架重构了天气助手。展示了如何通过继承 `BaseTool` 和使用 `@register_tool` 装饰器优雅地定义工具，以及如何使用 `Assistant` 类快速构建支持 WebUI 和 TUI 的智能体。
  - `2.双模智能天气助手.py`：展示了**混合工具调用**的能力。Agent 同时挂载了本地 Python 工具（`WeatherTool`）和通过 **MCP 协议** 连接的外部 Node.js 服务（`amap-maps-mcp-server`），体现了 Agent 的扩展性。

- **高级数据分析智能体实战 (`3.ticket-agent`)**：
  - `1.门票数据智能分析助手.py`：基础的 **Text-to-SQL** Agent。能够理解自然语言问题（如“查询各省份销量”），自动生成 SQL 查询 MySQL 数据库，并返回 Markdown 格式的数据表格。
  - `2.门票数据智能分析与可视化助手.py`：在 SQL 查询的基础上增加了**自动可视化**能力。智能体能够根据查询结果的数据类型（数值列 vs 分类列），自动调用 `matplotlib` 绘制柱状图，实现“图表 + 数据”的双模态输出。
  - `3.门票数据智能透视与可视化助手.py`：可视化能力的进阶版。引入了**智能透视 (Pivot Table)** 逻辑，能够自动识别多维度数据（如“不同月份的不同票种销量”），绘制**堆积柱状图 (Stacked Bar)**，并解决了 Matplotlib 中文乱码和特殊符号转义的问题。
  - `4.餐饮营收智能洞察与归因分析助手.py`：**全天最强 Agent**。它不再局限于查询，而是具备了**深度分析**能力：
    - **线性回归 (Linear Regression)**：用于归因分析，量化不同客群（年卡、散客、促销）对总营收的边际贡献。
    - **决策树 (CART)**：用于关键驱动因素分析，自动构建决策树模型，找出影响营收的核心因子（如天气、节假日、活动），并生成可视化的树状图。
    - **动态绘图**：支持执行大模型生成的 Python 绘图代码，满足个性化的可视化需求。
