## Day 6 总结

主要是在「Text-to-SQL」应用开发上，从基础的模型直接生成，进阶到使用 Agent 进行自主探索与纠错，最后到使用 Vanna 框架构建具备 RAG 能力的生产级 SQL 助手。大概可以分成三条线：基础模型生成与评测、LangChain Agent 智能体、以及 Vanna 框架应用。

### 记忆卡片
- **Text-to-SQL 范式**：从简单的 "Prompt -> SQL" 进化到 "Agent -> 查 Schema -> 写 SQL -> 纠错" 的智能模式，再到 "RAG -> 检索相似 DDL/SQL -> 生成" 的增强模式。
- **SQL Agent**：利用 LangChain 的 `SQLDatabaseToolkit` 和 `create_sql_agent`，让模型具备"行动能力"，能先查询表结构 (Schema) 再写 SQL，遇到错误还能自动修正 (Self-correction)。
- **Vanna 框架**：一个专门为 SQL 生成设计的 RAG 框架。它通过向量化存储 DDL（建表语句）、文档 (Documentation) 和 问答对 (SQL QA)，显著提升了模型对特定数据库的理解能力，解决了通用模型不识"私有表结构"的问题。
- **自动化评测**：验证 SQL 生成质量最直接的方法是 Execution Accuracy (执行准确率)，即把生成的 SQL 扔进数据库跑一下，看能不能跑通并返回结果。

### 理解
- **基础模型生成与评测 (`1.sql-compilot`)**：
  - `1.用Chat模型来写SQL查询.py`：使用通用的 Chat 模型 (如 Qwen-Turbo) 通过 Prompt Engineering 生成 SQL。
  - `2.用Coder模型来写SQL查询.py`：使用专门的代码模型 (如 DeepSeek-Coder) 生成 SQL，通常在语法准确性上表现更好。
  - `3.两种模型写SQL查询的结果评测.py`：一个自动化测试脚本，读取生成的 SQL，在真实数据库中执行，统计成功率，是评估模型效果的"金标准"。
- **LangChain SQL Agent (`2.sql-langchain`)**：
  - `1.langchain-sql_agent-deepseek.py`：使用 DeepSeek 模型构建 SQL Agent。Agent 不再是瞎猜字段，而是会先执行 `sql_db_list_tables` 和 `sql_db_schema` 工具去"看"数据库结构，再生成查询。
  - `2.langchain-sql_agent-健康险.py`：在复杂的健康险业务数据库上实战。展示了 Agent 如何处理"查询未支付保费"这种需要多表关联的复杂业务问题。
- **Vanna 框架应用 (`3.sql-vanna`)**：
  - `1.vanna-mysql-base-persistent.py`：核心脚本。实现了 Vanna 的**持久化存储**。它将数据库的 DDL 和业务知识向量化存入 ChromaDB，即使程序重启，训练过的"记忆"依然存在。还展示了如何通过 `vn.train` 注入"业务文档"和"Golden SQL"来修复检索不到表的问题。
  - `2.vanna-mysql-web.py`：在持久化的基础上，启动了一个 Flask Web 服务。提供了一个可视化的聊天界面，用户可以用自然语言查询，不仅能看到 SQL，还能直接看到查询结果的图表，是 Text-to-SQL 的完整落地形态。
