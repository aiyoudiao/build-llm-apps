## Day 3 总结：对比三款 AI Coding 工具，探索从 Excel 处理到数据可视化大屏

主要是「用 AI 帮我处理数据和画图」，不再是纯粹的对话模型，而是实打实的生产力工具。主要干了三件事：Excel 自动化处理、疫情数据大屏（Flask）、病床使用率大屏（Streamlit）。而且还顺便对比了一下 Cursor、Trae 和 Lingma 这些 AI 编程助手的表现。

### 记忆卡片

- **Excel 自动化**：用 `pandas` 代替 VLOOKUP，自动读取、合并多个 Excel 表格（比如把绩效表合进员工信息表）。
- **疫情数据大屏（Flask 版）**：用 Flask 做后端 API，ECharts 做前端展示，画折线图和地图，适合需要自定义网页样式的场景。
- **病床使用率大屏（Streamlit 版）**：用 Streamlit 纯 Python 写 Web 应用，配合 Plotly 画交互图，开发速度极快，适合数据分析师。
- **工具对比**：同一个需求（Excel 合并、疫情大屏），试了 Cursor、Trae、Lingma，发现它们在代码生成和理解需求上各有千秋，Cursor 和 Trae 国际版 最优，Lingma 其次。

### 理解

- **Excel 处理部分**：

  - 核心是 `pandas.merge`，相当于 SQL 的 JOIN 或者 Excel 的 VLOOKUP。
  - 练手场景：把《员工基本信息表》和《员工绩效表》按工号拼起来，生成一张总表。
  - 对应的代码在 `1.excel_merge_*` 目录下。
- **疫情数据大屏（Flask + ECharts）**：

  - 这是一个标准的前后端分离雏形。
  - 后端 `app.py` 用 Flask 读 Excel 数据，算出每日新增确诊，通过 API 返回 JSON。
  - 前端 `index.html` 用 ECharts 接数据，画出漂亮的折线图和香港地图分布。
  - 对应的代码在 `2.dashboard_epidemic_*` 目录下。
- **医院病床使用率大屏（Streamlit + Plotly）**：

  - 这是一个更现代的「数据应用」开发方式。
  - 不需要写 HTML/CSS/JS，直接在 Python 里写 `st.title`、`st.plotly_chart` 就能出网页。
  - 做了病床使用率分析、空闲病床分布图，还加了侧边栏筛选功能。
  - 对应的代码在 `3.hospital_bed_usage` 目录下。
- **“怎么自动处理 Excel”** → `1.excel_merge_cursor/merge_excel.py`
- **“怎么用 Python 做网页大屏”** → `2.dashboard_epidemic_trae/app.py` (Flask) 或 `3.hospital_bed_usage/dashboard.py` (Streamlit)
