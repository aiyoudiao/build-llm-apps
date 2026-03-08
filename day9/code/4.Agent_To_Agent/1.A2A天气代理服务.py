#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2A 天气代理服务 (A2A Weather Agent Service)

【功能概述】
本模块实现了一个遵循 A2A (Agent-to-Agent) 协议标准的微服务，专门用于向其他 AI Agent 提供天气查询能力。
它不是为人类用户设计的 Web 界面，而是作为“机器可发现、机器可调用”的基础设施组件。
核心特性包括：
1. 自动服务发现：通过标准路径 /.well-known/agent.json 暴露“Agent 名片”，供其他 Agent 自动检索能力。
2. 结构化任务交互：采用基于 Task ID 的异步任务模式，支持标准化的输入验证和结果返回。
3. 模拟数据引擎：内置本地天气数据库，用于演示协议交互流程（生产环境可替换为真实 API）。

【系统架构】
- 协议层：遵循 A2A 协议草案，定义 Agent 身份、输入 Schema 和认证方式。
- 接口层：基于 FastAPI 构建 RESTful 接口，处理任务提交与状态查询。
- 数据层：使用内存字典模拟天气数据，支持特定日期和地点的查询。

【核心流程】
1. [服务发现] 
   - 外部 Agent 访问 `/.well-known/agent.json`。
   - 本服务返回 `WEATHER_AGENT_CARD`，告知对方：我是谁、我能做什么、我的接口地址在哪、我需要什幺参数。
2. [任务提交] 
   - 外部 Agent 解析名片，构造符合 `input_schema` 的请求。
   - 发送 POST 请求到 `/api/tasks/weather`，携带 `task_id` 和 `params` (日期/地点)。
3. [处理与验证] 
   - 服务接收请求，校验日期格式及是否存在于数据库中。
   - 若校验失败，返回 HTTP 400 错误；若成功，查询模拟数据。
4. [结果返回] 
   - 返回标准响应结构：包含 `task_id`、执行状态 (`status`) 和具体数据载荷 (`artifact`)。

【依赖说明】
- Python 库：`fastapi`, `uvicorn`, `pydantic`。
- 安装命令：`pip install fastapi uvicorn pydantic`。

【运行方式】
- 直接运行：`python A2A天气代理服务.py`
- 服务地址：默认启动在 http://0.0.0.0:8000
- 测试发现：访问 http://localhost:8000/.well-known/agent.json
- 测试调用：POST http://localhost:8000/api/tasks/weather
"""

from fastapi import FastAPI, HTTPException
from datetime import date
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional

# 初始化 FastAPI 应用
app = FastAPI(
    title="A2A Weather Agent",
    description="遵循 A2A 协议的智能天气查询代理服务",
    version="1.0.0"
)

# ==========================================
# 1. 常量与配置
# ==========================================

# [模拟数据] 本地天气数据库
# 注意：实际生产环境中，此处应替换为调用真实气象 API (如 OpenWeatherMap, 高德天气等)
# 当前仅支持 2025-05-08 至 2025-05-10 的数据用于演示
SUPPORTED_DATES = {
    "2025-05-08": {"temperature": "25℃", "condition": "雷阵雨"},
    "2025-05-09": {"temperature": "18℃", "condition": "小雨转晴"},
    "2025-05-10": {"temperature": "22℃", "condition": "多云转晴"}
}

# [A2A 协议] Agent 身份名片 (Agent Card)
# 此字典将通过 /.well-known/agent.json 端点暴露，是其他 Agent 发现本服务的关键
WEATHER_AGENT_CARD = {
    "name": "WeatherAgent",             # 服务名称
    "version": "1.0",                   # 版本号
    "description": "提供指定日期的天气数据查询服务", # 能力描述
    "endpoints": {
        # 任务提交入口：其他 Agent 向此 URL 发送 POST 请求
        "task_submit": "/api/tasks/weather",
        # 状态订阅入口 (SSE)：用于长任务的状态推送 (本示例暂未实现该端点逻辑，但声明了协议支持)
        "sse_subscribe": "/api/tasks/updates"
    },
    # [输入规范] 定义调用者必须提供的参数结构 (JSON Schema)
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string", 
                "format": "date", 
                "description": "查询日期，格式 YYYY-MM-DD"
            },
            "location": {
                "type": "string", 
                "enum": ["北京"], 
                "description": "查询城市，当前仅支持北京"
            }
        },
        "required": ["date"] # 标记必填字段
    },
    # [安全认证] 声明支持的认证方式
    "authentication": {"methods": ["API_Key"]}
}

# ==========================================
# 2. 数据模型定义
# ==========================================

class WeatherTaskRequest(BaseModel):
    """
    天气查询任务请求模型
    
    遵循 A2A 任务提交规范，包含任务唯一标识和具体参数。
    """
    task_id: str          # 任务唯一 ID，用于追踪和幂等性控制
    params: Dict[str, Any] # 动态参数字典，需符合 input_schema 定义

# ==========================================
# 3. 接口路由实现
# ==========================================

@app.get("/.well-known/agent.json")
async def get_agent_card() -> dict:
    """
    [服务发现] 获取 Agent 名片
    
    这是 A2A 协议的标准入口。当其他 AI Agent 想要寻找天气服务时，
    它们会首先访问此路径来获取本服务的元数据、接口地址和参数规范。
    
    Returns:
        dict: 完整的 Agent Card 信息
    """
    return WEATHER_AGENT_CARD

@app.post("/api/tasks/weather")
async def handle_weather_task(request: WeatherTaskRequest) -> dict:
    """
    [任务处理] 执行天气查询任务
    
    接收来自其他 Agent 的任务请求，验证参数并返回结构化结果。
    采用“任务完成即返回”的同步模拟模式，但保留了 task_id 以符合异步协议结构。
    
    Args:
        request: 包含 task_id 和 params 的请求对象
        
    Returns:
        dict: 标准任务响应，包含 status 和 artifact (数据载荷)
        
    Raises:
        HTTPException: 当日期无效或参数缺失时抛出 400 错误
    """
    # [参数提取] 从请求中获取目标日期
    target_date = request.params.get("date")
    location = request.params.get("location", "北京") # 默认值为北京
    
    # [业务验证] 
    # 1. 检查日期是否提供
    # 2. 检查日期是否在支持的数据库范围内 (模拟真实 API 的数据可用性检查)
    if not target_date:
        raise HTTPException(status_code=400, detail="缺少必填参数 'date'")
    
    if target_date not in SUPPORTED_DATES:
        # 提供更友好的错误提示，告知用户支持的日期范围
        available_dates = ", ".join(SUPPORTED_DATES.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"无效日期 '{target_date}'。当前模拟数据仅支持: {available_dates}"
        )
    
    # [数据检索] 从模拟数据库中获取天气信息
    weather_data = SUPPORTED_DATES[target_date]
    
    # [构建响应] 遵循 A2A 响应结构
    return {
        "task_id": request.task_id,       # 回显任务 ID，确保调用者能对应请求
        "status": "completed",            # 任务状态：已完成
        "artifact": {                     # 核心数据载荷
            "date": target_date,
            "location": location,
            "weather": weather_data       # 具体的温度和天气状况
        }
    }

# ==========================================
# 4. 程序入口
# ==========================================
if __name__ == "__main__":
    # [服务启动]
    # host="0.0.0.0" 允许外部网络访问 (适用于 Docker 或局域网部署)
    # port=8000 是 FastAPI 的默认端口
    print("🚀 正在启动 A2A 天气代理服务...")
    print(f"📡 服务发现地址：http://0.0.0.0:8000/.well-known/agent.json")
    print(f"📥 任务提交地址：http://0.0.0.0:8000/api/tasks/weather")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
