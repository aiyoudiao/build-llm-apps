#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2A 篮球活动决策代理 (A2A Basketball Activity Decision Agent)

【功能概述】
本模块实现了一个基于 A2A (Agent-to-Agent) 协议的高层业务代理。
它不直接生产数据，而是作为“编排者 (Orchestrator)”，动态发现并调用底层的“天气代理服务”，
根据实时天气数据自动执行“是否举行篮球活动”的业务决策逻辑。

【核心角色】
- 消费者 (Client)：主动发现并消费其他 Agent 的能力。
- 决策者 (Decider)：结合外部数据与内部规则，输出最终业务结论。

【系统架构】
- 服务发现层：通过标准路径 `/.well-known/agent.json` 动态获取下游服务的接口定义。
- 通信层：基于 HTTP/REST 构建符合 A2A 协议的任务提交机制。
- 决策层：内置业务规则引擎（雨雪天取消，晴天/多云天确认）。

【核心流程】
1. [初始化] 配置下游天气服务的基准 URL 和认证凭证。
2. [服务发现] 
   - 请求 `/.well-known/agent.json` 获取“Agent 名片”。
   - 解析名片，动态提取任务提交端点 (`task_submit`)，实现接口解耦。
3. [任务构造] 
   - 生成全局唯一的 `task_id` (UUID)，确保任务可追踪和幂等。
   - 封装查询参数 (日期、地点)。
4. [远程调用] 
   - 向动态提取的端点发送 POST 请求，携带认证头。
   - 接收并解析标准化的响应载荷 (`artifact`)。
5. [业务决策] 
   - 规则：若天气状况包含“雨”或“雪” $\rightarrow$ 取消活动。
   - 规则：否则 $\rightarrow$ 确认活动。
   - 容错：若调用失败，返回错误状态而非抛出异常中断主程序。

【依赖说明】
- Python 库：`requests`, `uuid`。
- 前置条件：必须有一个正在运行的“天气代理服务” (如 A2A天气代理服务.py)。
- 网络要求：需能访问 `http://localhost:8000`。

【运行示例】
- 直接运行：`python A2A篮球活动决策代理.py`
- 预期行为：自动查询 2025-05-08 的天气，并打印篮球活动安排结果。
"""
import os
import requests
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class BasketBallAgent:
    """
    篮球活动决策代理
    
    该类实现了基于 A2A 协议的智能决策逻辑，能够自动感知天气服务状态
    并根据气象条件决定户外篮球活动的可行性。
    """
    
    def __init__(self):
        # [配置] 下游天气服务的基准地址
        # 在实际生产中，此地址应来自服务注册中心或环境变量
        self.weather_agent_url = "http://localhost:8000"
        
        # [安全] API 认证密钥
        # ⚠️ 最佳实践：应从环境变量 os.getenv("WEATHER_API_KEY") 读取，严禁硬编码
        self.api_key = os.getenv("WEATHER_API_KEY", "SECRET_KEY")

    def _create_task(self, target_date: str) -> Dict[str, Any]:
        """
        创建符合 A2A 标准的任务对象
        
        生成唯一的任务 ID 并封装参数，确保请求符合下游服务的 Schema 定义。
        
        Args:
            target_date: 目标日期字符串 (格式: YYYY-MM-DD)
            
        Returns:
            dict: 标准化的任务请求体
        """
        return {
            "task_id": str(uuid.uuid4()),  # 生成 UUID 保证全局唯一性，支持幂等操作
            "params": {
                "date": target_date,
                "location": "北京"  # 默认查询地点，可根据需求扩展为用户输入
            }
        }

    def check_weather(self, target_date: str) -> Dict[str, Any]:
        """
        通过 A2A 协议动态查询天气
        
        此方法完整实现了 A2A 的服务发现与调用流程：
        1. 获取远程 Agent 的名片 (Agent Card)。
        2. 从名片中动态解析任务提交接口地址 (避免硬编码路径)。
        3. 构造并发送任务请求。
        
        Args:
            target_date: 查询日期
            
        Returns:
            dict: 解析后的天气数据载荷 (artifact)
            
        Raises:
            Exception: 当服务发现失败、网络错误或下游返回非 200 状态时抛出
        """
        try:
            # [步骤 1: 服务发现] 获取 Agent 名片
            # 这是 A2A 协议的核心：客户端不需要预先知道具体接口路径，只需知道基准 URL
            card_response = requests.get(
                f"{self.weather_agent_url}/.well-known/agent.json",
                timeout=5
            )
            card_response.raise_for_status()  # 检查 HTTP 错误
            agent_card = card_response.json()
            
            # [步骤 2: 动态路由] 从名片中提取实际的任务提交端点
            task_endpoint = agent_card.get("endpoints", {}).get("task_submit")
            if not task_endpoint:
                raise ValueError("Agent 名片中未找到 'task_submit' 端点定义")
            
            # [步骤 3: 任务构造]
            task_payload = self._create_task(target_date)
            
            # [步骤 4: 任务提交] 发送 POST 请求
            # 携带 Authorization 头进行认证 (符合名片中的 authentication 声明)
            response = requests.post(
                f"{self.weather_agent_url}{task_endpoint}",
                json=task_payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            # [步骤 5: 结果解析]
            if response.status_code == 200:
                result_data = response.json()
                # A2A 标准响应中，核心数据位于 'artifact' 字段
                return result_data.get("artifact", {})
            else:
                # 下游服务返回业务错误 (如日期无效)
                raise Exception(f"天气服务返回错误 [{response.status_code}]: {response.text}")
                
        except requests.exceptions.RequestException as e:
            # 网络层面错误 (连接拒绝、超时等)
            raise Exception(f"无法连接天气代理服务: {str(e)}")

    def schedule_meeting(self, date: str) -> Dict[str, Any]:
        """
        综合决策逻辑：根据天气安排篮球活动
        
        封装了具体的业务规则：
        - 规则 A: 若天气包含"雨"或"雪" $\rightarrow$ 取消。
        - 规则 B: 其他情况 $\rightarrow$ 确认。
        - 规则 C: 若数据获取失败 $\rightarrow$ 报错。
        
        Args:
            date: 计划活动的日期
            
        Returns:
            dict: 决策结果，包含 status (confirmed/cancelled/error) 及详细信息
        """
        try:
            # 1. 获取天气数据
            weather_result = self.check_weather(date)
            
            # 防御性编程：检查返回数据结构是否完整
            if not weather_result or "weather" not in weather_result:
                raise ValueError("返回的天气数据格式不正确")
                
            condition = weather_result["weather"].get("condition", "")
            
            # 2. 执行决策规则
            # 检查是否包含恶劣天气关键词
            if "雨" in condition or "雪" in condition:
                return {
                    "status": "cancelled",
                    "reason": f"检测到恶劣天气：{condition}",
                    "date": date
                }
            else:
                return {
                    "status": "confirmed",
                    "message": "天气良好，适合运动",
                    "weather": weather_result["weather"],
                    "date": date
                }
                
        except Exception as e:
            # 3. 异常捕获：确保单个任务失败不影响主程序运行
            return {
                "status": "error",
                "detail": str(e),
                "date": date
            }

# ==========================================
# 程序入口与演示
# ==========================================
if __name__ == "__main__":
    print("🏀 启动篮球活动决策代理...")
    
    # 实例化代理
    meeting_agent = BasketBallAgent()
    
    # [测试场景 1] 查询已知有数据的日期 (2025-05-08 在模拟数据中是雷阵雨)
    test_date = "2025-05-08"
    print(f"\n📅 正在评估 {test_date} 的篮球活动安排...")
    result = meeting_agent.schedule_meeting(test_date)
    
    # 格式化输出结果
    print("-" * 30)
    if result["status"] == "confirmed":
        print(f"✅ 活动确认: {result['message']}")
        print(f"   天气详情: {result['weather']}")
    elif result["status"] == "cancelled":
        print(f"❌ 活动取消: {result['reason']}")
    else:
        print(f"⚠️ 决策失败: {result['detail']}")
    print("-" * 30)
    
    # [测试场景 2] (可选) 测试另一个日期
    result_2 = meeting_agent.schedule_meeting("2025-05-10")
    print(f"\n📅 正在评估备选日期 {result_2['date']} 的篮球活动安排...")
    # 格式化输出结果
    print("-" * 30)
    if result_2["status"] == "confirmed":
        print(f"✅ 活动确认: {result_2['message']}")
        print(f"   天气详情: {result_2['weather']}")
    elif result_2["status"] == "cancelled":
        print(f"❌ 活动取消: {result_2['reason']}")
    else:
        print(f"⚠️ 决策失败: {result_2['detail']}")
    print("-" * 30)
