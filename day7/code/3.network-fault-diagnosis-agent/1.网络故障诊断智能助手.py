"""
================================================================================
LangChain 真实网络故障诊断 Agent (基于 LCEL 架构 + 系统级工具)
================================================================================

【核心功能】
一个具备真实执行能力的网络运维智能助手。它不仅能理解自然语言
描述的网络故障，还能自主调用操作系统底层的网络命令 (Ping, NSLookup, IP Config, Grep)
进行实际探测，并根据真实的返回结果进行逻辑推理，最终给出准确的诊断报告。

主要能力：
1. 真实连通性测试：调用系统 ping 命令，检测目标主机的可达性及延迟。
2. 真实 DNS 解析：调用 nslookup/dig，验证域名解析是否正确。
3. 本地接口巡检：调用 ipconfig/ifconfig，检查网卡状态、IP 配置及启停状态。
4. 智能日志检索：在系统日志中搜索关键词，定位网络错误根源。
5. 自主决策推理：Agent 根据实时反馈，动态调整诊断路径 (如：Ping 通但网页打不开 -> 自动检查 DNS 或端口)。

【技术优化】
1. 架构升级：
   - 采用 LangChain 0.2+ 标准的 create_react_agent + ReActJsonSingleInputOutputParser。
   - 解决旧版 Agent 无法准确传递多参数 (如关键词 + 时间范围) 的痛点。
2. 工具真实化：
   - 使用 subprocess 模块封装真实的系统命令，告别硬编码模拟数据。
   - 增加命令注入防护 (白名单校验)，确保执行安全。
3. 模型增强：
   - 升级至 ChatTongyi (Chat Model)，提升对复杂运维场景的理解力。
4. 鲁棒性设计：
   - 所有工具均包含超时控制 (Timeout) 和异常捕获，防止 Agent 因命令挂起而卡死。
   - 开启 handle_parsing_errors=True，允许模型自我纠正格式错误。

【执行流程步骤】
--------------------------------------------------------------------------------
Step 1: 定义系统级工具 (System Tools)
   - 工具 1 (ping_network): 封装 subprocess.run(['ping', ...])，解析输出提取延迟/丢包。
   - 工具 2 (resolve_dns): 封装 nslookup，提取解析后的 IP 地址。
   - 工具 3 (check_interface): 封装 ipconfig/ifconfig，过滤特定接口状态。
   - 工具 4 (search_logs): 封装 grep/tail，在 /var/log 或系统日志中检索错误。
   - *安全机制*: 所有输入参数经过正则白名单校验，防止 Shell 注入。

Step 2: 构建 ReAct 提示词 (Prompt Engineering)
   - 定义专用于运维场景的 System Prompt，强调“先测试后结论”。
   - 强制模型输出 JSON 格式的行动指令，确保参数精准传递。

Step 3: 组装 Agent 核心 (LCEL)
   - 初始化 ChatTongyi (qwen-turbo/max) 模型。
   - 绑定 JSON 解析器与工具列表。
   - 创建 AgentExecutor，设置最大迭代次数 (防止死循环) 和详细日志模式。

Step 4: 任务执行与诊断
   - 接收用户自然语言故障描述。
   - Agent 自主规划：思考 -> 调用真实命令 -> 观察真实输出 -> 再思考。
   - 综合所有探测结果，生成最终诊断建议。

【业务流程图解】
  用户故障描述 --> Agent 意图分析
      ├── (怀疑连通性) --> [执行真实 Ping] --> 获取真实延迟/丢包率
      ├── (怀疑域名) ---> [执行真实 DNS] --> 获取真实解析 IP
      ├── (怀疑本地) ---> [执行真实 Ifconfig] --> 获取真实网卡状态
      └── (怀疑报错) ---> [执行真实 Grep] --> 获取真实日志片段
      ↓
  收集真实探测数据 --> LLM 综合推理 --> 输出专业诊断报告

================================================================================
"""

import os
import re
import subprocess
import platform
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# LangChain Core & Community
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
import dashscope

# -----------------------------------------------------------------------------
# 1. 环境配置
# -----------------------------------------------------------------------------
load_dotenv()
api_key = os.environ.get('DASHSCOPE_API_KEY')
if not api_key:
    raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请检查 .env 文件或系统设置")
dashscope.api_key = api_key

# -----------------------------------------------------------------------------
# 2. 安全与辅助函数
# -----------------------------------------------------------------------------

def is_safe_input(text: str, allow_ip: bool = True) -> bool:
    """
    简单的安全校验，防止命令注入。
    只允许字母、数字、点、横杠、下划线。
    """
    if not text:
        return False
    # 白名单正则：允许字母、数字、. - _ : / (用于路径)
    pattern = r'^[a-zA-Z0-9.\-_:\/]+$'
    if re.match(pattern, text):
        return True
    return False

def get_system_command(cmd_base: str) -> List[str]:
    """根据操作系统返回正确的命令格式"""
    system_os = platform.system()
    if cmd_base == "ping":
        # Windows: ping -n 4, Linux/Mac: ping -c 4
        return ["ping", "-n", "4"] if system_os == "Windows" else ["ping", "-c", "4"]
    elif cmd_base == "nslookup":
        return ["nslookup"]
    elif cmd_base == "ipconfig":
        return ["ipconfig"] if system_os == "Windows" else ["ifconfig"]
    return []

# -----------------------------------------------------------------------------
# 3. 定义真实系统工具 (Real System Tools)
# -----------------------------------------------------------------------------

def ping_network(target: str) -> str:
    """
    执行真实的 Ping 操作，检查网络连通性。
    参数：target (域名或 IP)。
    注意：会自动限制 ping 次数为 4 次以防挂起。
    """
    if not is_safe_input(target):
        return "错误：无效的目标地址格式，包含非法字符。"
    
    try:
        cmd = get_system_command("ping") + [target]
        # 设置超时 10 秒
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            # 简单提取延迟 (不同系统输出格式不同，这里做通用提取尝试)
            time_match = re.search(r"time[=<](\d+\.?\d*)\s*ms", output)
            if time_match:
                return f"Ping {target} 成功。平均延迟: {time_match.group(1)}ms。\n完整输出摘要: {output[:200]}"
            return f"Ping {target} 成功。\n输出摘要: {output[:200]}"
        else:
            return f"Ping {target} 失败。\n错误信息: {result.stderr[:200] or result.stdout[:200]}"
            
    except subprocess.TimeoutExpired:
        return f"Ping {target} 超时：命令执行超过 10 秒，目标可能不可达或防火墙拦截。"
    except Exception as e:
        return f"Ping 执行出错: {str(e)}"

def resolve_dns(hostname: str) -> str:
    """
    执行真实的 DNS 查询。
    参数：hostname (域名)。
    """
    if not is_safe_input(hostname):
        return "错误：无效的主机名格式。"
    
    # 如果是 IP 地址，直接提示
    if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", hostname):
        return f"输入 '{hostname}' 是 IP 地址，无需 DNS 解析。"

    try:
        cmd = get_system_command("nslookup") + [hostname]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # 尝试提取 IP 地址行
            lines = result.stdout.split('\n')
            ips = []
            for line in lines:
                if "Address:" in line and "#" not in line: # 过滤掉 IPv6 注释等
                    parts = line.split()
                    if len(parts) > 1:
                        ips.append(parts[-1])
            
            if ips:
                return f"DNS 解析 {hostname} 成功。解析到的 IP: {', '.join(ips[:3])}\n完整摘要: {result.stdout[:300]}"
            return f"DNS 解析 {hostname} 完成，但未提取到明确 IP。\n输出: {result.stdout[:300]}"
        else:
            return f"DNS 解析 {hostname} 失败。\n错误: {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        return f"DNS 查询 {hostname} 超时。"
    except Exception as e:
        return f"DNS 查询出错: {str(e)}"

def check_interface(interface_name: Optional[str] = None) -> str:
    """
    检查本地网络接口状态。
    参数：interface_name (可选，如 'eth0', 'Wi-Fi')。若不填则显示所有。
    """
    # 允许空字符串或安全字符
    if interface_name and not is_safe_input(interface_name):
        return "错误：无效的接口名称格式。"

    try:
        cmd = get_system_command("ipconfig")
        if interface_name:
            # 某些系统不支持直接过滤，这里先获取全部再在 Python 中过滤，或尝试管道
            # 为安全起见，获取全部后在内存中过滤
            pass
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            if interface_name:
                # 简单查找包含接口名的段落 (粗略实现)
                # 更精确的实现需要解析具体 OS 的输出结构
                if interface_name.lower() in output.lower():
                    return f"找到接口 '{interface_name}' 的相关信息:\n{output[:500]}"
                else:
                    return f"未在输出中找到接口 '{interface_name}'。请检查名称是否正确。\n可用接口摘要: {output[:300]}"
            else:
                return f"本地网络接口状态摘要:\n{output[:600]}"
        else:
            return f"检查接口失败:\n{result.stderr[:200]}"
            
    except Exception as e:
        return f"接口检查出错: {str(e)}"

def search_logs(keywords: str, time_range: Optional[str] = "recent") -> str:
    """
    搜索系统日志中的网络错误。
    参数：keywords (搜索词), time_range (时间范围描述，主要用于提示，实际搜索最近 1000 行)。
    注意：Linux 搜索 /var/log/syslog 或 messages，Windows 暂时模拟或尝试 eventvwr (较复杂，此处简化为 Linux 风格或通用报错)。
    """
    if not is_safe_input(keywords, allow_ip=False): # 关键词允许更多字符？这里保守处理
        # 放宽关键词限制，允许空格和常见符号，但禁止危险符
        if re.search(r'[;&|`$()]', keywords):
            return "错误：关键词包含非法字符。"
    
    system_os = platform.system()
    try:
        if system_os == "Linux":
            log_files = ["/var/log/syslog", "/var/log/messages", "/var/log/system.log"]
            found_log = None
            for lf in log_files:
                if os.path.exists(lf):
                    found_log = lf
                    break
            
            if not found_log:
                return "未找到常见的系统日志文件 (/var/log/syslog 等)，可能需要 sudo 权限或日志路径不同。"
            
            # 使用 tail + grep
            # 命令：tail -n 1000 /var/log/syslog | grep -i "keyword"
            cmd = ["tail", "-n", "1000", found_log]
            tail_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            grep_cmd = ["grep", "-i", keywords]
            grep_proc = subprocess.run(grep_cmd, input=tail_proc.stdout, capture_output=True, text=True, timeout=5)
            
            if grep_proc.returncode == 0 and grep_proc.stdout:
                lines = grep_proc.stdout.strip().split('\n')
                return f"在日志中找到 {len(lines)} 条匹配 '{keywords}' 的记录 (最近 1000 行):\n" + "\n".join(lines[:5]) + ("\n... (更多记录被截断)" if len(lines) > 5 else "")
            else:
                return f"在最近 1000 行日志中未找到包含 '{keywords}' 的错误条目。"
        
        elif system_os == "Windows":
            # Windows 日志查询较复杂，此处简化模拟或使用 eventquery (需管理员)
            # 为了演示稳定性，返回一个引导性信息
            return f"[Windows 环境] 直接读取系统事件日志需要管理员权限。建议手动打开 '事件查看器' -> 'Windows 日志' -> '系统'，并筛选关键词 '{keywords}'。"
        
        else:
            return f"当前操作系统 {system_os} 暂不支持自动日志检索。"

    except subprocess.TimeoutExpired:
        return "日志搜索超时，日志文件可能过大。"
    except PermissionError:
        return "权限不足：读取系统日志需要管理员/root 权限。"
    except Exception as e:
        return f"日志搜索出错: {str(e)}"

@tool("Ping 网络测试")
def ping_network_tool(target: str) -> str:
    """
    检查与目标主机 (域名或 IP) 的网络连通性。
    参数：target (如 www.google.com 或 8.8.8.8)。
    输出：延迟或错误信息。
    """
    return ping_network(target)

@tool("DNS 解析查询")
def resolve_dns_tool(hostname: str) -> str:
    """
    查询域名对应的 IP 地址。
    参数：hostname (如 www.example.com)。
    输出：IP 地址或解析失败原因。
    """
    return resolve_dns(hostname)

@tool("本地接口检查")
def check_interface_tool(interface_name: Optional[str] = None) -> str:
    """
    检查本机网络适配器状态。
    参数：interface_name (可选，如 eth0, Wi-Fi)，留空则检查所有。
    输出：IP 配置和状态。
    """
    return check_interface(interface_name)

@tool("系统日志搜索")
def search_logs_tool(keywords: str, time_range: str = "recent") -> str:
    """
    在系统日志中搜索网络相关错误。
    参数：keywords (如 timeout, connection refused)，time_range (如 recent)。
    输出：匹配的日志条目。
    """
    return search_logs(keywords=keywords, time_range=time_range)

tools = [ping_network_tool, resolve_dns_tool, check_interface_tool, search_logs_tool]

# -----------------------------------------------------------------------------
# 4. 构建 Agent (LCEL + JSON Parser)
# -----------------------------------------------------------------------------

REACT_PROMPT_TEMPLATE = """You are an expert Network Operations Center (NOC) assistant. 
You have access to real system tools to diagnose network issues.

Tools available:
{tools}

Use the following format strictly:

Question: the user's network issue description
Thought: Analyze the problem and decide which tool to use first. Explain your reasoning in Chinese.
Action:
```json
{{"action": "<one of [{tool_names}]>", "action_input": {{...}}}}
```
Observation: The real output from the tool.
... (Repeat Thought/Action/Observation as needed)
Thought: I have gathered enough information to form a conclusion.
Final Answer: Provide a comprehensive diagnosis report in Chinese, including:
1. Summary of findings (Ping status, DNS status, etc.)
2. Root cause analysis.
3. Actionable recommendations.

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

def create_network_agent():
    llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActJsonSingleInputOutputParser(),
    )
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        early_stopping_method="force"
    )
    
    return executor

# -----------------------------------------------------------------------------
# 5. 主程序入口
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 真实网络故障诊断 Agent 已启动 ===")
    print(f"当前操作系统：{platform.system()}")
    print("提示：本 Agent 将执行真实的系统命令 (Ping, NSLookup 等)，请确保在安全环境中运行。\n")
    
    agent_executor = create_network_agent()
    
    # 测试用例
    tasks = [
        "我无法访问 www.baidu.com，浏览器一直转圈，帮我看看是网络不通还是 DNS 问题？",
        "我的电脑连不上网了，请检查一下我的网络接口状态，特别是是否有 IP 地址。",
        "系统里好像有很多连接超时的错误，请帮我在日志里搜一下 'timeout' 相关的记录。",
        "我想访问 internal.corp.local，但是不行，先帮我 Ping 一下，再查一下 DNS。"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[任务 {i}] : {task}")
        print(f"{'='*60}")
        try:
            response = agent_executor.invoke({"input": task})
            print(f"\n[✅ 最终诊断报告]:\n{response['output']}")
        except Exception as e:
            print(f"\n[❌ 诊断过程出错]: {e}")
        
        print("\n")
