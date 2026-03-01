"""
================================================================================
高级网络故障诊断智能助手 (可扩展的工具串联 Agent)
================================================================================

【核心功能】
构建了一个具备“状态保持”能力的智能网络运维助手。与传统的单步工具调用不同，
引入了一套基于 JSON 的“诊断上下文 (Context)”机制。Agent 在执行多步诊断
（如：提取目标 -> DNS 解析 -> Ping 测试 -> TCP 端口探测）时，会将每一步的中间结果
累积到同一个 JSON 对象中并传递给下一步，从而形成完整的证据链。

主要能力：
1. 智能目标提取：自动从自然语言描述中识别 URL、域名和 IP 地址。
2. 全链路探测：支持 DNS 解析、ICMP Ping、TCP 端口连通性测试。
3. 本地环境感知：自动采集本机 IP 配置和 DNS 服务器设置。
4. 容错与默认值：当用户未提供明确目标时，自动补充公共 DNS 和常用域名进行基准测试。
5. 结构化报告：最终输出基于真实探测数据的综合诊断报告。

【技术架构】
1. 上下文传递模式 (Context Passing Pattern)：
   - 所有工具接收 `context_json` 字符串，解析为字典，更新数据后序列化返回。
   - 解决了 LangChain 原生 Agent 在多步复杂参数传递中的局限性。
2. 安全沙箱：
   - 所有系统命令执行前均经过严格的白名单正则校验 (_safe_hostname)。
   - 防止命令注入攻击 (Command Injection)。
3. 跨平台兼容：
   - 自动识别 Windows/Linux/macOS，动态调整 ping/nslookup/ifconfig 命令参数。
4. 现代化 Agent：
   - 基于 LangChain ReAct + JSON Parser 架构，确保工具调用的参数结构准确。

【执行流程】
  用户问题 --> [init_context] 初始化上下文
      ↓
  [extract_targets] 正则提取域名/IP --> [fill_default_targets] (若无目标则补全)
      ↓
  [collect_local_snapshot] 采集本机网络状态
      ↓
  [dns_lookup_targets] 批量解析域名
      ↓
  [ping_from_context] 对域名和解析IP进行 Ping 测试
      ↓
  [tcp_probe_from_context] 对关键端口 (80/443/53) 进行 TCP 握手测试
      ↓
  LLM 综合所有 JSON 数据 --> 输出最终诊断报告

================================================================================
"""

import json
import os
import platform
import re
import socket
import subprocess
import time
import ipaddress
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# -----------------------------------------------------------------------------
# 第三方库导入说明
# -----------------------------------------------------------------------------
import dashscope
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


# -----------------------------------------------------------------------------
# 基础辅助函数 (Infrastructure Helpers)
# -----------------------------------------------------------------------------

def _json_dumps(obj: Any) -> str:
    """
    统一的 JSON 序列化工具。
    确保中文正常显示 (ensure_ascii=False) 且格式美观 (indent=2)。
    """
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)


def _safe_hostname(host: str) -> bool:
    """
    【安全校验】严格的主机名/IP 白名单检查。
    防止命令注入攻击 (Command Injection)。
    只允许：字母、数字、点、横杠、下划线、冒号 (IPv6)。
    拒绝：Shell 特殊字符 (; | & $ ` 等)。
    """
    if not host:
        return False
    if len(host) > 253:
        return False
    if re.search(r"[;&|`$()<>]", host):
        return False
    if re.fullmatch(r"[A-Za-z0-9.\-_:]+", host):
        return True
    return False


def _try_parse_ip(text: str) -> Optional[str]:
    """
    尝试将字符串解析为标准的 IP 地址 (IPv4/IPv6)。
    如果解析失败返回 None，用于区分域名和 IP。
    """
    try:
        return str(ipaddress.ip_address(text.strip()))
    except Exception:
        return None


def _run_cmd(cmd: List[str], timeout_s: int = 10) -> Dict[str, Any]:
    """
    【核心执行器】安全地执行系统子进程命令。
    功能：
    1. 捕获 stdout/stderr。
    2. 强制执行超时 (防止命令死挂)。
    3. 记录执行耗时和返回码。
    4. 统一错误处理格式。
    """
    started_at = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "duration_ms": int((time.time() - started_at) * 1000),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"timeout after {timeout_s}s",
            "duration_ms": int((time.time() - started_at) * 1000),
            "cmd": cmd,
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(e),
            "duration_ms": int((time.time() - started_at) * 1000),
            "cmd": cmd,
        }


# -----------------------------------------------------------------------------
# 数据提取与解析工具 (Data Extraction & Parsing)
# -----------------------------------------------------------------------------

def _extract_targets_from_text(text: str) -> Dict[str, Any]:
    """
    【智能提取】从自然语言文本中提取网络目标。
    策略：
    1. 正则匹配 HTTP/HTTPS URL。
    2. 解析 URL 中的 Hostname。
    3. 正则匹配 IPv4 地址。
    4. 正则匹配通用域名格式。
    5. 去重并分类 (URLs, Hostnames, IPs)。
    """
    candidates: List[str] = []
    urls: List[str] = []
    hostnames: List[str] = []
    ips: List[str] = []
    raw = text or ""

    for m in re.findall(r"https?://[^\s'\"<>]+", raw, flags=re.IGNORECASE):
        urls.append(m)

    for url in urls:
        try:
            u = urlparse(url)
            if u.hostname:
                candidates.append(u.hostname)
        except Exception:
            pass

    for m in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw):
        ip = _try_parse_ip(m)
        if ip:
            ips.append(ip)

    for m in re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]{0,62}(?:\.[a-zA-Z0-9][a-zA-Z0-9\-]{0,62})+\b", raw):
        candidates.append(m)

    seen: set[str] = set()
    for c in candidates:
        c = c.strip().rstrip(".")
        if not c or c in seen:
            continue
        seen.add(c)
        ip = _try_parse_ip(c)
        if ip:
            ips.append(ip)
            continue
        if _safe_hostname(c):
            hostnames.append(c)

    def _uniq(seq: List[str]) -> List[str]:
        out: List[str] = []
        s: set[str] = set()
        for x in seq:
            if x not in s:
                s.add(x)
                out.append(x)
        return out

    return {
        "urls": _uniq(urls),
        "hostnames": _uniq(hostnames),
        "ips": _uniq(ips),
    }


# -----------------------------------------------------------------------------
# 底层网络探测原语 (Low-level Network Primitives)
# -----------------------------------------------------------------------------

def _nslookup(hostname: str, timeout_s: int = 10) -> Dict[str, Any]:
    """
    执行 DNS 查询 (nslookup)。
    解析输出中的 'Address' 行，提取 IP 列表。
    """
    if not _safe_hostname(hostname):
        return {"ok": False, "hostname": hostname, "error": "invalid hostname"}
    res = _run_cmd(["nslookup", hostname], timeout_s=timeout_s)
    ips: List[str] = []
    if res["stdout"]:
        for line in res["stdout"].splitlines():
            if "Address:" in line and "#" not in line:
                parts = line.split()
                if parts:
                    ip = _try_parse_ip(parts[-1])
                    if ip:
                        ips.append(ip)
    return {
        "ok": bool(ips) and res["ok"],
        "hostname": hostname,
        "ips": list(dict.fromkeys(ips)),
        "raw_ok": res["ok"],
        "stdout_head": (res["stdout"] or "")[:600],
        "stderr_head": (res["stderr"] or "")[:300],
        "duration_ms": res["duration_ms"],
    }


def _ping(target: str, count: int = 4, timeout_s: int = 10) -> Dict[str, Any]:
    """
    执行 ICMP Ping 测试。
    跨平台适配：Windows (-n) vs Linux/Mac (-c)。
    智能解析：提取丢包率 (packet loss) 和平均延迟 (avg RTT)。
    """
    if not target:
        return {"ok": False, "target": target, "error": "empty target"}
    ip = _try_parse_ip(target)
    if ip is None and not _safe_hostname(target):
        return {"ok": False, "target": target, "error": "invalid target"}

    system_os = platform.system()
    if system_os == "Windows":
        cmd = ["ping", "-n", str(count), target]
    else:
        cmd = ["ping", "-c", str(count), target]

    res = _run_cmd(cmd, timeout_s=timeout_s)
    stdout = res["stdout"] or ""

    packet_loss: Optional[float] = None
    avg_ms: Optional[float] = None

    m_loss = re.search(r"(\d+(?:\.\d+)?)%\s*packet loss", stdout)
    if m_loss:
        try:
            packet_loss = float(m_loss.group(1))
        except Exception:
            pass

    m_rtt = re.search(r"=\s*([\d.]+)/([\d.]+)/([\d.]+)", stdout)
    if m_rtt:
        try:
            avg_ms = float(m_rtt.group(2))
        except Exception:
            pass

    m_time = re.search(r"time[=<]\s*([\d.]+)\s*ms", stdout)
    if avg_ms is None and m_time:
        try:
            avg_ms = float(m_time.group(1))
        except Exception:
            pass

    ok = res["ok"] and (packet_loss is None or packet_loss < 100.0)
    return {
        "ok": ok,
        "target": target,
        "packet_loss_pct": packet_loss,
        "avg_ms": avg_ms,
        "stdout_head": stdout[:700],
        "stderr_head": (res["stderr"] or "")[:200],
        "duration_ms": res["duration_ms"],
    }


def _tcp_connect(host: str, port: int, timeout_s: int = 3) -> Dict[str, Any]:
    """
    执行 TCP 三次握手测试。
    使用 Python socket 库，比 telnet 更轻量且易于控制超时。
    用于检测特定服务端口 (如 80, 443, 22) 是否开放。
    """
    if not host:
        return {"ok": False, "host": host, "port": port, "error": "empty host"}
    ip = _try_parse_ip(host)
    if ip is None and not _safe_hostname(host):
        return {"ok": False, "host": host, "port": port, "error": "invalid host"}

    started_at = time.time()
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return {"ok": True, "host": host, "port": int(port), "duration_ms": int((time.time() - started_at) * 1000)}
    except Exception as e:
        return {"ok": False, "host": host, "port": int(port), "error": str(e), "duration_ms": int((time.time() - started_at) * 1000)}


def _ifconfig_snapshot(timeout_s: int = 10) -> Dict[str, Any]:
    """
    采集本机网络接口配置。
    Windows: ipconfig /all
    Linux/Mac: ifconfig
    提取所有 IPv4 地址用于后续分析。
    """
    system_os = platform.system()
    if system_os == "Windows":
        cmd = ["ipconfig", "/all"]
    else:
        cmd = ["ifconfig"]
    res = _run_cmd(cmd, timeout_s=timeout_s)
    stdout = res["stdout"] or ""
    ipv4s = list(dict.fromkeys(re.findall(r"\binet\s+(\d{1,3}(?:\.\d{1,3}){3})\b", stdout)))
    return {
        "ok": res["ok"],
        "os": system_os,
        "ipv4s": ipv4s,
        "stdout_head": stdout[:900],
        "stderr_head": (res["stderr"] or "")[:200],
        "duration_ms": res["duration_ms"],
    }


def _dns_config_snapshot(timeout_s: int = 10) -> Dict[str, Any]:
    """
    采集本机 DNS 服务器配置。
    不同 OS 采用不同策略：
    - macOS: scutil --dns
    - Linux: 读取 /etc/resolv.conf
    - Windows: (简化处理，暂不深入解析 ipconfig 中的 DNS 部分)
    """
    system_os = platform.system()
    if system_os == "Darwin":
        res = _run_cmd(["scutil", "--dns"], timeout_s=timeout_s)
        stdout = res["stdout"] or ""
        servers = list(dict.fromkeys(re.findall(r"nameserver\[[0-9]+\]\s*:\s*([0-9.]+)", stdout)))
        return {
            "ok": res["ok"],
            "os": system_os,
            "dns_servers": servers,
            "stdout_head": stdout[:1200],
            "stderr_head": (res["stderr"] or "")[:200],
            "duration_ms": res["duration_ms"],
        }
    if system_os == "Linux":
        try:
            with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            servers = list(dict.fromkeys(re.findall(r"^nameserver\s+([0-9.]+)\s*$", content, flags=re.MULTILINE)))
            return {"ok": True, "os": system_os, "dns_servers": servers, "stdout_head": content[:1200], "stderr_head": "", "duration_ms": 0}
        except Exception as e:
            return {"ok": False, "os": system_os, "dns_servers": [], "stdout_head": "", "stderr_head": str(e), "duration_ms": 0}
    return {"ok": False, "os": system_os, "dns_servers": [], "stdout_head": "", "stderr_head": "unsupported os", "duration_ms": 0}


# -----------------------------------------------------------------------------
# LangChain 工具定义 (Tool Definitions with Context Passing)
# -----------------------------------------------------------------------------
# 所有工具均采用 "读取 JSON 上下文 -> 更新数据 -> 返回新 JSON" 的模式

@tool("初始化诊断上下文")
def init_context(problem: str) -> str:
    """创建一个可在多工具之间串联传递的诊断上下文(JSON字符串)。"""
    ctx: Dict[str, Any] = {
        "problem": problem,
        "os": platform.system(),
        "targets": {"urls": [], "hostnames": [], "ips": []},
        "local": {},
        "dns": {},
        "ping": {},
        "tcp": {},
        "notes": [],
    }
    return _json_dumps(ctx)


@tool("提取目标")
def extract_targets(context_json: str) -> str:
    """从上下文中的 problem 提取 URL/域名/IP，并写回 targets。"""
    ctx = json.loads(context_json)
    targets = _extract_targets_from_text(ctx.get("problem", ""))
    ctx["targets"] = targets
    if not (targets.get("urls") or targets.get("hostnames") or targets.get("ips")):
        ctx["notes"].append("未从问题描述中提取到明确目标，建议补充域名/IP/URL。")
    return _json_dumps(ctx)


@tool("采集本地网络快照")
def collect_local_snapshot(context_json: str) -> str:
    """采集本机网络关键状态(ifconfig/ipconfig, DNS 配置)，写回 local。"""
    ctx = json.loads(context_json)
    ctx["local"]["ifconfig"] = _ifconfig_snapshot()
    ctx["local"]["dns_config"] = _dns_config_snapshot()
    return _json_dumps(ctx)

@tool("补全默认目标")
def fill_default_targets(context_json: str) -> str:
    """
    防御性编程：当未提取到目标时，补充一组常用探测目标。
    防止因用户描述模糊导致诊断流程中断。
    """
    ctx = json.loads(context_json)
    targets = ctx.get("targets") or {}
    urls = targets.get("urls") or []
    hostnames = targets.get("hostnames") or []
    ips = targets.get("ips") or []

    if urls or hostnames or ips:
        return _json_dumps(ctx)

    ctx["targets"] = {
        "urls": [],
        "hostnames": ["www.baidu.com"],
        "ips": ["114.114.114.114", "8.8.8.8"],
    }
    ctx["notes"].append("未提供明确目标，已补全默认探测目标以继续诊断。")
    return _json_dumps(ctx)


@tool("批量 DNS 解析")
def dns_lookup_targets(context_json: str, timeout_s: int = 10) -> str:
    """遍历 targets.hostnames 执行 nslookup，并将结果写回 dns 字段。"""
    ctx = json.loads(context_json)
    hostnames = (ctx.get("targets") or {}).get("hostnames") or []
    results: Dict[str, Any] = {}
    for h in hostnames:
        results[h] = _nslookup(h, timeout_s=timeout_s)
    ctx["dns"] = results
    return _json_dumps(ctx)


@tool("连通性探测 Ping")
def ping_from_context(context_json: str, count: int = 4, timeout_s: int = 10) -> str:
    """
    对 targets 中的主机名、IP 以及 DNS 解析出的 IP 进行 Ping 探测。
    结果写回 ping 字段，区分 host:xxx 和 ip:xxx 标签。
    """
    ctx = json.loads(context_json)
    targets = ctx.get("targets") or {}
    hostnames: List[str] = targets.get("hostnames") or []
    ips: List[str] = targets.get("ips") or []
    dns: Dict[str, Any] = ctx.get("dns") or {}

    ping_results: Dict[str, Any] = {}

    for h in hostnames:
        ping_results[f"host:{h}"] = _ping(h, count=count, timeout_s=timeout_s)
        resolved_ips = ((dns.get(h) or {}).get("ips") or [])[:2]
        for ip in resolved_ips:
            ping_results[f"ip:{h}:{ip}"] = _ping(ip, count=count, timeout_s=timeout_s)

    for ip in ips:
        ping_results[f"ip:{ip}"] = _ping(ip, count=count, timeout_s=timeout_s)

    ctx["ping"] = ping_results
    return _json_dumps(ctx)


@tool("TCP 端口探测")
def tcp_probe_from_context(context_json: str, ports: Optional[List[int]] = None) -> str:
    """
    对关键目标进行 TCP 端口连通性测试。
    默认探测端口：53 (DNS), 80 (HTTP), 443 (HTTPS)。
    结果写回 tcp 字段，格式为 "host:port"。
    """
    ctx = json.loads(context_json)
    targets = ctx.get("targets") or {}
    hostnames: List[str] = targets.get("hostnames") or []
    ips: List[str] = targets.get("ips") or []
    dns: Dict[str, Any] = ctx.get("dns") or {}

    if ports is None:
        ports = [53, 80, 443]

    all_hosts: List[str] = []
    all_hosts.extend(hostnames)
    all_hosts.extend(ips)
    for h in hostnames:
        all_hosts.extend(((dns.get(h) or {}).get("ips") or [])[:1])

    uniq_hosts: List[str] = []
    seen: set[str] = set()
    for h in all_hosts:
        if h not in seen and h:
            seen.add(h)
            uniq_hosts.append(h)

    results: Dict[str, Any] = {}
    for h in uniq_hosts:
        for p in ports:
            results[f"{h}:{int(p)}"] = _tcp_connect(h, int(p))

    ctx["tcp"] = results
    return _json_dumps(ctx)


@tool("输出诊断上下文")
def print_context(context_json: str) -> str:
    """直接输出当前上下文(JSON字符串)，用于调试或人工复核。"""
    return context_json


# -----------------------------------------------------------------------------
# Agent 构建与配置 (Agent Construction)
# -----------------------------------------------------------------------------

REACT_PROMPT_TEMPLATE = """You are an expert network diagnostic assistant.
You MUST use tools to collect evidence before concluding.

You have access to the following tools:
{tools}

Important:
- Most tools take a parameter named context_json and return an updated context_json.
- When chaining tools, always pass the latest context_json output as the next tool's input.
- If no targets are extracted, call 补全默认目标 to proceed with diagnosis.

Use the following format strictly:

Question: the user's network issue description
Thought: explain reasoning in Chinese
Action:
```json
{{"action": "<one of [{tool_names}]>", "action_input": {{...}}}}
```
Observation: tool output
... (repeat)
Thought: I now know the final answer
Final Answer: respond in Chinese with a structured report and actionable steps

Begin!

Question: {input}
{agent_scratchpad}
"""


def create_advanced_network_agent() -> AgentExecutor:
    """
    工厂函数：创建并配置高级网络诊断 Agent。
    """
    llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"))
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    agent = create_react_agent(
        llm=llm,
        tools=[
            init_context,
            extract_targets,
            collect_local_snapshot,
            fill_default_targets,
            dns_lookup_targets,
            ping_from_context,
            tcp_probe_from_context,
        ],
        prompt=prompt,
        output_parser=ReActJsonSingleInputOutputParser(),
    )

    return AgentExecutor(
        agent=agent,
        tools=[
            init_context,
            extract_targets,
            collect_local_snapshot,
            fill_default_targets,
            dns_lookup_targets,
            ping_from_context,
            tcp_probe_from_context,
        ],
        verbose=True,
        handle_parsing_errors="输出不符合格式要求。请严格按模板输出，并在需要调用工具时只输出 Action JSON 代码块。",
        max_iterations=18,
        early_stopping_method="force",
    )


# -----------------------------------------------------------------------------
# 主程序入口 (Main Entry Point)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请检查 .env 文件或系统设置")
    dashscope.api_key = api_key

    print("=== 高级网络故障诊断智能助手 已启动 ===")
    print(f"当前操作系统：{platform.system()}\n")

    # 实例化 Agent
    agent_executor = create_advanced_network_agent()

    # 定义测试任务
    tasks = [
        "我无法访问 www.baidu.com，浏览器一直转圈，帮我看看是网络不通还是 DNS 问题？",
        "我想访问 internal.corp.local，但是不行，先帮我 Ping 一下，再查一下 DNS。",
        "我的电脑连不上网了，请检查一下我的网络接口状态，特别是是否有 IP 地址。",
    ]

    # 执行任务循环
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[任务 {i}] : {task}")
        print(f"{'='*60}")
        try:
            resp = agent_executor.invoke({"input": task})
            print(f"\n[✅ 最终诊断报告]:\n{resp.get('output')}")
        except Exception as e:
            print(f"\n[❌ 诊断过程出错]: {e}")
        print()
