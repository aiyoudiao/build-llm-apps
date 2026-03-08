#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桌面TXT文件助手 (Desktop TXT File Assistant)

【功能概述】
本脚本基于 MCP (Model Context Protocol) 协议构建一个本地服务，旨在为 AI 模型提供
安全访问用户桌面 .txt 文本文件的能力。它允许 AI 执行以下操作：
1. 统计桌面上 .txt 文件的总数。
2. 列出桌面上所有 .txt 文件的名称。
3. 读取指定 .txt 文件的具体内容。

【工作流程】
1. 初始化 FastMCP 服务器实例。
2. 注册三个核心工具函数 (Tools)。
3. 启动服务器，监听来自 MCP 客户端（AI 助手）的调用请求。
4. 接收到请求后，动态解析用户桌面路径，执行文件操作并返回结果。

【安全机制】
- 路径限制：所有操作严格限制在用户桌面目录 (~/Desktop) 内，防止遍历系统其他目录。
- 后缀检查：读取文件时强制校验文件后缀是否为 .txt，防止读取非文本文件或潜在的危险文件。
- 异常捕获：文件读取操作包裹在 try-except 块中，确保服务不会因单个文件错误而崩溃。

【依赖库】
- mcp: Model Context Protocol 服务端实现
- pathlib, os: 标准库，用于跨平台路径处理

【使用方法】
1. 运行本脚本 mcp dev 1.桌面TXT文件助手.py
2. 确保 MCP 客户端已启动并连接至本服务器。
3. 在 MCP 客户端中，发送以下指令：
   - count_desktop_txt_files()：统计桌面上的 .txt 文件数量。
   - list_desktop_txt_files()：列出桌面上的所有 .txt 文件名称。
   - read_txt_file("filename.txt")：读取指定 .txt 文件的内容。
"""

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ==========================================
# 1. 初始化 MCP 服务器
# ==========================================
# 创建名为 "桌面 TXT 文件统计器" 的 MCP 服务实例
mcp = FastMCP("桌面 TXT 文件统计器")

# ==========================================
# 2. 定义工具函数 (Tools)
# ==========================================

@mcp.tool()
def count_desktop_txt_files() -> int:
    """
    统计桌面上 .txt 文件的数量
    
    Returns:
        int: 桌面上的 .txt 文件总数
    """
    # [关键步骤] 动态获取当前用户的桌面绝对路径，兼容不同操作系统
    desktop_path = Path(os.path.expanduser("~/Desktop"))

    # [核心逻辑] 使用 glob 模式匹配所有以 .txt 结尾的文件
    # list() 将生成器转换为列表以便计算长度
    txt_files = list(desktop_path.glob("*.txt"))
    
    return len(txt_files)


@mcp.tool()
def list_desktop_txt_files() -> str:
    """
    获取桌面上所有 .txt 文件的名称列表
    
    Returns:
        str: 格式化的文件列表字符串，若无文件则返回提示信息
    """
    # [关键步骤] 再次确认桌面路径，确保操作范围正确
    desktop_path = Path(os.path.expanduser("~/Desktop"))

    # [核心逻辑] 检索所有 .txt 文件
    txt_files = list(desktop_path.glob("*.txt"))

    # [分支处理] 如果未找到任何文件，返回友好的提示信息
    if not txt_files:
        return "桌面上没有找到 .txt 文件。"

    # [格式化输出] 将文件对象转换为易读的字符串列表格式
    # 例如: "- report.txt\n- notes.txt"
    file_list = "\n".join([f"- {file.name}" for file in txt_files])
    
    return f"在桌面上找到 {len(txt_files)} 个 .txt 文件：\n{file_list}"


@mcp.tool()
def read_txt_file(filename: str) -> str:
    """
    读取桌面上指定 .txt 文件的内容
    
    Args:
        filename (str): 目标文件名 (例如: "meeting_notes.txt")
        
    Returns:
        str: 文件内容或详细的错误信息
    """
    # [路径构建] 组合桌面根路径与用户提供的文件名
    desktop_path = Path(os.path.expanduser("~/Desktop"))
    file_path = desktop_path / filename
    
    # [安全检查 1] 验证文件是否真实存在，避免 FileNotFoundError
    if not file_path.exists():
        return f"错误：文件 '{filename}' 不存在于桌面上。"
    
    # [安全检查 2] 强制校验文件后缀，确保只读取 .txt 文件
    # 防止用户误操作读取图片、可执行文件或其他敏感数据
    if file_path.suffix.lower() != '.txt':
        return f"错误：文件 '{filename}' 不是 txt 文件 (检测到后缀: {file_path.suffix})。"
    
    try:
        # [文件读取] 以 UTF-8 编码打开并读取全文
        # 使用 with 语句确保文件在使用后自动关闭，释放资源
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return f"文件 '{filename}' 的内容：\n\n{content}"
        
    except UnicodeDecodeError:
        # 专门处理编码错误，提示文件可能不是纯文本或编码格式不对
        return f"错误：无法读取文件 '{filename}'，可能是编码格式非 UTF-8 或文件已损坏。"
    except PermissionError:
        # 处理权限不足的情况
        return f"错误：没有权限读取文件 '{filename}'。"
    except Exception as e:
        # [兜底处理] 捕获所有其他未知异常，防止服务崩溃
        return f"读取文件时发生未知错误：{str(e)}"

# ==========================================
# 3. 程序入口
# ==========================================
if __name__ == "__main__":
    # 启动 MCP 服务器，开始监听客户端连接
    # 此时脚本将阻塞运行，直到接收到停止信号
    print("正在启动桌面 TXT 文件助手服务...")
    mcp.run()
