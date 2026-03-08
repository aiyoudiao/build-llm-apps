#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发规范查询助手 (Development Specification Query Assistant)

【功能概述】
本脚本基于 MCP (Model Context Protocol) 协议构建一个轻量级知识库服务。
它将主流编程语言的开发规范和最佳实践硬编码在内存中，为 AI 模型提供即时、准确的
编码标准查询能力。AI 可利用此服务回答关于代码风格、命名约定、错误处理等问题。

【支持语言】
当前内置支持以下 5 种语言的规范：
- Python (PEP 8)
- JavaScript (ESLint/ES6+)
- Java (Oracle Standards)
- Go (Go Idioms)
- Rust (Rustfmt/Ownership)

【工作流程】
1. 初始化 FastMCP 服务器实例。
2. 加载内置的 `DEV_SPECS` 字典数据库（内存驻留，无需外部文件）。
3. 注册三个核心工具函数：
   - get_dev_spec: 查询单一语言的详细规范。
   - list_supported_languages: 获取支持的语言列表。
   - get_all_specs: 获取所有语言的规范汇总。
4. 启动服务器，响应 AI 客户端的查询请求。
5. 接收请求后，动态检索字典数据，格式化为 Markdown 并返回。

【特点与优势】
- 零依赖文件：所有数据内嵌于代码，部署简单，无文件路径权限问题。
- 快速响应：内存直接读取，无 I/O 延迟。
- 格式化输出：返回内容预置 Markdown 语法，便于 AI 直接渲染展示。
- 输入容错：自动处理语言名称的大小写和多余空格。

【扩展建议】
若需添加新语言或更新规范，只需修改 `DEV_SPECS` 字典结构，无需更改逻辑代码。

【使用方法】
1. 运行本脚本 mcp dev 2.开发规范查询助手.py
2. 确保 MCP 客户端已启动并连接至本服务器。
3. 在 MCP 客户端中，发送以下指令：
   - get_dev_spec("python")：查询 Python 开发规范。
   - list_supported_languages()：列出所有支持的语言。
   - get_all_specs()：获取所有语言的规范汇总。
"""

import json
from mcp.server.fastmcp import FastMCP

# ==========================================
# 1. 初始化 MCP 服务器
# ==========================================
# 创建名为 "开发规范文档查询器" 的 MCP 服务实例
mcp = FastMCP("开发规范文档查询器")

# ==========================================
# 2. 内置知识库数据 (In-Memory Database)
# ==========================================
# 存储各语言开发规范的字典结构
# 键 (Key): 语言标识符 (小写)
# 值 (Value): 包含名称、版本、描述及具体指南字典的对象
DEV_SPECS = {
    "python": {
        "name": "Python开发规范",
        "version": "1.0",
        "description": "Python项目开发规范和最佳实践",
        "guidelines": {
            "code_style": "使用PEP 8编码规范",
            "naming_convention": "变量名使用snake_case，类名使用PascalCase",
            "docstring": "所有函数和类都应包含docstring",
            "imports": "导入语句应按标准库、第三方库、本地库的顺序分组",
            "error_handling": "合理使用try-except处理异常",
            "testing": "编写单元测试，覆盖率应达到80%以上"
        }
    },
    "javascript": {
        "name": "JavaScript开发规范",
        "version": "1.0",
        "description": "JavaScript项目开发规范和最佳实践",
        "guidelines": {
            "code_style": "使用ESLint进行代码检查",
            "naming_convention": "变量和函数名使用camelCase",
            "semicolons": "在语句结束时使用分号",
            "const_let": "优先使用const，必要时使用let",
            "arrow_functions": "在适当情况下使用箭头函数",
            "promises_async_await": "使用async/await处理异步操作"
        }
    },
    "java": {
        "name": "Java开发规范",
        "version": "1.0",
        "description": "Java项目开发规范和最佳实践",
        "guidelines": {
            "code_style": "遵循Oracle Java编码规范",
            "naming_convention": "类名使用PascalCase，方法和变量使用camelCase",
            "packages": "合理组织包结构，使用小写字母",
            "access_modifiers": "正确使用public、private、protected修饰符",
            "exception_handling": "使用try-catch-finally处理异常",
            "javadoc": "为公共API编写Javadoc注释"
        }
    },
    "go": {
        "name": "Go开发规范",
        "version": "1.0",
        "description": "Go项目开发规范和最佳实践",
        "guidelines": {
            "code_style": "使用gofmt格式化代码",
            "naming_convention": "导出的标识符使用PascalCase，内部使用camelCase",
            "packages": "小写包名，使用简洁描述性名称",
            "error_handling": "始终检查错误返回值",
            "documentation": "为导出的函数和类型编写注释",
            "interfaces": "小接口优先，单方法接口常见"
        }
    },
    "rust": {
        "name": "Rust开发规范",
        "version": "1.0",
        "description": "Rust项目开发规范和最佳实践",
        "guidelines": {
            "code_style": "使用rustfmt格式化代码",
            "naming_convention": "常量使用SCREAMING_SNAKE_CASE，其他使用snake_case",
            "ownership": "理解所有权、借用和生命周期",
            "error_handling": "使用Result和Option处理错误",
            "traits": "合理使用trait定义共享行为",
            "documentation": "使用rustdoc编写文档"
        }
    }
}

# ==========================================
# 3. 定义工具函数 (Tools)
# ==========================================

@mcp.tool()
def get_dev_spec(language: str) -> str:
    """
    获取指定编程语言的详细开发规范文档
    
    Args:
        language (str): 编程语言名称 (例如: "python", "JavaScript", " GO ")
        
    Returns:
        str: 格式化的 Markdown 文档，包含规范详情；若语言不支持则返回错误提示
    """
    # [输入预处理] 统一转换为小写并去除首尾空格，提高匹配容错率
    language = language.lower().strip()
    
    # [核心逻辑] 在内置数据库中查找对应的语言规范
    if language in DEV_SPECS:
        spec = DEV_SPECS[language]
        
        # [构建响应] 组装 Markdown 格式的回复
        # 标题部分
        result = f"## {spec['name']} (版本: {spec['version']})\n"
        # 描述部分
        result += f"**描述**: {spec['description']}\n\n"
        result += "**开发指南**:\n"
        
        # 遍历具体指南项，格式化输出
        # 将 key 中的下划线替换为空格并首字母大写 (e.g., code_style -> Code Style)
        for key, value in spec['guidelines'].items():
            formatted_key = key.replace('_', ' ').title()
            result += f"- **{formatted_key}**: {value}\n"
        
        return result
    else:
        # [异常处理] 语言不存在时，返回友好提示及支持列表
        available_languages = ", ".join(DEV_SPECS.keys())
        return f"错误：不支持的语言 '{language}'。\n支持的语言有: {available_languages}"


@mcp.tool()
def list_supported_languages() -> str:
    """
    获取当前知识库支持的所有编程语言列表
    
    Returns:
        str: 格式化的语言列表，包含标识符和全称
    """
    languages = list(DEV_SPECS.keys())
    
    if languages:
        result = "支持的编程语言:\n"
        # 遍历并展示每个语言的标识符和人类可读的名称
        for lang in languages:
            lang_info = DEV_SPECS[lang]
            result += f"- `{lang}`: {lang_info['name']}\n"
        return result
    else:
        return "当前没有支持的编程语言 (数据库为空)"


@mcp.tool()
def get_all_specs() -> str:
    """
    获取所有支持语言的开发规范文档汇总
    
    Returns:
        str: 包含所有语言规范的完整 Markdown 文档
    """
    if not DEV_SPECS:
        return "当前没有开发规范文档"
    
    result = "# 所有支持的开发规范汇总\n\n"
    
    # 遍历整个数据库，拼接每个语言的规范内容
    for lang, spec in DEV_SPECS.items():
        result += f"## {spec['name']} ({lang})\n"
        result += f"**描述**: {spec['description']}\n\n"
        result += "**开发指南**:\n"
        
        for key, value in spec['guidelines'].items():
            formatted_key = key.replace('_', ' ').title()
            result += f"- **{formatted_key}**: {value}\n"
        
        result += "\n---\n\n"  # 添加分隔线以区分不同语言
    
    return result

# ==========================================
# 4. 程序入口
# ==========================================
if __name__ == "__main__":
    # 启动 MCP 服务器
    # 此操作将阻塞进程，直到接收到终止信号
    print("正在启动开发规范查询助手服务...")
    mcp.run()
