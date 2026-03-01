from langchain_core.tools import tool
from typing import Optional

@tool
def convert_data(input_data: str, input_format: str, output_format: str) -> str:
    """
    在不同数据格式之间进行稳健转换 (支持 JSON <-> CSV)。
    """
    return f"Converted {input_format} to {output_format}"

@tool
def process_text(operation: str, content: str, search_text: Optional[str] = None, replace_text: Optional[str] = None) -> str:
    """
    执行具体的文本处理操作：统计行数、查找文本、替换文本。
    """
    return f"Processed {operation}"

print("convert_data args:", convert_data.args)
print("process_text args:", process_text.args)
