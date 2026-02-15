
import dashscope
from dashscope.api_entities.dashscope_response import Role
import os

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

def get_completion(prompt, model="deepseek-v3"):
    messages = [{"role": "user", "content": prompt}]
    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message'
    )
    return response.output.choices[0].message.content

# 任务描述
instruction = """
你的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称，月费价格，月流量。
根据用户输入，识别用户在上述三种属性上的需求是什么。
"""

input = """
用户：我需要一个月费100元，月流量100GB的流量套餐。
"""

prompt = f"""
# 目标
{instruction}

# 输入
{input}
"""

print(prompt)
print()
print(get_completion(prompt))

# 输出格式，触发大模型输出JSON格式
output_format = """
输出格式为JSON
"""

prompt = f"""
# 目标
{instruction}

# 输入
{input}

# 输出格式
{output_format}
"""

print(prompt)
print()
print(get_completion(prompt))

instruction = """
给定一段用户与客服沟通手机流量套餐的对话。
你的任务是判断客服的回答是否符合下面的规范：

- 必须有礼貌
- 必须用官方口吻，不能使用网络用语
- 介绍套餐时，必须准确提及产品名称、月费价格和月流量总量。以上信息缺失一项或多项，或信息与事实不符，都算消息不准确
- 不可以是话题终结者

已知产品包括：

- 经济套餐：月费50元，月流量50GB
- 畅游套餐：月费200元，月流量300GB
- 无限套餐：月费300元，月流量2000GB
- 校园套餐：月费100元，月流量200GB，仅用于在校学生办理

"""

output_format = """
如果符合规范，输出：{"is_valid": true}
如果不符合规范，输出：{"is_valid": false, "reason": "原因"}
"""

context = """
用户：我需要一个月费100元，月流量100GB的流量套餐。
客服：我们有经济套餐，月费100元，月流量200GB。是否符合您的需求？
"""

cot = ""
# 触发长推理链
cot = "请一步一步分析对话"

prompt = f"""
# 目标
{instruction}
# 推理链
{cot}

# 输入
{context}

# 输出格式
{output_format}
"""

print(prompt)
print()
print(get_completion(prompt))

user_prompt = """
做一个手机流量套餐的客服代表，叫小瓜。可以帮助用户选择最合适的流量套餐产品。可以选择的套餐包括：
经济套餐，月费50元，10G流量；
畅游套餐，月费180元，100G流量；
无限套餐，月费300元，1000G流量；
校园套餐，月费150元，200G流量，仅限在校生。
"""

instruction = """
你是一名专业的提示词创作者，你的目标是帮助我根据我的需求打造更好的提示词。

你将生成以下部分：
提示词：{根据我的需求提供更好的提示词}
优化建议：{用简练段落分析如何改进提示词，需给出严格批判性建议}
问题示例：{提出最多3个问题，以用于和用户更好的交流}
"""

prompt = f"""
# 目标
{instruction}

# 用户提示词
{user_prompt}
"""

print(prompt)
print()
print(get_completion(prompt))

