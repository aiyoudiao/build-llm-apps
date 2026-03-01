from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi # 推荐使用 Chat 模型
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import dashscope
from dotenv import load_dotenv

load_dotenv()

# 配置 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 初始化模型 (去掉 stream=True，LCEL 的 .stream() 方法会自动处理流式)
llm = ChatTongyi(model_name="qwen-turbo", dashscope_api_key=api_key)

# 1. 定义步骤
# 注意：所有 Prompt 的输入变量名最好保持一致，或者通过 RunnablePassthrough 明确传递
prompt_translate_en = ChatPromptTemplate.from_template("Translate this to English: {input}")
prompt_analyze = ChatPromptTemplate.from_template("Analyze this text briefly: {text}")
prompt_translate_cn = ChatPromptTemplate.from_template("Translate this back to Chinese: {text}")

# 2. 构建链
# 逻辑：
# 输入 {"input": "..."} 
# -> 翻译成英文 (输出字符串) 
# -> 映射为 {"text": "英文字符串"} 
# -> 分析 (输出字符串) 
# -> 映射为 {"text": "分析结果字符串"} 
# -> 回译 (输出最终字符串)

chain = (
    prompt_translate_en 
    | llm 
    | StrOutputParser() 
    | (lambda x: {"text": x})  # 【关键】将上一步的字符串结果包装成字典，供下一步使用
    | prompt_analyze 
    | llm 
    | StrOutputParser() 
    | (lambda x: {"text": x})  # 【关键】再次包装
    | prompt_translate_cn 
    | llm 
    | StrOutputParser()
)

# 3. 执行与流式输出
input_text = "北京有哪些好吃的地方，简略回答不超过 200 字"
print(f"开始处理：{input_text}\n---\n")

try:
    # .stream() 会流式输出最后一步的结果
    # 注意：中间的翻译和分析步骤通常会等待完成后再进行下一步，
    # 只有最后一步是实时的打字机效果。这是串行链的物理限制。
    for chunk in chain.stream({"input": input_text}):
        print(chunk, end="", flush=True)
    print("\n---\n处理完成")
except Exception as e:
    print(f"\n发生错误：{e}")
