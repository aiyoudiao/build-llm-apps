from langchain.prompts  import PromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
import dashscope
import os
from dotenv import load_dotenv

load_dotenv()

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key
# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型

# 创建Prompt Template
prompt = PromptTemplate(
    input_variables=["product"],
    template="一家生产{product}的公司，取什么名字比较好？",
)

# 新推荐用法：将 prompt 和 llm 组合成一个"可运行序列"
chain = prompt | llm

# 使用 invoke 方法传入输入
result1 = chain.invoke({"product": "五彩袜子"})
print(result1)
