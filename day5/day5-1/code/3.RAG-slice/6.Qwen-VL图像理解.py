import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import dashscope
from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import Role

# 加载环境变量，主要用于获取 DASHSCOPE_API_KEY
# 确保项目根目录下有 .env 文件，并包含 DASHSCOPE_API_KEY
load_dotenv()

# 设置 DashScope API Key
# 从环境变量中读取 API Key，如果不存在则会抛出错误或导致调用失败
api_key = os.getenv("DASHSCOPE_API_KEY")
if api_key:
    dashscope.api_key = api_key
else:
    print(" 警告: 未找到 DASHSCOPE_API_KEY 环境变量，请检查 .env 文件配置。")

def analyze_image(image_path: str, prompt: str, model: str = 'qwen-vl-plus') -> None:
    """
    使用 Qwen-VL 模型进行图像理解与分析。
    
    参数:
        image_path (str): 图像文件的路径（本地路径需以 file:// 开头，或使用 HTTP URL）。
        prompt (str): 用户输入的文本提示词（问题）。
        model (str): 使用的模型名称，默认为 'qwen-vl-plus'。
    """
    
    # 构建对话消息列表
    # 包含系统提示词和用户输入（图像+文本）
    messages = [
        {
            'role': 'system',
            'content': [{
                'text': 'You are a helpful assistant.' # 系统预设指令
            }]
        },
        {
            'role': 'user',
            'content': [
                {
                    'image': image_path # 图像内容
                },
                {
                    'text': prompt      # 文本问题
                },
            ]
        }
    ]
    
    print(f" 开始调用模型: {model}")
    print(f" 分析图片: {image_path}")
    print(f" 问题: {prompt}")
    print("-" * 50)

    try:
        # 调用 DashScope 的多模态对话接口
        # MultiModalConversation.call 用于发送请求并获取响应
        response = MultiModalConversation.call(model=model, messages=messages)
        
        # 检查响应状态
        if response.status_code == 200:
            # 解析并打印模型的回复内容
            # response.output.choices[0].message.content 是一个列表，通常包含文本回复
            if response.output and response.output.choices:
                content_list = response.output.choices[0].message.content
                # 遍历内容列表，找到文本类型的回复
                for item in content_list:
                    if 'text' in item:
                        print("\n 模型回复:")
                        print(item['text'])
            else:
                print(" 未收到有效回复内容")
        else:
            # 打印错误信息
            print(f" 调用失败: code={response.code}, message={response.message}")
            
    except Exception as e:
        print(f" 发生异常: {str(e)}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 定义本地图片路径
    # 注意：本地文件路径需要使用 'file://' 协议前缀
    # 请确保当前目录下存在名为 '6.万圣节.jpeg' 的文件，或者修改为实际存在的图片路径
    
    # 原代码使用的是相对路径 'file://6.万圣节.jpeg'，这依赖于运行时的当前工作目录
    local_file_path = 'file://6.万圣节.jpeg'
    
    # 定义用户提问
    question = '这是一张什么海报？'
    
    # 执行图像分析
    analyze_image(local_file_path, question)

"""
参考输出示例 (基于 6.万圣节.jpeg):

这张海报是上海迪士尼度假区（Shanghai Disney Resort）的万圣节主题活动宣传海报。海报的主题是“万圣趴 玩心大开”，英文标题为“HALLOWEEN TIME ROCK ON, WICKED FUN!”，意在邀请游客参与一场充满欢乐和刺激的万圣节派对。

### 海报详细解析：

1. **背景与主题**：
   - 海报的背景是一座色彩斑斓的迪士尼城堡，城堡被灯光照亮，呈现出粉红色、紫色和蓝色的梦幻效果，营造出一种神秘而欢乐的氛围。
   - 夜空中有烟花绽放，增添了节日的喜庆气氛。

2. **主要角色**：
   - **左侧**：一位身穿黑色长袍、手持绿色法杖的角色，显然是迪士尼的经典反派角色——玛琳菲森（Maleficent）。她的绿色皮肤和尖锐的角非常显眼，象征着万圣节的邪恶与神秘。
   - **右侧**：另一位穿着深色服装的角色，可能是迪士尼的另一个经典反派角色——乌苏拉（Ursula）。她戴着高高的帽子，表情狡黠，手中似乎握着某种权杖或法器。
   - **中间**：一只戴着金色皇冠的唐老鸭（Donald Duck），显得非常可爱和滑稽。他的表情和装扮与周围的反派角色形成鲜明对比，增加了海报的趣味性。

3. **文字信息**：
   - **顶部**：左上角有“SHANGHAI DISNEY RESORT 上海迪士尼度假区”的标志，表明这是在上海迪士尼度假区举办的活动。
   - **中间**：主标题“万圣趴 玩心大开”用金色的艺术字体书写，下方是英文标题“HALLOWEEN TIME ROCK ON, WICKED FUN!”，进一步强调了活动的主题和氛围。
   - **底部**：右下角有迪士尼的版权标志“©Disney”，表明这是官方发布的海报。

4. **整体设计**：
   - 海报采用了圆形的镜框设计，仿佛通过一面魔法镜子看到的场景，增加了视觉上的层次感和神秘感。
   - 色彩鲜艳，对比强烈，既有万圣节的黑暗元素，也有迪士尼的童话色彩，整体风格既恐怖又不失童趣。

### 总结：
这张海报成功地传达了上海迪士尼度假区万圣节主题活动的氛围，通过经典的迪士尼角色和丰富的视觉元素，吸引游客参与这场充满欢乐和刺激的万圣节派对。海报的设计不仅突出了万圣节的主题，还保留了迪士尼特有的童话魅力，适合各个年龄段的游客。
"""
