import os
import torch
import threading
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/files
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_dir = f"../../../models/{model_name}"

# 判断目录下是否包含该模型
if not os.path.exists(model_dir):
    print(f"开始下载模型 {model_name}")
    snapshot_download(model_name, cache_dir="./models")
    print(f"{model_name} 下载完成")
else:
    print(f"模型 {model_name} 已存在于 {model_dir}")
    print("请勿重复下载")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"使用设备 {device}")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
prompt = "帮我写一个二分查找法"
messages = [
    {"role": "system", "content": "你是一个专业的编程助手，你的任务是根据用户的问题，生成符合要求的代码。"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 一次性输出
# generated_ids = model.generate(**model_inputs, max_new_tokens=2000)
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)

# 分批次输出
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(
    **model_inputs,
    max_new_tokens=2000,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer,
)

thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    print(new_text, end="", flush=True)
print()



