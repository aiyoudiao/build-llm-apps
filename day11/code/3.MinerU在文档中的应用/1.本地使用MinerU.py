import subprocess
import sys
import os
from pathlib import Path

# https://mineru.net/
# uv run mineru -p ./files/Qwen3-tech_report.pdf -o ./output -b pipeline

def run_mineru_command():
    # 1. 定义输入输出路径
    input_pdf = "./files/Qwen3-tech_report.pdf"
    output_dir = "./output"

    # 2. 基础校验
    if not Path(input_pdf).exists():
        print(f"❌ 错误：找不到文件 -> {input_pdf}")
        return False
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 3. 构造命令列表
    # 对应: uv run mineru -p <input> -o <output> -b pipeline --device cpu
    command = [
        "uv", 
        "run", 
        "mineru", 
        "-p", input_pdf, 
        "-o", output_dir, 
        "-b", "pipeline", 
        "--device", "cpu"
    ]

    print(f"🚀 开始执行命令: {' '.join(command)}")
    print("-" * 50)

    try:
        # 4. 执行命令
        # shell=False (使用列表形式) 更安全，能正确处理路径中的空格
        # check=True: 如果命令返回非零退出码（报错），则抛出 CalledProcessError
        # text=True: 将输出解码为字符串
        # bufsize=1, flush=True: 实现实时打印日志（流式输出）
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将错误输出合并到标准输出
            text=True,
            bufsize=1
        )

        # 实时打印输出（带过滤）
        if process.stdout:
            for line in process.stdout:
                # 定义需要过滤的警告关键词
                skip_keywords = [
                    "Could not get FontBBox", 
                    "FontBBox from font descriptor"
                ]
                
                # 如果行中包含这些关键词，则跳过不打印
                if any(keyword in line for keyword in skip_keywords):
                    continue
                
                print(line, end="")
                sys.stdout.flush()

        # 等待进程结束并获取返回码
        process.wait()

        if process.returncode == 0:
            print("-" * 50)
            print(f"✅ 执行成功！结果已保存至: {output_dir}")
            return True
        else:
            print("-" * 50)
            print(f"❌ 执行失败，退出码: {process.returncode}")
            return False

    except FileNotFoundError:
        print("❌ 错误：找不到 'uv' 命令。请确保已安装 uv 并配置在环境变量中。")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行 (Ctrl+C)。正在终止子进程...")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_mineru_command()
    # 如果需要在外部脚本根据此脚本的结果做判断，可以在此退出
    sys.exit(0 if success else 1)
