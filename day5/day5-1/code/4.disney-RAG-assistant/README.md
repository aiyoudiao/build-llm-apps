# 迪士尼多模态 RAG 助手使用说明

一个基于检索增强生成 (RAG) 的多模态问答助手，能够理解 Word 文档和图片内容。

## 1. 环境准备

确保您的系统满足以下要求：
- Python 3.11.14 (或其他 3.8+ 版本)
- 虚拟环境 venv (建议使用)

## 2. 安装依赖

请在虚拟环境中安装所需的 Python 库。

```bash
# 激活虚拟环境
source "venv/bin/activate"

# 安装依赖
pip install openai faiss-cpu python-docx PyMuPDF Pillow pytesseract transformers torch requests python-dotenv
```

## 3. 安装 OCR 引擎 (可选但推荐)

脚本使用 Tesseract OCR 引擎来提取图片中的文字。如果未安装，脚本仍然可以运行，但将无法提取图片中的文字信息，只能依赖 CLIP 模型进行图像特征匹配。

**macOS 安装:**
```bash
brew install tesseract
```

**Windows 安装:**
请访问 [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) 下载安装包。

## 4. 配置 API Key

脚本需要阿里云百炼 (DashScope) 的 API Key 来调用 Embedding 模型和大语言模型。

1. 在 `code/4.disney-RAG-assistant/` 目录下创建一个名为 `.env` 的文件。
2. 在文件中添加以下内容：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
*(请将 `sk-xxx` 替换为您实际的 API Key)*

## 5. 运行脚本

确保您已激活虚拟环境，并在 `code/4.disney-RAG-assistant/` 目录下运行：

```bash
cd "day5-1/code/4.disney-RAG-assistant/"
python multimodal-bot.py
```

## 6. 代码逻辑说明

脚本的主要执行流程如下：

1.  **初始化**: 加载环境变量，初始化 OpenAI 客户端和 CLIP 模型。
2.  **构建知识库 (支持增量更新)**:
    *   **哈希检测**: 自动扫描文档和图片目录，计算文件 MD5 哈希值。
    *   **增量判断**: 如果本地存在向量库且文件哈希值未变，则直接加载本地索引（秒级启动）。
    *   **重建索引**: 如果检测到文件变化或无本地索引，则执行解析、OCR、Embedding 和建库流程，并保存到 `vector_store` 目录。
3.  **内容处理**:
    *   **解析 Word 文档**: 使用 `python-docx` 提取文本和表格。
    *   **处理图片**: 使用 `pytesseract` 进行 OCR 识别，使用 `CLIP` 提取图像特征向量。
    *   **生成向量**: 使用阿里云 `text-embedding-v4` 生成文本向量。
    *   **建立索引**: 使用 `faiss` 建立向量索引，支持快速检索。
4.  **RAG 问答**:
    *   **检索**: 将用户问题向量化，检索最相关的文本片段和图片。
    *   **生成**: 将检索到的上下文组装成 Prompt，调用 `qwen-plus` 大模型生成回答。

## 7. 常见问题与故障排除

### 1. Segmentation Fault (段错误)
如果在 macOS 上运行脚本时遇到 `Segmentation fault` 错误，这通常是由于 `tokenizers` 库的多进程管理与 macOS 的系统库冲突，或者是 PyTorch 与 OpenMP 的兼容性问题。

为了解决这个问题，我们在代码开头添加了以下环境变量配置：
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizers 并行
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"     # 允许 OpenMP 库重复加载
os.environ["OMP_NUM_THREADS"] = "1"             # 限制 OpenMP 线程数
```
此外，代码中还显式将 CLIP 模型加载到 CPU 上运行：`clip_model.to('cpu')`。

### 2. 加载 CLIP 模型慢
首次运行时需要下载 CLIP 模型 (约 600MB)，请保持网络通畅。如果下载失败，请检查网络连接或尝试手动下载模型。

### 3. 报错 `command not found: tesseract`
说明未安装 Tesseract OCR 引擎。您可以按照第 3 步安装它，或者忽略此错误 (脚本会自动捕获异常并跳过文字识别，仅使用 CLIP 进行图像特征匹配)。

### 4. 报错 `Please set 'DASHSCOPE_API_KEY'`
请检查 `.env` 文件是否配置正确，或者直接在系统环境变量中设置该 Key。
