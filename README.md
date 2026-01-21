# Polymath (ex-Scribe)

**Polymath** 是一个基于 **MCP (Model Context Protocol)** 架构的通用 AI 智能体，旨在成为你的全能数字副手。它不仅能写作，还能阅读代码、分析数据、识别图像，并拥有长期记忆。

## 🌟 核心特性

*   **真正的 MCP 架构:** 采用 Client-Server 模式，能力无限扩展。
*   **多模态视觉 (Eyes):** 支持 **Gemini Vision** 和 **OpenAI GPT-4o**，支持 OCR 和图像理解。
*   **长期记忆 (Memory):** 基于 **ChromaDB** 的本地向量记忆库，支持按项目隔离。
*   **全能文件读写:** 支持 PDF, Word, PPT, Excel 及纯文本的深度解析与生成。
*   **代码执行 (Computer):** 内置 Python 解释器，可进行数据分析和绘图。
*   **安全沙箱:** 文件操作限制在项目目录内，敏感操作需人工审批 (HITL)。

## Features v0.4 (Multi-Mode Agent)
- **Expert Modes:** Switch between specialized personas to get the best results.
  - **Planner (Default):** Breaks down tasks and coordinates execution.
  - **Coder (`/mode coder`):** Focuses on code quality, testing, and implementation.
  - **Architect (`/mode architect`):** Focuses on system design and documentation.
- **Code Intelligence:**
  - **Smart Outline:** Uses AST parsing (`read_file_outline`).
  - **Symbol Navigation:** Find definitions instantly (`find_symbol`).
- **Project Memory:** `/init` to create `POLYMATH.md`.
- **Task Management:** Built-in Todo list (`task` server).
- **Transparency:** Visual Diffs & Chain of Thought visibility.

## Installation

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/polymath-ai/polymath.git
    cd polymath
    ```

2.  **安装 (推荐使用 pipx 或 venv):**
    ```bash
    pip install .
    ```
    
    安装成功后，系统会多出一个 `pl` (或 `polymath`) 命令。

## ⚙️ 配置

Polymath 会按以下顺序查找配置：
1.  当前目录: `.polymath/config.json` (项目级)
2.  用户目录: `~/.polymath/config.json` (全局级)
3.  默认配置: 内置于安装包中

建议在项目根目录下创建 `.env` 文件来管理密钥：
```bash
GOOGLE_API_KEY="your_google_api_key"
OPENAI_API_KEY="your_openai_api_key" # 可选
```

## 🚀 使用方法

### 命令行模式 (CLI)
这是主要的交互方式。支持项目隔离。

```bash
# 在任意目录下启动 Polymath
pl start

# 启动特定项目 (记忆库独立)
pl start --project "ProjectAlpha"
```

### Web 调试模式 (ADK)
使用 Google ADK 提供的 Web UI 可视化调试工具调用。

```bash
adk web src/agent.py
```

## 📂 项目结构

*   `src/host/`: MCP Client 主程序 (CLI)。
*   `src/servers/`: 独立的 MCP Servers (Memory, Vision, Filesystem, etc.)。
*   `materials/`: 默认的资料存放目录。
*   `drafts/`: AI 生成内容的默认保存目录。
*   `.polymath/`: 存储配置和长期记忆数据库。

## 🛡️ 安全机制

*   **HITL (Human-in-the-loop):** 当 AI 尝试执行代码、写文件或修改记忆时，CLI 会弹出醒目的**红色警示**并显示具体参数。
*   **Path Jailing:** 文件系统操作被限制在当前工作目录，防止越权访问。