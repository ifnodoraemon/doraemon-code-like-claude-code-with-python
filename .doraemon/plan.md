# Plan: AskUser 工具 + 图片/多模态输入

## 概览

```
┌─────────────────────────────────────────────────────────────┐
│  两项新能力的实现计划                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ask_user 工具 — LLM 通过工具调用向用户提问               │
│  2. 图片输入 — 用户通过 @image.png 发送图片给 LLM            │
│                                                             │
│  改动文件:                                                   │
│    新建: src/servers/ask_user.py          (ask_user 实现)     │
│    修改: src/host/tools.py               (注册 ask_user)     │
│    修改: src/core/tool_selector.py       (ask_user 加入模式)  │
│    修改: src/host/cli/tool_execution.py  (ask_user 特殊处理)  │
│    修改: src/core/model_utils.py         (content 多模态)     │
│    修改: src/core/model_client_direct.py (图片 parts 转换)    │
│    修改: src/host/cli/chat_loop.py       (@image 展开)       │
│    修改: src/core/context_manager.py     (图片序列化)         │
│                                                             │
│  不改动: src/services/vision.py (已有,但本次不用它)           │
│  思路: 图片直接注入到主对话流, 不走旁路 vision 适配器          │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: ask_user 工具

### 1.1 新建 `src/servers/ask_user.py`

```python
# 核心函数签名
def ask_user(
    question: str,
    options: str = "",      # 逗号分隔的选项, 如 "Yes,No,Maybe"
    multi_select: bool = False,
) -> str:
    """向用户提问并等待回答。

    当你需要用户输入才能继续时使用此工具。
    可以是自由文本输入, 也可以提供选项列表。

    Args:
        question: 要问用户的问题
        options: 可选的选项列表(逗号分隔), 如 "TypeScript,Python,Go"
        multi_select: 为 True 时允许用户选择多个选项

    Returns:
        用户的回答文本
    """
```

**实现逻辑:**
- 无 options → `Prompt.ask(question)` 自由输入
- 有 options → 显示编号列表 + "Other" 选项, 用户输入编号或自定义
- multi_select → 允许逗号分隔的多选 "1,3"
- 使用 Rich Panel 渲染问题, 视觉上突出

**为什么 options 是 str 而不是 list:**
工具参数自动从 Python 类型签名提取 JSON Schema。当前 `_extract_parameters` 只支持基础类型 (str/int/float/bool/list/dict)。用逗号分隔字符串更简单, 避免复杂的 list[str] schema 处理。

### 1.2 修改 `src/host/tools.py` — 注册

在 `_create_default_registry()` 中添加:
```python
from src.servers.ask_user import ask_user
registry.register(ask_user, sensitive=False, timeout=300.0)  # 等待用户输入可能较久
```

### 1.3 修改 `src/core/tool_selector.py` — 加入所有模式

ask_user 放入 `READ_TOOLS` (plan 和 build 都可用):
```python
READ_TOOLS = [
    "read",
    "search",
    "ask_user",  # 新增
]
```

### 1.4 修改 `src/host/cli/tool_execution.py` — 特殊处理

ask_user 不走常规 HITL 流程 (它本身就是人机交互), 但需要在 headless 模式下拒绝:
```python
# 在 execute_tool 函数开头, 紧接 hook 处理之后:
if tool_name == "ask_user" and headless:
    return {
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "result": "Headless mode: Cannot ask user for input.",
    }
```

---

## Part 2: 图片/多模态输入

### 2.1 修改 `src/core/model_utils.py` — Message.content 支持多模态

```python
@dataclass
class Message:
    """Unified message format."""
    role: str
    content: str | list[dict] | None = None  # str 或多模态 parts
    ...

    def to_dict(self) -> dict:
        # content 可能是 str 或 list, 直接序列化
        ...
```

新增辅助函数:
```python
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}

def make_image_part(image_path: str) -> dict:
    """将图片文件转为多模态 content part (base64)."""
    ...
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime_type, "data": b64_data},
        "path": image_path,  # 元数据, 用于显示/序列化
    }

def make_text_part(text: str) -> dict:
    return {"type": "text", "text": text}

def get_content_text(content: str | list[dict] | None) -> str:
    """从 content 中提取纯文本(用于 token 估算/摘要)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""
```

### 2.2 修改 `src/host/cli/chat_loop.py` — @image 展开

在 `expand_file_references()` 中, 当检测到图片扩展名时, 不再将文件内容作为文本展开, 而是返回一个特殊标记, 稍后在主循环中解析为多模态 content:

```python
# 新函数: 解析用户输入中的图片引用
def parse_image_references(text: str) -> tuple[str, list[str]]:
    """从用户输入中提取 @image.png 引用。

    Returns:
        (clean_text, image_paths) - 去除图片引用后的文本 + 图片路径列表
    """
```

在主循环的 `user_input` 处理阶段 (line ~952), 在 `expand_file_references` 之前:
```python
# 先提取图片引用
clean_text, image_paths = parse_image_references(user_input)
user_input = expand_file_references(clean_text)

# 构建 content (文本 or 多模态)
if image_paths:
    content_parts = [make_text_part(user_input)]
    for img_path in image_paths:
        content_parts.append(make_image_part(img_path))
    user_message = Message(role="user", content=content_parts)
else:
    user_message = Message(role="user", content=user_input)
```

### 2.3 修改 `src/core/model_client_direct.py` — Provider 图片转换

**Google (_chat_google / _stream_google):**
```python
if msg.get("content"):
    content = msg["content"]
    if isinstance(content, str):
        parts.append(types.Part(text=content))
    elif isinstance(content, list):
        for part in content:
            if part["type"] == "text":
                parts.append(types.Part(text=part["text"]))
            elif part["type"] == "image":
                source = part["source"]
                parts.append(types.Part.from_bytes(
                    data=base64.b64decode(source["data"]),
                    mime_type=source["media_type"],
                ))
```

**OpenAI (_chat_openai / _stream_openai):**
OpenAI 的 messages API 原生支持 `content: list`, 格式为:
```python
content = [
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]
```
直接将我们的格式转换即可。

**Anthropic (_chat_anthropic / _stream_anthropic):**
Anthropic 的格式:
```python
content = [
    {"type": "text", "text": "..."},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
]
```
这和我们的内部格式完全一致, 几乎零转换。

**Ollama:**
不支持图片, 回退到纯文本 (提取 text parts)。

### 2.4 修改 `src/core/context_manager.py` — 图片序列化

- `Message.content` 现在可能是 list, `to_dict()` / `from_dict()` 需兼容
- token 估算: 图片按固定 token 数估算 (如 1000 tokens/image)
- 摘要时: 只提取文本部分, 忽略图片 (避免 JSON 膨胀)
- 持久化: 保存时将 base64 数据写入临时文件, JSON 中只存文件引用 (防止 conversation.json 膨胀)

**简化方案 (v1):** 持久化时直接丢弃图片 base64 数据, 只保留 `[Image: path]` 占位符。图片是一次性上下文, 后续对话可以通过重新 `@image.png` 引用。

---

## 实现顺序

```
Step 1: ask_user 工具 (4 个文件)
  ├── 新建 src/servers/ask_user.py
  ├── 修改 src/host/tools.py (注册)
  ├── 修改 src/core/tool_selector.py (加入模式)
  └── 修改 src/host/cli/tool_execution.py (headless 处理)

Step 2: 多模态基础设施 (2 个文件)
  ├── 修改 src/core/model_utils.py (Message.content 类型 + 辅助函数)
  └── 修改 src/core/context_manager.py (兼容多模态序列化)

Step 3: Provider 适配 (1 个文件)
  └── 修改 src/core/model_client_direct.py (4 个 provider 的图片转换)

Step 4: CLI 图片输入 (1 个文件)
  └── 修改 src/host/cli/chat_loop.py (@image 解析 + 多模态 Message 构建)
```

每个 step 独立 commit。
