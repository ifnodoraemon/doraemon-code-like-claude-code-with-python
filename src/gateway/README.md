# Model Gateway

统一的 AI 模型网关，支持多家模型提供商。

## 两种使用模式

### Mode 1: Direct Mode（直连模式）
用户自己配置各家模型的 API Key，CLI 直接调用各家 API。

```
┌─────────────────┐
│   Agent CLI     │
│  ┌───────────┐  │
│  │ModelClient│──┼──► Google API (GOOGLE_API_KEY)
│  │ (Direct)  │──┼──► OpenAI API (OPENAI_API_KEY)
│  └───────────┘──┼──► Anthropic API (ANTHROPIC_API_KEY)
└─────────────────┘
```

**适用场景**: 个人开发者、快速开始

### Mode 2: Gateway Mode（网关模式）
用户只配置 Gateway URL 和 Key，Gateway 服务器管理所有 API Key。

```
┌─────────────────┐     ┌─────────────────────────────────┐
│   Agent CLI     │     │       Model Gateway Server      │
│  ┌───────────┐  │     │  ┌─────────────────────────┐    │
│  │ModelClient│──┼────►│  │  Provider Adapters      │    │
│  │ (Gateway) │  │     │  │  ├─► Google (API Key)   │    │
│  └───────────┘  │     │  │  ├─► OpenAI (API Key)   │    │
└─────────────────┘     │  │  └─► Anthropic (API Key)│    │
只需: URL + Key         │  └─────────────────────────┘    │
                        └─────────────────────────────────┘
```

**适用场景**: 团队协作、企业部署、SaaS 服务

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent CLI                                    │
│          只需配置: AGENT_GATEWAY_URL + AGENT_API_KEY                 │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Model Gateway Server                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Unified API (OpenAI Compatible)              │  │
│  │  POST /v1/chat/completions  - Chat (streaming supported)       │  │
│  │  GET  /v1/models            - List models                      │  │
│  │  GET  /health               - Health check                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                 │                                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Provider Adapters                           │  │
│  │  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐            │  │
│  │  │ Google │  │ OpenAI │  │Anthropic │                        │  │
│  │  │ Gemini │  │  GPT   │  │  Claude  │                        │  │
│  │  └────────┘  └────────┘  └──────────┘                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn httpx google-genai openai anthropic
```

### 2. 配置环境变量

```bash
# Server 端（提供各家模型 API Key）
export GOOGLE_API_KEY="your-google-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# 可选：设置 Gateway 的 API Key
export AGENT_API_KEY="your-gateway-key"
```

### 3. 启动 Gateway Server

```bash
# 方式 1: 使用模块
python -m src.gateway.server

# 方式 2: 使用 uvicorn
uvicorn src.gateway.server:app --host 0.0.0.0 --port 8000
```

### 4. CLI 端配置

CLI 只需要两个配置：

```bash
# .env 文件
AGENT_GATEWAY_URL=http://localhost:8000
AGENT_API_KEY=your-gateway-key
```

## API 使用

### Chat Completion

```python
from src.gateway import GatewayClient

async with GatewayClient() as client:
    # 使用任意模型，Gateway 自动路由
    response = await client.chat(
        model="gemini-2.5-flash",  # 或 "gpt-4o", "claude-sonnet", etc.
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
    )
    print(response["choices"][0]["message"]["content"])
```

### Streaming

```python
async with GatewayClient() as client:
    async for chunk in client.chat_stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a poem"}],
    ):
        if chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)
```

### List Models

```python
async with GatewayClient() as client:
    models = await client.list_models()
    for model in models:
        print(f"{model['id']}: {model['name']} ({model['provider']})")
```

## 支持的模型

### Google Gemini
- `gemini-2.5-pro-preview` (aliases: g25p)
- `gemini-2.5-flash-preview` (aliases: flash, g25f)
- `gemini-2.0-flash`

### OpenAI
- `gpt-4o` (aliases: 4o)
- `gpt-4o-mini` (aliases: mini, 4o-mini)
- `o1` (reasoning model)
- `o3-mini` (fast reasoning)

### Anthropic Claude
- `claude-sonnet-4-20250514` (aliases: sonnet)
- `claude-opus-4-20250514` (aliases: opus)
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022` (aliases: haiku)

## 模型路由

Gateway 根据模型 ID 前缀自动路由：

| 前缀 | Provider |
|------|----------|
| `gemini-` | Google |
| `gpt-`, `o1`, `o3` | OpenAI |
| `claude-` | Anthropic |

## 配置选项

### Server 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `GOOGLE_API_KEY` | Google API Key | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `ANTHROPIC_API_KEY` | Anthropic API Key | - |
| `AGENT_API_KEY` | Gateway API Key | - |
| `GATEWAY_HOST` | 监听地址 | 0.0.0.0 |
| `GATEWAY_PORT` | 监听端口 | 8000 |

### Client 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `AGENT_GATEWAY_URL` | Gateway 地址 | http://localhost:8000 |
| `AGENT_API_KEY` | Gateway API Key | - |
| `AGENT_GATEWAY_TIMEOUT` | 请求超时(秒) | 120 |

## 扩展

添加新的 Provider：

1. 在 `adapters/` 下创建新的 adapter 文件
2. 继承 `BaseAdapter` 类
3. 实现 `chat()`, `chat_stream()`, `get_models()` 方法
4. 在 `router.py` 中注册

```python
from .adapters.base import BaseAdapter

class MyProviderAdapter(BaseAdapter):
    provider_name = "myprovider"

    async def initialize(self) -> None:
        # 初始化客户端
        pass

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # 实现聊天
        pass

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        # 实现流式聊天
        pass

    def get_models(self) -> list[ModelInfo]:
        # 返回支持的模型列表
        pass
```
