# 性能优化计划

## 1. 当前性能基线

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 启动时间 | ~2s | <0.5s |
| 内存占用 | ~300MB | <100MB |
| 首次响应 | ~500ms | <200ms |
| 工具调用延迟 | ~100ms | <50ms |
| 安装包大小 | ~500MB | <100MB |

## 2. 启动优化

### 2.1 延迟加载 (已部分实现)

```python
# 当前: 启动时加载所有 36 个工具
registry = get_default_registry()  # ~2s

# 优化: 只加载核心工具，其他按需加载
CORE_TOOLS = ["read", "write", "search", "run"]  # 4 个核心
OPTIONAL_TOOLS = [...]  # 32 个可选，按需加载
```

### 2.2 懒加载 MCP 服务器

```python
# 当前: 启动时连接所有 MCP 服务器
await registry.connect_all()

# 优化: 按需连接
async def get_tool(name):
    if name in mcp_tools and not connected:
        await connect_server(mcp_tools[name])
    return await call_tool(name, args)
```

## 3. 内存优化

### 3.1 AgentState 消息限制

```python
# 当前: 无限制
class AgentState:
    messages: list[Message]  # 可能无限增长

# 优化: 固定窗口 + 压缩
class AgentState:
    max_messages: int = 50
    
    def add_message(self, msg):
        self.messages.append(msg)
        if len(self.messages) > self.max_messages:
            self._compress_old_messages()
```

### 3.2 流式处理大文件

```python
# 当前: 一次性读取
content = file.read_text()  # 大文件占内存

# 优化: 流式处理
async def read_large_file(path, chunk_size=8192):
    async with aiofiles.open(path) as f:
        async for chunk in f:
            yield chunk
```

### 3.3 工具结果截断

```python
# 已实现: MAX_RESULT_LENGTH = 50000
# 优化: 更智能的截断
MAX_RESULT_LENGTH = 10000  # 减少
TRUNCATION_STRATEGY = "smart"  # 保留关键部分
```

## 4. 并发优化

### 4.1 并行工具调用

```python
# 当前: 串行执行
for tool_call in tool_calls:
    result = await execute_tool(tool_call)

# 优化: 并行执行独立工具
async def execute_tools_parallel(tool_calls):
    independent = group_independent_calls(tool_calls)
    results = await asyncio.gather(*[
        execute_tool(tc) for tc in independent
    ])
    return results
```

### 4.2 预取和缓存

```python
# 添加工具结果缓存
class ToolCache:
    _cache: dict[str, str] = {}
    _ttl: int = 300  # 5 分钟
    
    @classmethod
    async def get_or_call(cls, tool_name, args):
        key = f"{tool_name}:{hash(args)}"
        if key in cls._cache:
            return cls._cache[key]
        result = await call_tool(tool_name, args)
        cls._cache[key] = result
        return result
```

## 5. 依赖精简

### 5.1 可选依赖分离

```toml
# pyproject.toml
[project]
dependencies = [
    # 核心依赖 (~20MB)
    "google-genai>=1.0.0",
    "typer>=0.9.0",
    "rich>=14.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
browser = ["playwright>=1.40.0"]  # +100MB
search = ["chromadb>=0.4.0"]      # +200MB
database = ["sqlalchemy>=2.0.0"]  # +50MB
all = ["playwright", "chromadb", "sqlalchemy"]
```

### 5.2 按需安装

```bash
# 最小安装
pip install doraemon-code

# 带浏览器
pip install "doraemon-code[browser]"

# 完整安装
pip install "doraemon-code[all]"
```

## 6. 实施优先级

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P0 | AgentState 消息限制 | 内存 -50% |
| P0 | 工具结果缓存 | 速度 +30% |
| P1 | 可选依赖分离 | 安装 -80% |
| P1 | 并行工具调用 | 速度 +50% |
| P2 | 流式文件处理 | 内存 -30% |
| P2 | 懒加载 MCP | 启动 -50% |

## 7. 监控指标

```python
# 添加性能监控
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss
    
    def report(self):
        return {
            "uptime": time.time() - self.start_time,
            "memory_mb": (psutil.Process().memory_info().rss - self.memory_start) / 1024 / 1024,
            "tool_calls": len(self.tool_history),
            "cache_hits": self.cache_hits,
        }
```
