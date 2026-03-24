# Doraemon Code Bug 审查报告

**审查日期**: 2026-03-24  
**审查范围**: 核心模块、服务器模块、工具执行模块

---

## 目录

1. [严重问题 (Critical)](#1-严重问题-critical)
2. [高优先级问题 (High)](#2-高优先级问题-high)
3. [中优先级问题 (Medium)](#3-中优先级问题-medium)
4. [低优先级问题 (Low)](#4-低优先级问题-low)
5. [潜在改进建议](#5-潜在改进建议)

---

## 1. 严重问题 (Critical)

### BUG-001: 临时文件泄漏导致磁盘空间耗尽

**文件**: `src/servers/shell.py:118-138`  
**严重程度**: 🔴 Critical

**问题描述**:
`execute_command_background()` 中创建的临时日志文件在进程启动失败时不会被清理，且进程正常结束后也依赖 `cleanup_finished_processes()` 来清理，但该函数只在 `list_background_processes()` 中被调用。

```python
# 问题代码
stdout_file = tempfile.NamedTemporaryFile(
    mode='w', prefix='doraemon_bg_', suffix='.log', delete=False
)
log_file_path = stdout_file.name
proc = subprocess.Popen(...)  # 如果这里抛出异常，文件泄漏

# 进程结束后，文件只在 cleanup_finished_processes() 中删除
# 但该函数很少被调用
```

**影响**:
- 长期运行的服务器会积累大量临时文件
- 磁盘空间可能耗尽

**修复建议**:
```python
@mcp.tool()
def execute_command_background(...) -> str:
    stdout_file = None
    try:
        stdout_file = tempfile.NamedTemporaryFile(
            mode='w', prefix='doraemon_bg_', suffix='.log', delete=False
        )
        log_file_path = stdout_file.name
        proc = subprocess.Popen(...)
        # 成功后才注册，失败则删除
        pid = _register_background_process(proc, command, resolved_dir, log_file=log_file_path)
        return f"Started background process with PID: {pid}"
    except Exception as e:
        # 清理临时文件
        if stdout_file and os.path.exists(stdout_file.name):
            os.unlink(stdout_file.name)
        return f"Error starting background process: {str(e)}"
```

---

### BUG-002: MCP 客户端进程资源泄漏

**文件**: `src/core/mcp_client.py:170-195`  
**严重程度**: 🔴 Critical

**问题描述**:
`MCPConnection._read_responses()` 中使用 `asyncio.to_thread()` 包装同步 IO 操作，但如果进程在读取期间终止，可能导致线程泄漏或无限等待。

```python
# 问题代码
async def _read_responses(self):
    while True:
        readline = self._process.stdout.readline
        if inspect.iscoroutinefunction(readline):
            line = await readline()
        else:
            line = await asyncio.to_thread(readline)  # 进程终止后可能无限等待
            if inspect.isawaitable(line):
                line = await line
```

**影响**:
- 连接断开后可能留下僵尸进程
- 线程池资源泄漏

**修复建议**:
```python
async def _read_responses(self):
    loop = asyncio.get_event_loop()
    while self._connected and self._process:
        try:
            # 使用超时防止无限等待
            line = await asyncio.wait_for(
                loop.run_in_executor(None, self._process.stdout.readline),
                timeout=5.0
            )
            if not line:
                break
            # ... 处理消息
        except asyncio.TimeoutError:
            if not self._connected or self._process.poll() is not None:
                break
            continue
        except Exception as e:
            logger.error(f"Error reading from MCP server: {e}")
            break
```

---

### BUG-003: 重试逻辑可能抛出 RuntimeError

**文件**: `src/core/model_client_direct.py:65-82`  
**严重程度**: 🔴 Critical

**问题描述**:
`_retry_async()` 函数在 `last_exc` 为 None 时抛出 `RuntimeError`，但这在逻辑上不应该发生，且该异常不会被上层正确处理。

```python
# 问题代码
async def _retry_async(coro_fn, *args, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if not _is_retryable(e) or attempt >= MAX_RETRIES - 1:
                raise
            last_exc = e
            # ...
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_retry_async failed: no attempts were made")  # 逻辑上不可达
```

**影响**:
- 如果代码逻辑有误，会导致难以调试的错误

**修复建议**:
```python
async def _retry_async(coro_fn, *args, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if not _is_retryable(e):
                raise
            last_exc = e
            if attempt >= MAX_RETRIES - 1:
                raise
            # ...
    # 明确处理未重试的情况
    raise RuntimeError("_retry_async: no attempts were made (MAX_RETRIES=0?)")
```

---

### BUG-004: 数据库连接未显式关闭

**文件**: `src/servers/database.py:28-34`  
**严重程度**: 🔴 Critical

**问题描述**:
`_get_connection()` 创建的 SQLite 连接从未被显式关闭，依赖垃圾回收器。

```python
# 问题代码
def _get_connection(db_path: str) -> sqlite3.Connection:
    valid_path = validate_path(db_path)
    conn = sqlite3.connect(valid_path)
    conn.row_factory = sqlite3.Row
    return conn  # 连接从未关闭
```

**影响**:
- 大量查询会积累数据库连接
- 可能导致 "database is locked" 错误
- 资源泄漏

**修复建议**:
```python
@mcp.tool()
def db_read_query(query: str, db_path: str, params: list | None = None) -> str:
    conn = None
    try:
        conn = _get_connection(db_path)
        cursor = conn.execute(query, params or [])
        rows = cursor.fetchall()
        return json.dumps([dict(row) for row in rows], indent=2)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()
```

---

## 2. 高优先级问题 (High)

### BUG-005: Browser 全局资源未正确清理

**文件**: `src/servers/browser.py:15-35`  
**严重程度**: 🟠 High

**问题描述**:
`_browser` 和 `_playwright` 全局变量在异常情况下未被清理，且模块没有提供注册清理钩子的机制。

```python
# 问题代码
_browser: Browser | None = None
_playwright = None

async def get_browser():
    async with _browser_lock:
        if _browser is None:
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(headless=True)
        return _browser  # 异常时 _browser 可能部分初始化
```

**影响**:
- 浏览器进程可能成为孤儿进程
- 内存泄漏

**修复建议**:
```python
import atexit

_browser: Browser | None = None
_playwright = None
_browser_lock = asyncio.Lock()
_initialized = False

async def get_browser():
    global _browser, _playwright, _initialized
    async with _browser_lock:
        if _initialized and _browser is not None:
            return _browser
        try:
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(headless=True)
            _initialized = True
            return _browser
        except Exception:
            # 清理部分初始化的状态
            if _browser:
                try:
                    await _browser.close()
                except:
                    pass
            if _playwright:
                try:
                    await _playwright.stop()
                except:
                    pass
            _browser = None
            _playwright = None
            _initialized = False
            raise

def sync_close_browser():
    """同步清理钩子，用于 atexit"""
    if _browser or _playwright:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(close_browser())
            else:
                loop.run_until_complete(close_browser())
        except:
            pass

atexit.register(sync_close_browser)
```

---

### BUG-006: CircuitBreaker 混合同步/异步锁

**文件**: `src/core/errors.py:243-280`  
**严重程度**: 🟠 High

**问题描述**:
`CircuitBreaker` 使用 `threading.Lock()` 保护状态，但同时支持同步和异步调用，这在异步上下文中可能导致死锁。

```python
# 问题代码
class CircuitBreaker:
    def __init__(self, ...):
        self._lock = threading.Lock()  # 同步锁

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self._lock:  # 在异步函数中使用同步锁
            # ... 可能阻塞事件循环
```

**影响**:
- 在高并发异步场景下可能死锁
- 阻塞事件循环导致性能下降

**修复建议**:
```python
import asyncio

class CircuitBreaker:
    def __init__(self, ...):
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self._sync_lock:
            # 同步版本使用同步锁
            ...

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        async with self._async_lock:  # 异步版本使用异步锁
            ...
```

---

### BUG-007: Shell 多线程输出读取存在竞态条件

**文件**: `src/servers/shell.py:73-110`  
**严重程度**: 🟠 High

**问题描述**:
使用 Queue 在主线程和 reader 线程间传递输出，但存在竞态条件：
1. 进程结束后，reader 线程可能仍在处理最后的输出
2. `output_queue.empty()` 检查和 `get_nowait()` 之间存在竞态

```python
# 问题代码
while True:
    try:
        line = output_queue.get(timeout=idle_poll_interval)
        output_lines.append(line)
        last_activity_time = time.time()
        while True:
            try:
                line = output_queue.get_nowait()  # 竞态: 可能在检查后、获取前有新数据
                output_lines.append(line)
            except queue.Empty:
                break
    except queue.Empty:
        pass

    return_code = process.poll()
    if return_code is not None and not t.is_alive() and output_queue.empty():
        break  # 可能在检查后收到新数据
```

**影响**:
- 可能丢失进程最后输出的内容
- 输出顺序可能错乱

**修复建议**:
```python
import queue

while True:
    try:
        line = output_queue.get(timeout=idle_poll_interval)
        output_lines.append(line)
        last_activity_time = time.time()
        # 一次性排空队列
        while True:
            try:
                line = output_queue.get_nowait()
                output_lines.append(line)
            except queue.Empty:
                break
    except queue.Empty:
        pass

    return_code = process.poll()
    if return_code is not None:
        # 确保读取所有剩余输出
        t.join(timeout=1.0)
        while True:
            try:
                line = output_queue.get_nowait()
                output_lines.append(line)
            except queue.Empty:
                break
        break
```

---

### BUG-008: Session 索引写入非原子操作

**文件**: `src/core/session.py:111-116`  
**严重程度**: 🟠 High

**问题描述**:
`_save_index()` 将索引写入文件，但不是原子操作，如果程序在写入过程中崩溃，会导致索引文件损坏。

```python
# 问题代码
def _save_index(self):
    index_path = self.base_dir / "index.json"
    data = {
        "version": 1,
        "sessions": [m.to_dict() for m in self._index.values()],
    }
    index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # 如果写入过程中崩溃，文件会损坏
```

**影响**:
- 程序崩溃可能导致会话索引丢失
- 需要手动恢复

**修复建议**:
```python
import tempfile

def _save_index(self):
    index_path = self.base_dir / "index.json"
    data = {
        "version": 1,
        "sessions": [m.to_dict() for m in self._index.values()],
    }
    # 原子写入: 先写临时文件，再重命名
    temp_path = index_path.with_suffix('.tmp')
    try:
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(index_path)  # 原子操作
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise
```

---

### BUG-009: Gateway 客户端冗余检查

**文件**: `src/core/model_client_gateway.py:92-97, 203-206`  
**严重程度**: 🟠 High

**问题描述**:
`_make_api_call()` 和 `chat_stream()` 中存在冗余的 None 检查，第二次检查永远不会为 True。

```python
# 问题代码 (第92-97行)
if self._client is None:
    await self.connect()
if self._client is None:  # 冗余: connect() 要么成功要么抛出异常
    raise ConfigurationError("Failed to initialize HTTP client")
```

**影响**:
- 代码冗余，降低可读性
- 可能让开发者误以为 `connect()` 可能静默失败

**修复建议**:
```python
if self._client is None:
    await self.connect()
# connect() 成功则 _client 已设置，失败则抛出异常
```

---

## 3. 中优先级问题 (Medium)

### BUG-010: 工具步骤限制过低

**文件**: `src/host/cli/tool_processor.py:33`  
**严重程度**: 🟡 Medium

**问题描述**:
`MAX_TOOL_STEPS = 15` 可能对于复杂的代理任务过低。

```python
MAX_TOOL_STEPS = 15  # 对于复杂任务可能不够
```

**影响**:
- 复杂的多步骤任务可能被截断
- 用户体验下降

**修复建议**:
```python
# 从配置读取，提供默认值
MAX_TOOL_STEPS = int(os.getenv("AGENT_MAX_TOOL_STEPS", "50"))
```

---

### BUG-011: 多行编辑回滚可能写入损坏内容

**文件**: `src/servers/filesystem_unified.py:475-510`  
**严重程度**: 🟡 Medium

**问题描述**:
`multi_edit()` 在编辑失败时回滚到原始内容，但如果原始内容在内存中已损坏（编码问题等），会写入损坏的内容。

```python
# 问题代码
def multi_edit(path: str, edits: list[dict]) -> str:
    with open(valid_path, encoding="utf-8") as f:
        content = f.read()

    original_content = content  # 如果 read() 遇到编码错误，original_content 可能不完整

    for i, edit in enumerate(edits):
        if old_string not in content:
            # 回滚
            with open(valid_path, "w", encoding="utf-8") as f:
                f.write(original_content)  # 可能写入不完整的内容
```

**影响**:
- 文件可能被截断或损坏

**修复建议**:
```python
def multi_edit(path: str, edits: list[dict]) -> str:
    try:
        with open(valid_path, encoding="utf-8") as f:
            original_content = f.read()
    except UnicodeDecodeError as e:
        return f"Error: Cannot read file as UTF-8: {e}"

    # 备份原文件
    backup_path = valid_path.with_suffix(valid_path.suffix + '.bak')
    try:
        shutil.copy2(valid_path, backup_path)
        
        # 执行编辑...
        
    except Exception as e:
        # 恢复备份
        shutil.copy2(backup_path, valid_path)
        raise
    finally:
        if backup_path.exists():
            backup_path.unlink()
```

---

### BUG-012: 环境变量过滤不完整

**文件**: `src/core/subprocess_utils.py:7-17`  
**严重程度**: 🟡 Medium

**问题描述**:
危险环境变量列表不完整，缺少一些可能导致安全风险的变量。

```python
# 问题代码
_DANGEROUS_ENV_VARS = frozenset({
    "PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    "PYTHONPATH", "NODE_OPTIONS", "BASH_ENV", "ENV", "CDPATH",
    "PERL5OPT", "RUBYOPT", "JAVA_TOOL_OPTIONS",
})
# 缺少: PYTHONHOME, PYTHONINSPECT, NODE_PATH, npm_config_*, etc.
```

**影响**:
- 可能被利用绕过安全限制

**修复建议**:
```python
_DANGEROUS_ENV_VARS = frozenset({
    # 动态链接器
    "PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    "DYLD_FORCE_FLAT_NAMESPACE", "DYLD_LIBRARY_PATH",
    # Python
    "PYTHONPATH", "PYTHONHOME", "PYTHONINSPECT", "PYTHONSTARTUP",
    # Node.js
    "NODE_OPTIONS", "NODE_PATH",
    # Shell
    "BASH_ENV", "ENV", "CDPATH", "PS1", "HISTFILE",
    # 其他运行时
    "PERL5OPT", "RUBYOPT", "JAVA_TOOL_OPTIONS", "JAVA_OPTIONS",
    # npm/yarn
    "npm_config_script", "npm_config_prefix",
})
```

---

### BUG-013: 工具循环检测可能误报

**文件**: `src/host/cli/tool_execution.py:83-116`  
**严重程度**: 🟡 Medium

**问题描述**:
`detect_tool_loop()` 使用简单的签名匹配，可能对合理的重复操作产生误报。

```python
# 问题代码
if len(previous_tool_calls) >= 3:
    last_three = previous_tool_calls[-3:]
    if all(s == current_call_signature for s in last_three):
        return True, f"Loop detected: {tool_name} called repeatedly with same args."
```

**影响**:
- 某些合法的重复操作（如批量处理）会被误判为循环

**修复建议**:
```python
# 添加白名单或智能检测
LOOP_WHITELIST = {"read", "search", "glob"}  # 读操作可以重复

def detect_tool_loop(tool_name, args, previous_tool_calls):
    if tool_name in LOOP_WHITELIST:
        return False, ""

    # ... 原有逻辑
```

---

### BUG-014: 空响应处理边界情况

**文件**: `src/host/cli/tool_processor.py:341-345`  
**严重程度**: 🟡 Medium

**问题描述**:
空响应检查逻辑可能遗漏某些边界情况。

```python
# 问题代码
if not response.content and not response.tool_calls:
    if tool_steps == 1:
        console.print("[red]Empty response[/red]")
    break
# 如果 response.content 是空字符串 "" 且 response.tool_calls 是 []，
# 条件为 True，但实际上可能是正常响应
```

**影响**:
- 可能错误地中断正常流程

**修复建议**:
```python
# 更精确的检查
has_content = response.content and response.content.strip()
has_tool_calls = response.tool_calls and len(response.tool_calls) > 0

if not has_content and not has_tool_calls:
    if tool_steps == 1:
        console.print("[red]Empty response[/red]")
    break
```

---

### BUG-015: Provider 检测可能匹配错误

**文件**: `src/core/model_client_direct.py:155-175`  
**严重程度**: 🟡 Medium

**问题描述**:
`_detect_provider()` 使用简单的模型名前缀匹配，可能导致错误匹配。

```python
# 问题代码
PROVIDER_PATTERNS = {
    Provider.GOOGLE: ["gemini-", "palm-"],
    Provider.OPENAI: ["gpt-", "o1", "o3"],  # "o1" 太短，可能误匹配
    Provider.ANTHROPIC: ["claude-"],
}

# 例如: "o1-test-model" 可能被误识别
```

**影响**:
- 使用错误的 provider 发送请求导致失败

**修复建议**:
```python
PROVIDER_PATTERNS = {
    Provider.GOOGLE: ["gemini-", "palm-"],
    Provider.OPENAI: ["gpt-", "o1-", "o3-"],  # 添加连字符
    Provider.ANTHROPIC: ["claude-"],
}
```

---

## 4. 低优先级问题 (Low)

### BUG-016: 日志级别不一致

**文件**: 多个文件  
**严重程度**: 🟢 Low

**问题描述**:
日志级别使用不一致，某些重要的错误使用 `logger.warning()` 而非 `logger.error()`。

**示例**:
- `src/core/config.py:55`: 配置加载失败使用 `logger.warning()`
- `src/servers/shell.py:48`: 危险命令阻止使用 `logger.warning()`

**修复建议**:
统一日志级别规范，确保:
- 可恢复的错误使用 `warning`
- 不可恢复的错误使用 `error`
- 预期的异常情况使用 `info`

---

### BUG-017: 类型注解不完整

**文件**: 多个文件  
**严重程度**: 🟢 Low

**问题描述**:
部分函数缺少类型注解或使用 `Any` 过多。

**示例**:
```python
# src/host/cli/tool_processor.py
class ToolCallContext:
    registry: Any  # 应该是 ToolRegistry
    checkpoint_mgr: Any  # 应该是 CheckpointManager
    hook_mgr: HookManager  # 这个有类型
```

**修复建议**:
添加完整的类型注解，提高代码可维护性。

---

### BUG-018: 魔法数字未定义为常量

**文件**: 多个文件  
**严重程度**: 🟢 Low

**问题描述**:
代码中存在多处魔法数字。

**示例**:
- `src/host/cli/chat_loop.py:26`: `MAX_INLINE_FILE_CHARS = 50_000`
- `src/servers/browser.py:52`: `text[:10000]` 硬编码

**修复建议**:
将所有魔法数字定义为命名常量，并从配置读取。

---

## 5. 潜在改进建议

### 改进-001: 统一资源管理

建议使用统一的资源管理模式，例如：

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_browser():
    """统一管理浏览器资源"""
    browser = None
    playwright = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        yield browser
    finally:
        if browser:
            await browser.close()
        if playwright:
            await playwright.stop()
```

### 改进-002: 添加资源监控

建议添加资源使用监控，包括：
- 打开的文件描述符数量
- 活跃的网络连接
- 后台进程数量
- 内存使用情况

### 改进-003: 统一错误处理

建议创建统一的错误处理装饰器：

```python
def safe_tool_execution(func):
    """统一处理工具执行中的异常"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except PermissionError as e:
            return f"Permission denied: {e}"
        except FileNotFoundError as e:
            return f"File not found: {e}"
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return f"Error: {str(e)}"
    return wrapper
```

### 改进-004: 添加健康检查端点

建议添加系统健康检查机制：

```python
def health_check() -> dict:
    """检查系统健康状态"""
    return {
        "browser": _browser is not None,
        "active_processes": len(_background_processes),
        "open_connections": len(_mcp_connections),
        "disk_space": shutil.disk_usage("/").free,
    }
```

---

## 总结

| 严重程度 | 数量 | 主要类别 |
|---------|------|---------|
| Critical | 4 | 资源泄漏 |
| High | 5 | 并发问题、错误处理 |
| Medium | 5 | 边界条件、安全性 |
| Low | 3 | 代码质量 |

**优先修复建议**:
1. BUG-001: 临时文件泄漏
2. BUG-002: MCP 进程资源泄漏
3. BUG-004: 数据库连接未关闭
4. BUG-006: CircuitBreaker 死锁风险
5. BUG-008: Session 索引非原子写入
