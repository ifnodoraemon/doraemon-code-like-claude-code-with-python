# Doraemon Code 架构审查报告

## 1. 当前架构问题

### 1.1 没有统一的 Agent 抽象

**当前状态**：`main.py` (1209 行) 包含所有逻辑

```
当前流程（自定义流程）:
┌─────────────────────────────────────────────────────────────┐
│ chat_loop()                                                 │
│   ├── initialize_chat_runtime() - 初始化 10+ 个 Manager    │
│   ├── dispatch_user_input()                                │
│   │     ├── handle_bash_mode()                             │
│   │     ├── CommandHandler.handle()                        │
│   │     └── execute_agent_turn()                           │
│   │           ├── process_user_input()                     │
│   │           ├── stream_model_response()                  │
│   │           ├── process_tool_calls()                     │
│   │           │     ├── HITL approval                      │
│   │           │     ├── execute tools                      │
│   │           │     └── retry on errors                    │
│   │           └── handle context overflow                  │
│   └── persist_session_state()                              │
└─────────────────────────────────────────────────────────────┘
```

**问题**：
- ❌ 没有标准的 Agent 类
- ❌ ReAct 循环隐藏在 `execute_agent_turn()` 中
- ❌ 工具调用逻辑分散在多个文件
- ❌ 难以测试和扩展

### 1.2 Manager 爆炸

```python
# initialization.py 中的初始化
managers = {
    "model_client": model_client,       # 调用 LLM
    "registry": registry,               # 工具注册
    "tool_selector": tool_selector,     # 工具选择
    "ctx": ctx,                         # 上下文管理
    "skills": skills,             # Skills
    "checkpoints": checkpoints,   # 检查点
    "task_mgr": task_mgr,               # 后台任务
    "hooks": hooks,               # Hooks
    "cost_tracker": cost_tracker,       # 成本追踪
    "cmd_history": cmd_history,         # 命令历史
    "bash_executor": bash_executor,     # Bash 执行
    "session_mgr": session_mgr,         # 会话管理
    "permission_mgr": permission_mgr,   # 权限管理
}
```

**问题**：
- ❌ 13 个独立 Manager，职责不清
- ❌ 传递 13 个参数给每个函数
- ❌ 状态散落各处

### 1.3 对比标准 Agentic 架构

**标准 ReAct Agent** (LangChain/LlamaIndex 模式):

```python
class Agent:
    def __init__(self, tools, llm, memory):
        self.tools = tools
        self.llm = llm
        self.memory = memory
    
    async def run(self, input: str) -> str:
        while not self.is_finished:
            # 1. Observe
            observation = self.observe()
            
            # 2. Think
            thought = await self.think(observation)
            
            # 3. Act
            action = self.decide_action(thought)
            
            if action.is_tool_call:
                result = await self.execute_tool(action)
                self.memory.add(result)
            else:
                return action.response
```

**我们当前的代码**：
- ❌ 没有 `Agent` 类
- ❌ ReAct 循环埋在 1200 行代码里
- ❌ 无法独立测试 Agent 逻辑

---

## 2. 标准 Agentic 架构

### 2.1 核心 Agent 类

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class Observation:
    """Agent 观察到的内容"""
    user_input: str
    tool_results: list[dict[str, Any]] | None = None
    errors: list[str] | None = None

@dataclass  
class Thought:
    """Agent 的思考"""
    reasoning: str
    tool_calls: list[dict] | None = None
    response: str | None = None
    is_finished: bool = False

@dataclass
class Action:
    """Agent 决定的动作"""
    type: str  # "tool_call" | "respond" | "error"
    tool_name: str | None = None
    tool_args: dict | None = None
    response: str | None = None

class BaseAgent(ABC):
    """标准 Agent 抽象"""
    
    @abstractmethod
    async def observe(self) -> Observation:
        """观察环境状态"""
        pass
    
    @abstractmethod
    async def think(self, observation: Observation) -> Thought:
        """推理下一步"""
        pass
    
    @abstractmethod
    async def act(self, thought: Thought) -> Action:
        """执行动作"""
        pass
    
    async def run(self, input: str) -> str:
        """主循环 - ReAct"""
        while True:
            observation = await self.observe()
            thought = await self.think(observation)
            action = await self.act(thought)
            
            if action.type == "respond":
                return action.response
            elif action.type == "tool_call":
                result = await self.execute_tool(action)
                self.memory.add(result)
            elif action.type == "error":
                return f"Error: {action.response}"
```

### 2.2 简化后的架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent (统一入口)                        │
├─────────────────────────────────────────────────────────────┤
│  observe() ──► think() ──► act() ──► [loop or return]      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   LLM    │  │  Tools   │  │  Memory  │  │  Hooks   │    │
│  │ Client   │  │ Registry │  │ Manager  │  │ Manager  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 文件结构对比

**当前**:
```
src/host/cli/
├── main.py        # 1209 行，所有逻辑
├── main.py             # 396 行，CLI 入口
├── commands.py         # 命令处理
├── commands_core.py    # 更多命令
├── tool_processor.py   # 工具处理
├── tool_execution.py   # 工具执行
├── initialization.py   # 初始化
└── ...
```

**应该**:
```
src/agent/
├── __init__.py
├── base.py             # BaseAgent 抽象类
├── react.py            # ReActAgent 实现
├── state.py            # AgentState (替代多个 Manager)
├── tools.py            # ToolExecutor (统一工具执行)
├── memory.py           # AgentMemory (上下文 + 记忆)
└── hooks.py            # AgentHooks (生命周期钩子)
```

---

## 3. 改进计划

### 3.1 Phase 1: 创建 Agent 抽象 (不破坏现有代码)

```python
# src/agent/base.py
class BaseAgent(ABC):
    """标准 Agent 接口，对标 LangChain/LlamaIndex"""
    
    async def run(self, input: str, **kwargs) -> AgentResult:
        """主入口"""
        raise NotImplementedError

# src/agent/react.py  
class ReActAgent(BaseAgent):
    """ReAct 模式 Agent"""
    
    async def run(self, input: str, **kwargs) -> AgentResult:
        self.state.set_input(input)
        
        while not self.state.is_finished:
            observation = await self.observe()
            thought = await self.think(observation)
            action = await self.act(thought)
            
            if action.is_final:
                break
        
        return AgentResult(
            response=self.state.response,
            tool_calls=self.state.tool_history,
        )
```

### 3.2 Phase 2: 统一状态管理

```python
# src/agent/state.py
@dataclass
class AgentState:
    """替代 13 个 Manager"""
    
    # 核心状态
    messages: list[Message]
    tool_results: list[ToolResult]
    
    # 配置
    mode: str = "build"  # plan/build
    max_turns: int = 100
    
    # 历史记录
    tool_history: list[ToolCall] = field(default_factory=list)
    
    # 不再需要单独的 Manager
    # - ContextManager -> messages
    # - CheckpointManager -> 在 save/restore 方法中
    # - CostTracker -> usage 字段
    # - SessionManager -> 在 Agent 层面处理
```

### 3.3 Phase 3: 简化 chat_loop

```python
# src/host/cli/main.py (简化后)
async def chat_loop(project: str, **kwargs):
    agent = ReActAgent(
        llm=await ModelClient.create(),
        tools=ToolRegistry(),
        memory=AgentMemory(project),
    )
    
    while True:
        user_input = await read_input()
        if user_input == "exit":
            break
        
        result = await agent.run(user_input)
        display(result)
```

---

## 4. 测试对比

### 4.1 当前测试方式

```python
# 需要初始化 13 个 Manager
async def test_chat():
    model_client = await initialize_model_client()
    registry = initialize_registry()
    ctx = initialize_context_manager("test")
    skills = initialize_skill_manager()
    # ... 初始化 10+ 个组件
    
    # 然后才能测试
    result = await execute_agent_turn(...)
```

### 4.2 标准化后测试

```python
# 只需要 Mock Agent 依赖
async def test_agent():
    agent = ReActAgent(
        llm=MockLLM(),
        tools=MockTools(),
        memory=InMemoryMemory(),
    )
    
    result = await agent.run("Hello")
    assert result.response == "Hi there!"
```

---

## 5. 立即行动

### 优先级排序

| 优先级 | 任务 | 复杂度 |
|--------|------|--------|
| P0 | 创建 `BaseAgent` 抽象类 | 低 |
| P0 | 创建 `ReActAgent` 实现 | 中 |
| P1 | 创建 `AgentState` 统一状态 | 中 |
| P1 | 重构 `ToolExecutor` | 中 |
| P2 | 迁移 chat_loop 到 Agent | 高 |
| P2 | 删除冗余 Manager | 高 |

### 第一步：创建 Agent 抽象

```python
# src/agent/__init__.py
from .base import BaseAgent, AgentResult
from .react import ReActAgent

__all__ = ["BaseAgent", "ReActAgent", "AgentResult"]
```

这样可以：
1. ✅ 不破坏现有代码
2. ✅ 提供标准化接口
3. ✅ 支持测试
4. ✅ 为未来迁移做准备
