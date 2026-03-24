# Doraemon Code 测试与 Benchmark 计划

## 1. 测试金字塔

```
                    ┌─────────────┐
                    │   E2E 测试   │  (用户场景)
                    │   5-10 个    │
                ┌───┴─────────────┴───┐
                │    集成测试          │  (组件交互)
                │    20-30 个          │
            ┌───┴─────────────────────┴───┐
            │        单元测试              │  (函数级别)
            │        100+ 个               │
            └─────────────────────────────┘
```

## 2. 新功能测试矩阵

### 2.1 MCP 集成测试

| 测试 ID | 测试场景 | 预期结果 | 优先级 |
|---------|----------|----------|--------|
| MCP-001 | 加载配置文件中的 mcpServers | 所有服务器配置正确加载 | P0 |
| MCP-002 | 连接 stdio MCP 服务器 | 成功建立连接，列出工具 | P0 |
| MCP-003 | 连接 HTTP MCP 服务器 | 成功建立连接，支持 OAuth | P1 |
| MCP-004 | 连接 SSE MCP 服务器 | 成功建立连接，支持流式 | P1 |
| MCP-005 | 工具发现和注册 | MCP 工具与内置工具合并 | P0 |
| MCP-006 | 工具调用路由 | 正确路由到对应 MCP 服务器 | P0 |
| MCP-007 | 错误处理 - 服务器无响应 | 优雅降级，不影响其他服务器 | P1 |
| MCP-008 | 环境变量展开 | ${VAR} 和 ${VAR:-default} 正确解析 | P1 |
| MCP-009 | 动态加载/卸载服务器 | 热加载新服务器配置 | P2 |
| MCP-010 | 资源访问 | 通过 @server:uri 访问资源 | P2 |

### 2.2 Commands/Workflow 测试

| 测试 ID | 测试场景 | 预期结果 | 优先级 |
|---------|----------|----------|--------|
| CMD-001 | 解析 .md 命令文件 | 正确提取 RUN/READ 指令 | P0 |
| CMD-002 | 执行 RUN 命令 | 命令输出正确捕获 | P0 |
| CMD-003 | 执行 READ 文件 | 文件内容正确加载 | P0 |
| CMD-004 | 参数替换 $ARG | 参数正确替换 | P0 |
| CMD-005 | 命令链式执行 | 多个命令按顺序执行 | P1 |
| CMD-006 | 条件执行 | 根据条件跳过/执行命令 | P2 |
| CMD-007 | 命令错误处理 | 单个命令失败不影响整体 | P1 |
| CMD-008 | 命令缓存 | 相同命令结果缓存 | P2 |

### 2.3 Skills 测试

| 测试 ID | 测试场景 | 预期结果 | 优先级 |
|---------|----------|----------|--------|
| SKL-001 | 加载 SKILL.md | 正确解析 frontmatter 和内容 | P0 |
| SKL-002 | 触发词匹配 | 根据上下文自动加载相关 skill | P0 |
| SKL-003 | 优先级排序 | 高优先级 skill 优先加载 | P1 |
| SKL-004 | 依赖加载 | requires 字段正确处理 | P1 |
| SKL-005 | 附加文件加载 | files 字段指定的文件正确加载 | P1 |
| SKL-006 | Token 限制 | 总内容不超过 max_skill_tokens | P0 |
| SKL-007 | 动态上下文注入 | !`command` 语法正确执行 | P2 |
| SKL-008 | 项目级 vs 用户级 | 正确合并多级 skills | P2 |

### 2.4 Hooks 测试 (现有基础上增强)

| 测试 ID | 测试场景 | 预期结果 | 优先级 |
|---------|----------|----------|--------|
| HOK-001 | PreToolUse 阻止 | 返回 DENY 决策，工具不执行 | P0 |
| HOK-002 | PreToolUse 修改 | 修改 tool_input 后执行 | P0 |
| HOK-003 | PostToolUse 修改 | 修改 tool_output 返回给模型 | P1 |
| HOK-004 | 并行 Hook 执行 | 多个 Hook 并行执行正确 | P1 |
| HOK-005 | Hook 超时处理 | 超时后优雅处理 | P1 |
| HOK-006 | SessionStart/End | 生命周期 Hook 正确触发 | P1 |
| HOK-007 | HTTP Hook | POST 请求正确发送 | P2 |
| HOK-008 | Prompt Hook | LLM 评估 Hook | P2 |
| HOK-009 | Agent Hook | 子代理验证 Hook | P2 |

### 2.5 Agentic 能力测试

| 测试 ID | 测试场景 | 预期结果 | 优先级 |
|---------|----------|----------|--------|
| AGT-001 | 多步骤任务 | 自动拆分并执行 | P0 |
| AGT-002 | 错误恢复 | 单步失败后重试/跳过 | P1 |
| AGT-003 | 上下文压缩 | 超过阈值自动压缩 | P1 |
| AGT-004 | 工具选择优化 | 选择最少工具完成任务 | P1 |
| AGT-005 | 并行工具调用 | 独立工具并行执行 | P1 |
| AGT-006 | 任务规划 | Plan 模式生成计划 | P0 |
| AGT-007 | 计划执行 | Build 模式执行计划 | P0 |

## 3. Benchmark 指标

### 3.1 性能基准

```yaml
latency:
  simple_query_p50: 1000ms    # 简单查询 P50 延迟
  simple_query_p99: 3000ms    # 简单查询 P99 延迟
  tool_call_p50: 500ms        # 工具调用 P50 延迟
  tool_call_p99: 2000ms       # 工具调用 P99 延迟
  first_token: 200ms          # 首 token 延迟

throughput:
  tokens_per_second: 50       # 每秒生成 token 数
  tool_calls_per_minute: 30   # 每分钟工具调用数
  concurrent_sessions: 5      # 并发会话数

resource:
  memory_baseline_mb: 300     # 基础内存占用
  memory_per_session_mb: 50   # 每会话内存增量
  cpu_idle_percent: 5         # 空闲 CPU 占用
```

### 3.2 质量基准 (对标 Claude Code)

```yaml
accuracy:
  file_edit_success_rate: 0.95    # 文件编辑成功率
  code_generation_correct: 0.85   # 代码生成正确率
  tool_selection_accuracy: 0.90   # 工具选择准确率
  task_completion_rate: 0.80      # 任务完成率

efficiency:
  avg_tool_calls_per_task: 5      # 每任务平均工具调用
  avg_tokens_per_task: 2000       # 每任务平均 token
  context_utilization: 0.7        # 上下文利用率
  error_recovery_rate: 0.8        # 错误恢复率
```

### 3.3 MCP 特定基准

```yaml
mcp_connection_time_ms: 500       # MCP 连接时间
mcp_tool_discovery_ms: 200        # 工具发现时间
mcp_overhead_percent: 5           # MCP 额外开销
mcp_server_uptime: 0.99           # MCP 服务器可用性
```

## 4. 测试数据集

### 4.1 标准任务集

```json
{
  "tasks": [
    {
      "id": "simple_file_read",
      "name": "读取文件",
      "category": "filesystem",
      "difficulty": "easy",
      "prompt": "读取 README.md 文件的前 50 行",
      "expected_tools": ["read"],
      "success_criteria": ["返回正确内容", "行数不超过 50"]
    },
    {
      "id": "create_python_file",
      "name": "创建 Python 文件",
      "category": "coding",
      "difficulty": "easy",
      "prompt": "创建一个 hello.py 文件，包含一个 main 函数打印 Hello World",
      "expected_tools": ["write"],
      "success_criteria": ["文件创建成功", "包含 main 函数", "代码可运行"]
    },
    {
      "id": "fix_bug",
      "name": "修复 Bug",
      "category": "coding",
      "difficulty": "medium",
      "prompt": "修复 src/example.py 中第 42 行的除零错误",
      "expected_tools": ["read", "write"],
      "success_criteria": ["错误已修复", "不引入新问题"]
    },
    {
      "id": "refactor_function",
      "name": "重构函数",
      "category": "coding",
      "difficulty": "medium",
      "prompt": "将 src/utils.py 中的 process_data 函数拆分为多个小函数",
      "expected_tools": ["read", "search", "write"],
      "success_criteria": ["功能不变", "代码更清晰", "测试通过"]
    },
    {
      "id": "multi_file_edit",
      "name": "多文件编辑",
      "category": "coding",
      "difficulty": "hard",
      "prompt": "重命名 src/old_name.py 为 src/new_name.py，并更新所有引用",
      "expected_tools": ["read", "search", "write"],
      "success_criteria": ["文件重命名成功", "所有引用更新", "无遗漏"]
    },
    {
      "id": "web_scraper",
      "name": "网页抓取",
      "category": "browser",
      "difficulty": "medium",
      "prompt": "访问 example.com 并提取页面标题",
      "expected_tools": ["browse_page"],
      "success_criteria": ["成功访问", "提取正确标题"]
    }
  ]
}
```

### 4.2 边缘案例集

```json
{
  "edge_cases": [
    {
      "id": "empty_file",
      "prompt": "读取空文件",
      "expected": "返回空内容，不报错"
    },
    {
      "id": "large_file",
      "prompt": "读取 10MB 文件",
      "expected": "正确截断或分块处理"
    },
    {
      "id": "binary_file",
      "prompt": "读取二进制文件",
      "expected": "正确识别并处理"
    },
    {
      "id": "invalid_path",
      "prompt": "访问不存在的路径",
      "expected": "返回有意义的错误信息"
    },
    {
      "id": "permission_denied",
      "prompt": "访问无权限文件",
      "expected": "正确处理权限错误"
    },
    {
      "id": "concurrent_write",
      "prompt": "同时写入同一文件",
      "expected": "正确处理冲突"
    }
  ]
}
```

## 5. CI/CD 集成

### 5.1 测试阶段

```yaml
stages:
  - lint          # 代码风格检查
  - unit          # 单元测试
  - integration   # 集成测试
  - benchmark     # 性能基准
  - eval          # 效果评估 (手动触发)
```

### 5.2 测试命令

```bash
# 快速测试 (CI 每次提交)
pytest tests/core tests/servers -v -x --timeout=60

# 完整测试 (PR 合并)
pytest tests/ -v --timeout=300 --cov=src --cov-report=xml

# Benchmark (每日构建)
pytest tests/benchmarks -v --benchmark

# 效果评估 (发布前)
python scripts/run_evals.py --full
```

## 6. 测试文件结构

```
tests/
├── core/
│   ├── test_mcp_integration.py      # MCP 集成测试 (新增)
│   ├── test_commands.py             # Commands 测试 (新增)
│   ├── test_skills.py               # Skills 测试 (已有)
│   └── test_hooks.py                # Hooks 测试 (已有)
├── integration/
│   ├── test_full_workflow.py        # 完整工作流测试
│   └── test_mcp_servers.py          # MCP 服务器集成测试
├── benchmarks/
│   ├── test_performance_benchmarks.py  # 性能基准 (已有)
│   ├── test_mcp_benchmarks.py          # MCP 基准 (新增)
│   └── test_agentic_benchmarks.py      # Agentic 基准 (新增)
├── evals/
│   ├── tasks/
│   │   └── agent_eval_tasks.json    # 评估任务 (已有)
│   └── agent_evaluator.py           # 评估器 (已有)
└── fixtures/
    ├── mock_mcp_servers/            # Mock MCP 服务器
    └── sample_projects/             # 测试项目
```

## 7. 成功标准

### 7.1 测试覆盖率

- 单元测试覆盖率: ≥ 80%
- 集成测试覆盖关键路径: 100%
- 新功能必须有测试

### 7.2 Benchmark 达标

- 所有性能指标在基线范围内
- 无性能回归 (±10% 容忍)
- 质量指标达到竞品 80% 水平

### 7.3 发布门槛

- 所有单元测试通过
- 所有集成测试通过
- 无 P0/P1 级别未解决问题
- Benchmark 无回归
