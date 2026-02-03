# Doraemon Code 评估系统

完整的 Agent 评估框架，支持多维度评估、并行执行、LLM 评判和详细指标收集。

## 目录结构

```
tests/evals/
├── agent_evaluator.py           # 基础评估器
├── baseline_runner.py            # 基线评估运行器
├── llm_judge_evaluator.py       # LLM 评判器
├── multi_model_comparator.py    # 多模型对比器
├── parallel_evaluator.py        # 并行评估器 (NEW)
├── metrics_collector.py         # 指标收集器 (NEW)
├── comprehensive_evaluator.py   # 完整评估系统 (NEW)
├── test_evaluation_system.py    # 评估系统测试 (NEW)
├── test_agent_capabilities.py   # Agent 能力测试
├── run_full_evaluation.sh       # 评估运行脚本
└── tasks/                       # 评估任务集
    ├── basic/                   # 基础任务 (难度 1-3)
    │   └── file_and_code_tasks.json
    ├── intermediate/            # 中级任务 (难度 4-6) (NEW)
    │   └── tasks.json
    ├── advanced/                # 高级任务 (难度 7-9)
    │   └── tasks.json
    ├── expert/                  # 专家任务 (难度 10)
    │   └── tasks.json
    └── agent_eval_tasks.json    # Agent 综合评估任务
```

## 核心功能

### 1. 并行评估 (parallel_evaluator.py)

支持多线程、多进程和异步并行评估，大幅提升评估速度。

```python
from tests.evals.parallel_evaluator import ParallelEvaluator

# 多线程并行 (适合 I/O 密集型)
evaluator = ParallelEvaluator(max_workers=4, use_processes=False)
summary = evaluator.run_parallel_evaluation(agent_factory, task_files, n_trials=3)

# 多进程并行 (适合 CPU 密集型)
evaluator = ParallelEvaluator(max_workers=4, use_processes=True)
summary = evaluator.run_parallel_evaluation(agent_factory, task_files)

# 异步并行 (适合高并发)
evaluator = ParallelEvaluator(max_workers=10)
summary = await evaluator.run_async_evaluation(agent_factory, task_files)
```

**特性:**
- 自动计算加速比
- 支持多次试验 (n_trials)
- 详细的并行执行统计
- 错误隔离和恢复

### 2. 指标收集 (metrics_collector.py)

全面收集和分析评估指标。

```python
from tests.evals.metrics_collector import MetricsCollector

collector = MetricsCollector()
collector.start_collection()

# 运行评估并记录结果
for result in evaluation_results:
    collector.record_task_result(result)

collector.end_collection()

# 打印和保存指标
collector.print_summary()
collector.save_metrics()
```

**收集的指标:**
- **核心指标**: 成功率、执行时间、任务统计
- **工具指标**: 工具调用次数、使用分布、效率
- **难度指标**: 按难度的成功率和耗时
- **类别指标**: 按类别的性能分析
- **LLM 指标**: LLM 评判分数统计
- **性能指标**: P50/P95/P99 延迟
- **错误指标**: 错误类型和频率

### 3. 完整评估系统 (comprehensive_evaluator.py)

整合所有评估组件的统一入口。

```python
from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator

# 创建评估器
evaluator = ComprehensiveEvaluator(
    output_dir="eval_results",
    parallel=True,
    max_workers=4,
    n_trials=3,
    use_llm_judge=True
)

# 运行完整评估
report = evaluator.run_full_evaluation(agent_factory)

# 运行基线评估
baseline = evaluator.run_baseline_evaluation(agent_factory)

# 多模型对比
comparison = evaluator.run_model_comparison({
    "gemini-2.0": create_gemini_agent,
    "gpt-4": create_gpt4_agent,
    "claude-3.5": create_claude_agent
})

# 与基线对比
comparison = evaluator.compare_with_baseline(current_results)
```

**输出:**
- JSON 格式详细报告
- Markdown 格式可读报告
- 指标可视化数据
- 失败案例分析

### 4. LLM 评判 (llm_judge_evaluator.py)

使用 LLM 作为评判者评估复杂任务。

```python
from tests.evals.llm_judge_evaluator import LLMJudgeEvaluator

judge = LLMJudgeEvaluator(model="gemini-2.0-flash-exp")

# 评估单个响应
scores = judge.evaluate_response(task, agent_response, agent_actions)

# 批量评估
evaluated_results = judge.batch_evaluate(results)

# 生成汇总报告
summary = judge.generate_summary_report(evaluated_results)
```

**评估维度:**
- 任务完成度 (1-10)
- 工具使用 (1-10)
- 代码质量 (1-10)
- 问题解决 (1-10)
- 用户体验 (1-10)
- 完整性 (1-10)

### 5. 基线管理 (baseline_runner.py)

建立和维护性能基线。

```python
from tests.evals.baseline_runner import BaselineRunner

runner = BaselineRunner()

# 运行并保存基线
baseline = runner.run_baseline_evaluation(agent, task_files)
runner.print_baseline_report(baseline)

# 加载基线
baseline = runner.load_baseline()

# 对比当前结果
comparison = runner.compare_with_baseline(current_results)
```

### 6. 多模型对比 (multi_model_comparator.py)

对比不同模型的性能。

```python
from tests.evals.multi_model_comparator import MultiModelComparator

comparator = MultiModelComparator()

# 对比多个模型
results = comparator.compare_models(task_files)

# 生成对比报告
report = comparator.generate_comparison_report(results)
comparator.print_comparison(results)
```

## 评估任务集

### 基础任务 (Basic - 难度 1-3)

**文件操作:**
- 创建文件
- 读取文件
- 目录操作

**代码生成:**
- 简单函数
- 基础类定义

**示例:**
```json
{
  "id": "file-001",
  "category": "file_operations",
  "difficulty": "easy",
  "prompt": "Create a Python file named 'hello.py' that prints 'Hello, World!'",
  "expected_tools": ["write_file"],
  "success_criteria": [
    "文件存在",
    "内容正确",
    "语法正确"
  ]
}
```

### 中级任务 (Intermediate - 难度 4-6) [NEW]

**数据结构:**
- Stack, Queue 实现
- 算法实现 (二分查找等)

**代码编辑:**
- Bug 修复
- 代码重构
- 添加测试

**多文件操作:**
- 包结构创建
- 模块组织

**示例:**
```json
{
  "id": "inter-code-001",
  "category": "code_generation",
  "difficulty": 5,
  "prompt": "Implement a Stack class with push, pop, peek, and is_empty methods",
  "expected_tools": ["write_file"],
  "success_criteria": [
    "实现了所有方法",
    "有错误处理",
    "代码质量高"
  ]
}
```

### 高级任务 (Advanced - 难度 7-9)

**架构设计:**
- 微服务架构
- 数据库设计
- API 设计

**全栈开发:**
- REST API 实现
- 数据库集成
- 认证系统

**性能优化:**
- 代码优化
- 异步编程
- 缓存策略

**DevOps:**
- Docker 容器化
- CI/CD 配置
- 部署脚本

### 专家任务 (Expert - 难度 10)

**端到端项目:**
- 完整应用开发
- 前后端集成
- 测试和文档

**复杂算法:**
- 分布式算法
- 共识算法
- 机器学习集成

**系统设计:**
- 分布式系统
- 高可用架构
- 性能优化

## 使用指南

### 快速开始

```python
from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator

# 1. 定义 Agent 工厂函数
def create_agent():
    from src.host.cli.main import DoraemonAgent
    return DoraemonAgent()

# 2. 创建评估器
evaluator = ComprehensiveEvaluator(
    parallel=True,
    max_workers=4,
    n_trials=3
)

# 3. 运行评估
report = evaluator.run_full_evaluation(create_agent)

# 4. 查看结果
print(f"成功率: {report['metrics']['core_metrics']['success_rate']*100:.1f}%")
```

### 运行特定难度的任务

```python
# 只运行基础任务
task_files = ["tests/evals/tasks/basic/file_and_code_tasks.json"]
report = evaluator.run_full_evaluation(create_agent, task_files)

# 只运行高级任务
task_files = ["tests/evals/tasks/advanced/tasks.json"]
report = evaluator.run_full_evaluation(create_agent, task_files)
```

### 建立性能基线

```python
# 首次运行，建立基线
baseline = evaluator.run_baseline_evaluation(create_agent)

# 后续运行，与基线对比
current = evaluator.run_full_evaluation(create_agent)
comparison = evaluator.compare_with_baseline(current['summary'])

# 检查是否有性能退化
if comparison['regressions']:
    print("警告: 检测到性能退化!")
    for regression in comparison['regressions']:
        print(f"  - {regression}")
```

### 多模型对比

```python
def create_gemini_agent():
    return DoraemonAgent(model="gemini-2.0-flash-exp")

def create_gpt4_agent():
    return DoraemonAgent(model="gpt-4-turbo")

def create_claude_agent():
    return DoraemonAgent(model="claude-3-5-sonnet-20241022")

# 对比三个模型
comparison = evaluator.run_model_comparison({
    "Gemini 2.0": create_gemini_agent,
    "GPT-4": create_gpt4_agent,
    "Claude 3.5": create_claude_agent
})

# 查看最佳模型
print("最佳模型:", comparison['best_model'])
```

### 使用 pytest 运行

```bash
# 运行所有评估测试
pytest tests/evals/test_evaluation_system.py -v

# 运行特定测试
pytest tests/evals/test_evaluation_system.py::test_comprehensive_evaluation -v

# 运行慢速测试
pytest tests/evals/test_evaluation_system.py -v -m slow

# 生成覆盖率报告
pytest tests/evals/ --cov=tests/evals --cov-report=html
```

## 评估指标说明

### 核心指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **成功率** | 任务成功完成的百分比 | >90% |
| **平均耗时** | 每个任务的平均执行时间 | <3s |
| **工具调用效率** | 平均工具调用次数 | <10/任务 |

### 质量指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **代码质量** | LLM 评判的代码质量分数 | >8/10 |
| **任务完成度** | LLM 评判的完成度分数 | >8/10 |
| **用户体验** | LLM 评判的 UX 分数 | >8/10 |

### 性能指标

| 指标 | 说明 |
|------|------|
| **P50 延迟** | 50% 任务的执行时间 |
| **P95 延迟** | 95% 任务的执行时间 |
| **P99 延迟** | 99% 任务的执行时间 |
| **加速比** | 并行执行的加速倍数 |

## 最佳实践

### 1. 评估频率

- **每次提交**: 运行基础任务 (快速验证)
- **每天**: 运行完整评估 (全面检查)
- **每周**: 运行多模型对比 (性能优化)
- **每月**: 更新基线 (长期追踪)

### 2. 并行配置

```python
# CPU 密集型任务
evaluator = ComprehensiveEvaluator(
    parallel=True,
    max_workers=cpu_count(),
    use_processes=True
)

# I/O 密集型任务
evaluator = ComprehensiveEvaluator(
    parallel=True,
    max_workers=cpu_count() * 2,
    use_processes=False
)
```

### 3. 试验次数

```python
# 快速验证
n_trials = 1

# 稳定性测试
n_trials = 3

# 性能基准
n_trials = 5
```

### 4. LLM 评判

```python
# 开发阶段: 关闭 LLM 评判 (更快)
evaluator = ComprehensiveEvaluator(use_llm_judge=False)

# 正式评估: 开启 LLM 评判 (更准确)
evaluator = ComprehensiveEvaluator(use_llm_judge=True)
```

## 输出示例

### 控制台输出

```
================================================================================
Doraemon Code 完整评估系统
================================================================================

评估配置:
  任务文件: 4 个
  并行模式: 是
  并行度: 4
  试验次数: 3
  LLM 评判: 是

使用并行评估...
进度: 10/30
进度: 20/30
进度: 30/30

================================================================================
评估完成
================================================================================

核心指标:
  总任务数: 30
  成功任务: 27
  失败任务: 3
  成功率: 90.0%
  平均耗时: 2.34s
  总耗时: 70.20s

工具使用:
  总调用次数: 156
  平均调用/任务: 5.2
  最多使用的工具:
    - write_file: 45
    - read_file: 38
    - edit_file: 32

按难度统计:
  难度 1: 10/10 (100.0%) - 1.23s
  难度 5: 8/10 (80.0%) - 2.45s
  难度 8: 7/10 (70.0%) - 4.56s

LLM 评估分数:
  总体评分: 8.3/10
  任务完成: 8.5/10
  工具使用: 8.7/10
  代码质量: 7.9/10
```

### JSON 报告

```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "evaluation_type": "comprehensive",
    "parallel": true,
    "max_workers": 4,
    "n_trials": 3
  },
  "metrics": {
    "core_metrics": {
      "total_tasks": 30,
      "successful_tasks": 27,
      "success_rate": 0.9,
      "avg_execution_time": 2.34
    },
    "tool_metrics": {
      "total_tool_calls": 156,
      "avg_tool_calls_per_task": 5.2,
      "most_used_tools": [
        ["write_file", 45],
        ["read_file", 38]
      ]
    }
  }
}
```

## 故障排除

### 问题: 并行评估失败

**解决方案:**
```python
# 降低并行度
evaluator = ParallelEvaluator(max_workers=2)

# 或使用串行评估
evaluator = ComprehensiveEvaluator(parallel=False)
```

### 问题: LLM 评判超时

**解决方案:**
```python
# 关闭 LLM 评判
evaluator = ComprehensiveEvaluator(use_llm_judge=False)

# 或增加超时时间
judge = LLMJudgeEvaluator(timeout=60)
```

### 问题: 内存不足

**解决方案:**
```python
# 减少并行度
evaluator = ParallelEvaluator(max_workers=2)

# 减少试验次数
evaluator = ComprehensiveEvaluator(n_trials=1)

# 分批评估
for task_file in task_files:
    report = evaluator.run_full_evaluation(agent_factory, [task_file])
```

## 贡献指南

### 添加新任务

1. 选择合适的难度级别目录
2. 创建任务 JSON 文件
3. 遵循任务模板格式
4. 添加清晰的成功标准

### 添加新指标

1. 在 `metrics_collector.py` 中添加收集逻辑
2. 在 `_calculate_metrics()` 中添加计算逻辑
3. 在 `print_summary()` 中添加显示逻辑
4. 更新文档

### 添加新评估器

1. 继承 `AgentEvaluator` 基类
2. 实现评估逻辑
3. 添加测试
4. 更新文档

## 参考资料

- [Anthropic 评估最佳实践](../../docs/ANTHROPIC_EVALUATION_REFERENCE.md)
- [Agent 评估框架](../../docs/AGENT_EVALUATION_FRAMEWORK.md)
- [评估改进计划](../../docs/EVALUATION_IMPROVEMENT_PLAN.md)

## 版本历史

### v2.0 (2024-02-03)
- ✨ 新增并行评估器
- ✨ 新增指标收集器
- ✨ 新增完整评估系统
- ✨ 新增中级任务集
- ✨ 新增评估系统测试
- 📝 完善文档

### v1.0 (2024-01-27)
- 🎉 初始版本
- ✅ 基础评估器
- ✅ LLM 评判器
- ✅ 多模型对比器
- ✅ 基线管理器
