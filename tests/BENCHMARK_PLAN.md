# Doraemon Code 高级 Benchmark 计划

## 1. Benchmark 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Tier 4: 生产级评估                         │
│            开发者生产力研究 + 用户满意度                       │
├─────────────────────────────────────────────────────────────┤
│                   Tier 3: Agentic 行为                       │
│         工具编排 | 工作流完成 | 错误恢复 | 上下文管理           │
├─────────────────────────────────────────────────────────────┤
│                   Tier 2: 软件工程能力                        │
│         SWE-bench | 代码质量 | PR 创建 | Issue 解决           │
├─────────────────────────────────────────────────────────────┤
│                   Tier 1: 核心代码智能                        │
│         HumanEval+ | MBPP+ | BigCodeBench | Aider Polyglot   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Tier 1: 核心代码智能

### 2.1 HumanEval+ (基准代码生成)

```yaml
name: HumanEval+
source: EvalPlus
tasks: 164 Python 函数级问题
metrics:
  - pass@1: 首次通过率
  - pass@10: 10次尝试内通过率
extended_tests: 80x 测试用例扩展
timeout: 120s per task

target:
  pass@1: >85%  (对标 GPT-4: 82%, Claude-3.5: 92%)
  pass@10: >95%
```

**测试文件**: `tests/benchmarks/test_humaneval.py`

```python
@pytest.mark.benchmark
class TestHumanEval:
    """HumanEval+ 代码生成基准"""
    
    @pytest.fixture
    def humaneval_dataset(self):
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    
    async def test_function_generation(self, humaneval_dataset, benchmark):
        """测试函数级代码生成"""
        results = []
        for task in humaneval_dataset:
            result = await self.agent.run(
                input=f"Implement this function:\n{task['prompt']}"
            )
            passed = await self.verify_solution(result.code, task.test_cases)
            results.append(passed)
        
        pass_rate = sum(results) / len(results)
        assert pass_rate >= 0.85
```

### 2.2 MBPP+ (Python 基础能力)

```yaml
name: MBPP+
source: EvalPlus
tasks: 378 Python 基础问题
metrics:
  - pass@1: 首次通过率
extended_tests: 35x 测试用例扩展

target:
  pass@1: >90%
```

### 2.3 BigCodeBench (复杂指令)

```yaml
name: BigCodeBench
source: BigCode Project
tasks: 1,140 软件工程任务
splits:
  - complete: 代码补全
  - instruct: 自然语言指令
  - hard: 148 个最难题
metrics:
  - pass@1
  - functional_call_accuracy: 函数调用准确率

target:
  complete pass@1: >75%
  instruct pass@1: >70%
  hard pass@1: >50%
```

### 2.4 Aider Polyglot (多语言代码编辑)

```yaml
name: Aider Polyglot
source: Aider
tasks: 225 个 Exercism 最难题
languages: C++, Go, Java, JavaScript, Python, Rust
metrics:
  - percent_correct: 正确解决率
  - edit_format_compliance: 编辑格式符合率
  - cost: API 成本

target:
  percent_correct: >60%  (Aider GPT-4: 64%)
  edit_format_compliance: >95%
```

---

## 3. Tier 2: 软件工程能力

### 3.1 SWE-bench Verified (黄金标准)

```yaml
name: SWE-bench Verified
source: Princeton NLP + OpenAI
tasks: 500 个人工验证的 GitHub Issues
evaluation:
  - Docker 容器化
  - 真实代码库
  - 自动化测试验证
metrics:
  - resolved: 问题解决率
  - cost_per_issue: 每问题成本
  - time_to_resolution: 解决时间
  - first_try_rate: 首次成功

target:
  resolved: >50%  (Claude Code: 64%, GPT-4: 33%)
  cost_per_issue: <$2
  time_to_resolution: <5min avg
```

**测试文件**: `tests/benchmarks/test_swebench.py`

```python
@pytest.mark.benchmark
@pytest.mark.swebench
class TestSWEBench:
    """SWE-bench 代理能力基准"""
    
    @pytest.fixture
    def swebench_dataset(self):
        import swebench
        return swebench.load_dataset("princeton-nlp/SWE-bench_Verified")
    
    async def test_issue_resolution(self, swebench_dataset, benchmark):
        """测试真实 Issue 解决能力"""
        results = []
        for instance in swebench_dataset:
            # 1. 设置 Docker 环境
            container = await self.setup_container(instance)
            
            # 2. Agent 解决问题
            result = await self.agent.run(
                input=f"""
                Repository: {instance.repo}
                Issue: {instance.problem_statement}
                
                Please fix this issue. Make sure all tests pass.
                """,
                cwd=container.workdir
            )
            
            # 3. 验证补丁
            resolved = await self.run_tests(container, instance.test_patch)
            results.append({
                "instance_id": instance.instance_id,
                "resolved": resolved,
                "cost": result.cost,
                "time": result.duration,
                "tool_calls": len(result.tool_history),
            })
            
            await self.cleanup_container(container)
        
        # 生成报告
        report = self.generate_report(results)
        assert report["resolved_rate"] >= 0.50
    
    async def test_multifile_coordination(self):
        """测试多文件协调能力"""
        pass
    
    async def test_context_navigation(self):
        """测试上下文导航能力"""
        pass
```

### 3.2 代码质量评估

```yaml
name: Code Quality
tasks:
  - style_consistency: 代码风格一致性
  - documentation: 文档质量
  - security: 安全漏洞检测
  - performance: 性能优化建议

metrics:
  - style_score: 风格评分 (0-100)
  - doc_coverage: 文档覆盖率
  - security_issues: 安全问题数
  - perf_improvement: 性能提升比例
```

### 3.3 PR 创建与审核

```yaml
name: PR Workflow
tasks:
  - create_pr: 从 Issue 创建 PR
  - review_pr: 审核 PR 提建议
  - resolve_conflicts: 解决合并冲突

metrics:
  - pr_quality: PR 质量 (人工评审)
  - review_accuracy: 审核准确性
  - conflict_resolution_rate: 冲突解决率
```

---

## 4. Tier 3: Agentic 行为评估

### 4.1 工具编排能力

```yaml
name: Tool Orchestration
tasks:
  git_operations:
    - git_commit: 创建有意义的 commit
    - git_branch: 创建/切换分支
    - git_merge: 合并分支解决冲突
  
  package_management:
    - npm_install: 安装依赖
    - pip_install: Python 包管理
    - dependency_update: 更新依赖
  
  build_systems:
    - make_build: Makefile 构建
    - cmake_build: CMake 构建
    - docker_build: Docker 构建

metrics:
  - tool_success_rate: 工具调用成功率
  - tool_selection_accuracy: 工具选择准确率
  - tool_sequence_efficiency: 工具序列效率
```

**测试文件**: `tests/benchmarks/test_agentic_tools.py`

```python
@pytest.mark.benchmark
class TestAgenticTools:
    """Agentic 工具使用基准"""
    
    async def test_git_workflow(self, benchmark):
        """测试 Git 工作流"""
        task = """
        1. Create a new branch called 'feature/auth'
        2. Make changes to src/auth.py
        3. Commit with a descriptive message
        4. Push to remote
        """
        
        result = await self.agent.run(task)
        
        # 验证分支创建
        assert await self.git.branch_exists("feature/auth")
        # 验证提交消息质量
        commit_msg = await self.git.get_commit_message()
        assert len(commit_msg) > 10
        assert "auth" in commit_msg.lower()
    
    async def test_error_recovery(self, benchmark):
        """测试错误恢复能力"""
        task = "Run the tests and fix any failures"
        
        # 故意在代码中注入错误
        await self.inject_bug()
        
        result = await self.agent.run(task)
        
        # 验证错误被发现并修复
        assert "test failure" in result.response.lower()
        tests_passed = await self.run_tests()
        assert tests_passed
    
    async def test_tool_selection(self, benchmark):
        """测试工具选择准确性"""
        tasks = [
            ("Read the README file", "read"),
            ("Create a new file called test.py", "write"),
            ("Search for TODO comments", "search"),
            ("Run the test suite", "run"),
        ]
        
        correct = 0
        for task, expected_tool in tasks:
            result = await self.agent.run(task)
            if expected_tool in [tc.name for tc in result.tool_calls]:
                correct += 1
        
        accuracy = correct / len(tasks)
        assert accuracy >= 0.90  # 90% 工具选择准确率
```

### 4.2 工作流完成能力

```yaml
name: Workflow Completion
tasks:
  end_to_end:
    - add_feature: 添加完整功能
    - fix_bug: 修复 Bug 并验证
    - refactor: 重构代码
    - migrate: 迁移代码库

metrics:
  - completion_rate: 完成率
  - steps_to_complete: 完成步数
  - human_intervention_rate: 人工干预率
```

### 4.3 上下文管理能力

```yaml
name: Context Management
tasks:
  - large_codebase: 在大型代码库中导航
  - relevant_files: 识别相关文件
  - dependency_understanding: 理解依赖关系
  - cross_file_refactoring: 跨文件重构

metrics:
  - files_explored: 探索文件数
  - relevant_file_ratio: 相关文件比例
  - context_efficiency: 上下文效率 (有效信息 / 总token)
```

### 4.4 错误恢复能力

```yaml
name: Error Recovery
tasks:
  - test_failure: 测试失败后修复
  - build_failure: 构建失败后修复
  - runtime_error: 运行时错误修复
  - lint_error: Lint 错误修复

metrics:
  - recovery_rate: 恢复成功率
  - attempts_to_fix: 修复尝试次数
  - time_to_fix: 修复时间
```

---

## 5. Tier 4: 生产级评估

### 5.1 开发者生产力研究

```yaml
name: Developer Productivity Study
method: A/B 测试 + 问卷调查
participants: 50+ 开发者
duration: 2 周

metrics:
  - task_completion_time: 任务完成时间
  - code_quality_score: 代码质量评分
  - developer_satisfaction: 开发者满意度
  - learning_curve: 学习曲线
```

### 5.2 用户满意度调查

```yaml
name: User Satisfaction Survey
dimensions:
  - accuracy: 准确性
  - speed: 响应速度
  - usability: 易用性
  - trust: 信任度
  - overall: 总体满意度

target:
  nps: >50  # Net Promoter Score
  satisfaction: >4.0/5.0
```

---

## 6. Benchmark 数据集结构

```
tests/benchmarks/
├── datasets/
│   ├── humaneval+/           # HumanEval+ 数据集
│   ├── mbpp+/                # MBPP+ 数据集
│   ├── bigcodebench/         # BigCodeBench 数据集
│   ├── aider-polyglot/       # Aider 多语言数据集
│   └── swebench-verified/    # SWE-bench Verified
├── harness/
│   ├── docker_runner.py      # Docker 评估运行器
│   ├── test_executor.py      # 测试执行器
│   └── result_aggregator.py  # 结果聚合器
├── test_humaneval.py
├── test_mbpp.py
├── test_bigcodebench.py
├── test_aider_polyglot.py
├── test_swebench.py
├── test_agentic_tools.py
├── test_error_recovery.py
└── conftest.py
```

---

## 7. 评估流程

```yaml
phases:
  quick_check:
    duration: 5 minutes
    tests:
      - HumanEval+ (sample 20)
      - MBPP+ (sample 20)
    purpose: 快速回归检测

  daily_benchmark:
    duration: 30 minutes
    tests:
      - HumanEval+ (full)
      - MBPP+ (full)
      - Aider Polyglot (sample 50)
    purpose: 每日质量监控

  weekly_evaluation:
    duration: 4 hours
    tests:
      - BigCodeBench (full)
      - Aider Polyglot (full)
      - Agentic Tools (all)
    purpose: 周度能力评估

  release_evaluation:
    duration: 24 hours
    tests:
      - SWE-bench Verified (full)
      - All Agentic tests
      - Production study
    purpose: 发布前完整评估
```

---

## 8. 目标对比表

| Benchmark | GPT-4 | Claude 3.5 | Claude Code | Doraemon Target |
|-----------|-------|------------|-------------|-----------------|
| HumanEval+ pass@1 | 82% | 92% | - | **>85%** |
| MBPP+ pass@1 | 86% | 95% | - | **>90%** |
| BigCodeBench | 51% | 71% | - | **>60%** |
| Aider Polyglot | 64% | - | - | **>60%** |
| SWE-bench Verified | 33% | - | 64% | **>50%** |
| Tool Selection | - | - | ~95% | **>90%** |
| Error Recovery | - | - | ~85% | **>80%** |

---

## 9. CI 集成

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 每日 2am UTC
  workflow_dispatch:
    inputs:
      benchmark_level:
        description: 'Benchmark level'
        required: true
        default: 'quick'
        type: choice
        options:
          - quick
          - daily
          - weekly
          - release

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: pip install -e ".[dev,benchmark]"
      
      - name: Run Quick Benchmark
        if: inputs.benchmark_level == 'quick'
        run: pytest tests/benchmarks -m "quick" -v
      
      - name: Run Daily Benchmark
        if: inputs.benchmark_level == 'daily' || github.event_name == 'schedule'
        run: pytest tests/benchmarks -m "daily" -v
      
      - name: Run Weekly Benchmark
        if: inputs.benchmark_level == 'weekly'
        run: pytest tests/benchmarks -m "weekly" -v
      
      - name: Run Release Benchmark
        if: inputs.benchmark_level == 'release'
        run: |
          pytest tests/benchmarks -v --timeout=86400
          python scripts/generate_benchmark_report.py
      
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_results/
```

---

## 10. 下一步行动

| 优先级 | 任务 | 工作量 |
|--------|------|--------|
| P0 | 集成 HumanEval+ 评估 | 2天 |
| P0 | 集成 SWE-bench Verified 评估 | 3天 |
| P1 | 实现 Agentic 工具测试 | 2天 |
| P1 | 实现 Docker 评估运行器 | 2天 |
| P2 | 集成 BigCodeBench | 1天 |
| P2 | 集成 Aider Polyglot | 1天 |
| P2 | 创建 Dashboard 可视化 | 2天 |
