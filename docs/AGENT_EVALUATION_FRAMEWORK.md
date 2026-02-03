# 🤖 Agent 效果评估框架

## 🎯 评估目标

评估 Doraemon Code Agent 在真实场景下的表现，关注：
- **任务完成能力** - 能否正确完成用户任务
- **工具使用能力** - 能否正确选择和使用工具
- **问题解决能力** - 能否处理复杂问题和错误
- **代码质量** - 生成的代码是否高质量
- **用户体验** - 交互是否自然、有帮助

## 📊 Agent 能力评估维度

### 1. 任务完成能力 (Task Completion)

#### 评估指标
- **成功率** (Success Rate): 任务完成百分比
- **首次成功率** (First-Try Success): 无需重试的成功率
- **平均尝试次数** (Average Attempts): 完成任务的平均尝试次数

#### 评估方法
```python
def evaluate_task_completion(agent, task):
    """评估任务完成能力"""
    result = {
        "task_id": task["id"],
        "success": False,
        "attempts": 0,
        "errors": [],
    }
    
    max_attempts = 3
    for attempt in range(max_attempts):
        result["attempts"] += 1
        try:
            response = agent.execute(task["prompt"])
            if check_assertions(response, task["assertions"]):
                result["success"] = True
                break
        except Exception as e:
            result["errors"].append(str(e))
    
    return result
```

### 2. 工具使用能力 (Tool Usage)

#### 评估指标
- **工具选择准确率** (Tool Selection Accuracy): 选择正确工具的比例
- **工具调用效率** (Tool Call Efficiency): 完成任务所需的工具调用次数
- **工具参数正确率** (Parameter Accuracy): 工具参数正确的比例

#### 评估方法
```python
def evaluate_tool_usage(agent, task):
    """评估工具使用能力"""
    response = agent.execute(task["prompt"])
    
    # 分析工具调用
    tool_calls = extract_tool_calls(response)
    expected_tools = task.get("expected_tools", [])
    
    return {
        "tool_selection_accuracy": calculate_tool_accuracy(tool_calls, expected_tools),
        "tool_call_count": len(tool_calls),
        "tool_call_efficiency": len(tool_calls) / task.get("optimal_calls", len(tool_calls)),
        "parameter_accuracy": check_parameter_correctness(tool_calls),
    }
```

### 3. 问题解决能力 (Problem Solving)

#### 评估指标
- **错误恢复率** (Error Recovery Rate): 遇到错误后能恢复的比例
- **调试能力** (Debugging Ability): 能否诊断和修复问题
- **适应性** (Adaptability): 能否处理意外情况

#### 评估场景
```python
PROBLEM_SOLVING_SCENARIOS = [
    {
        "id": "error-recovery-001",
        "scenario": "文件不存在错误",
        "prompt": "Read the file 'nonexistent.txt' and summarize it",
        "expected_behavior": "检测文件不存在，提示用户或创建文件",
    },
    {
        "id": "error-recovery-002",
        "scenario": "权限错误",
        "prompt": "Write to /etc/passwd",
        "expected_behavior": "识别权限问题，拒绝操作并说明原因",
    },
    {
        "id": "debugging-001",
        "scenario": "代码有 bug",
        "prompt": "Fix the bug in buggy_code.py",
        "expected_behavior": "识别 bug，提供修复方案",
    },
]
```

### 4. 代码质量 (Code Quality)

#### 评估指标
- **语法正确性** (Syntax Correctness): 代码无语法错误
- **功能正确性** (Functional Correctness): 代码实现预期功能
- **代码风格** (Code Style): 符合最佳实践
- **可维护性** (Maintainability): 代码易读易维护

#### 评估方法
```python
def evaluate_code_quality(generated_code):
    """评估生成代码的质量"""
    scores = {
        "syntax": check_syntax(generated_code),
        "functionality": test_functionality(generated_code),
        "style": run_linter(generated_code),
        "maintainability": calculate_complexity(generated_code),
    }
    
    return {
        "overall_score": sum(scores.values()) / len(scores),
        "details": scores,
    }
```

### 5. 上下文理解 (Context Understanding)

#### 评估指标
- **上下文保持** (Context Retention): 能否记住之前的对话
- **指令理解** (Instruction Understanding): 能否正确理解用户意图
- **歧义处理** (Ambiguity Handling): 能否处理模糊指令

#### 评估场景
```python
CONTEXT_SCENARIOS = [
    {
        "id": "context-001",
        "conversation": [
            {"role": "user", "content": "Create a file called data.txt"},
            {"role": "assistant", "content": "Created data.txt"},
            {"role": "user", "content": "Now add 'hello' to it"},
        ],
        "expected": "应该向 data.txt 添加内容，而不是创建新文件",
    },
]
```

### 6. 用户体验 (User Experience)

#### 评估指标
- **响应清晰度** (Response Clarity): 回复是否清晰易懂
- **有用性** (Helpfulness): 回复是否有帮助
- **礼貌性** (Politeness): 交互是否友好
- **主动性** (Proactiveness): 能否主动提供建议

#### 评估方法
```python
def evaluate_user_experience(response):
    """使用 LLM-as-Judge 评估用户体验"""
    judge_prompt = f"""
    评估以下 AI 助手的回复质量：
    
    回复: {response}
    
    评分标准 (1-10):
    1. 清晰度: 回复是否清晰易懂
    2. 有用性: 回复是否有帮助
    3. 礼貌性: 交互是否友好
    4. 主动性: 是否提供额外建议
    
    请给出每项评分和总分。
    """
    
    return llm_judge(judge_prompt)
```

## 🎯 Agent 能力分级

### Level 1: 基础能力 (Basic)
- ✅ 简单文件操作 (读、写、列表)
- ✅ 基础代码生成 (函数、类)
- ✅ 简单问题回答

### Level 2: 中级能力 (Intermediate)
- ✅ 多文件操作
- ✅ 代码编辑和重构
- ✅ 错误诊断和修复
- ✅ 上下文理解

### Level 3: 高级能力 (Advanced)
- ✅ 复杂项目创建
- ✅ 架构设计
- ✅ 性能优化
- ✅ 安全审计

### Level 4: 专家能力 (Expert)
- ✅ 端到端项目开发
- ✅ 复杂问题解决
- ✅ 自主决策
- ✅ 创新解决方案

## 📋 评估任务集

### 基础任务 (30个)
```json
{
  "category": "basic",
  "tasks": [
    {
      "id": "basic-001",
      "name": "创建 Python 文件",
      "prompt": "Create a Python file that prints 'Hello World'",
      "difficulty": 1,
      "expected_tools": ["write_file"],
      "success_criteria": ["文件存在", "内容正确", "语法正确"]
    }
  ]
}
```

### 中级任务 (40个)
```json
{
  "category": "intermediate",
  "tasks": [
    {
      "id": "inter-001",
      "name": "调试代码",
      "prompt": "Fix the bug in calculator.py that causes division by zero",
      "difficulty": 5,
      "expected_tools": ["read_file", "edit_file"],
      "success_criteria": ["识别 bug", "正确修复", "添加错误处理"]
    }
  ]
}
```

### 高级任务 (20个)
```json
{
  "category": "advanced",
  "tasks": [
    {
      "id": "adv-001",
      "name": "创建 REST API",
      "prompt": "Create a simple REST API with FastAPI for user management",
      "difficulty": 8,
      "expected_tools": ["write_file", "execute_python"],
      "success_criteria": ["API 结构正确", "端点实现", "错误处理", "文档"]
    }
  ]
}
```

### 专家任务 (10个)
```json
{
  "category": "expert",
  "tasks": [
    {
      "id": "exp-001",
      "name": "端到端项目",
      "prompt": "Create a complete todo app with backend, frontend, and tests",
      "difficulty": 10,
      "expected_tools": ["write_file", "execute_python", "git"],
      "success_criteria": ["完整架构", "所有功能", "测试覆盖", "文档完整"]
    }
  ]
}
```

## 🔬 评估方法论

### 1. 自动化评估
```python
def automated_evaluation(agent, task_set):
    """自动化评估 Agent 性能"""
    results = []
    
    for task in task_set:
        result = {
            "task_id": task["id"],
            "task_name": task["name"],
            "difficulty": task["difficulty"],
        }
        
        # 执行任务
        start_time = time.time()
        response = agent.execute(task["prompt"])
        result["execution_time"] = time.time() - start_time
        
        # 检查成功标准
        result["success"] = check_success_criteria(response, task["success_criteria"])
        
        # 评估工具使用
        result["tool_usage"] = evaluate_tool_usage(response, task)
        
        # 评估代码质量
        if has_code(response):
            result["code_quality"] = evaluate_code_quality(extract_code(response))
        
        results.append(result)
    
    return aggregate_results(results)
```

### 2. LLM-as-Judge 评估
```python
def llm_judge_evaluation(agent, task):
    """使用 LLM 作为评判者"""
    response = agent.execute(task["prompt"])
    
    judge_prompt = f"""
    评估 AI Agent 对以下任务的完成情况：
    
    任务: {task["prompt"]}
    期望结果: {task["expected_outcome"]}
    实际响应: {response}
    
    评分标准 (1-10):
    1. 任务完成度: 是否完成了任务
    2. 解决方案质量: 解决方案是否优秀
    3. 代码质量: 代码是否高质量
    4. 用户体验: 交互是否友好
    
    请给出每项评分、总分和详细理由。
    """
    
    return llm_judge(judge_prompt)
```

### 3. 人工评估
```python
def human_evaluation(agent, task_set):
    """人工评估 Agent 性能"""
    # 选择代表性任务
    sample_tasks = random.sample(task_set, min(10, len(task_set)))
    
    results = []
    for task in sample_tasks:
        response = agent.execute(task["prompt"])
        
        # 人工评分
        print(f"\n任务: {task['prompt']}")
        print(f"响应: {response}")
        
        scores = {
            "completeness": int(input("完成度 (1-10): ")),
            "quality": int(input("质量 (1-10): ")),
            "usability": int(input("易用性 (1-10): ")),
        }
        
        results.append(scores)
    
    return aggregate_results(results)
```

## 📊 评估报告模板

```markdown
# Agent 效果评估报告

## 执行摘要
- 评估日期: {date}
- Agent 版本: {version}
- 评估任务数: {task_count}
- 总体评分: {overall_score}/10

## 能力评估

### 1. 任务完成能力
- 成功率: {success_rate}%
- 首次成功率: {first_try_rate}%
- 平均尝试次数: {avg_attempts}

### 2. 工具使用能力
- 工具选择准确率: {tool_accuracy}%
- 平均工具调用次数: {avg_tool_calls}
- 工具调用效率: {tool_efficiency}%

### 3. 代码质量
- 语法正确性: {syntax_score}/10
- 功能正确性: {functionality_score}/10
- 代码风格: {style_score}/10
- 可维护性: {maintainability_score}/10

### 4. 用户体验
- 响应清晰度: {clarity_score}/10
- 有用性: {helpfulness_score}/10
- 礼貌性: {politeness_score}/10

## 能力分级
- 基础能力: {basic_level}
- 中级能力: {intermediate_level}
- 高级能力: {advanced_level}
- 专家能力: {expert_level}

## 失败案例分析
{failure_analysis}

## 改进建议
{improvement_suggestions}
```

---

**文档版本**: 1.0
**创建日期**: 2026-02-03
