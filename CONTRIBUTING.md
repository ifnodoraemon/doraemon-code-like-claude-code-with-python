# Contributing to Doraemon Code

感谢你有兴趣为 Doraemon Code 做贡献！本文档将帮助你了解如何参与项目开发。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码风格指南](#代码风格指南)
- [测试要求](#测试要求)
- [提交代码](#提交代码)
- [报告问题](#报告问题)
- [功能建议](#功能建议)

## 行为准则

### 我们的承诺

为了营造一个开放和友好的环境，我们作为贡献者和维护者承诺：无论年龄、体型、残疾、种族、性别认同和表达、经验水平、教育程度、社会经济地位、国籍、外貌、种族、宗教或性取向如何，参与我们的项目和社区都将为每个人提供无骚扰的体验。

### 我们的标准

积极行为示例：
- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

## 如何贡献

### 报告 Bug

如果你发现了 bug，请通过 GitHub Issues 提交报告。提交前请：

1. **检查现有 Issues** - 确保问题尚未被报告
2. **使用 Issue 模板** - 填写所有必需的信息
3. **提供详细信息**：
   - 操作系统和 Python 版本
   - 复现步骤
   - 期望行为 vs 实际行为
   - 错误日志或截图

### 建议新功能

我们欢迎新功能建议！请：

1. 先在 Issues 中讨论你的想法
2. 说明功能的使用场景
3. 等待维护者反馈后再开始实现

### 提交代码

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 进行更改
4. 确保测试通过
5. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
6. 推送到分支 (`git push origin feature/AmazingFeature`)
7. 创建 Pull Request

## 开发环境设置

### 前置要求

- Python 3.10 或更高版本
- pip 或 uv (推荐)
- Git

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/ifnodoraemon/doraemon-code.git
cd doraemon-code

# 2. 创建虚拟环境 (推荐使用 uv)
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

# 3. 安装开发依赖
pip install -e ".[dev]"
# 或使用 uv
uv pip install -e ".[dev]"

# 4. 安装 pre-commit 钩子 (可选)
pre-commit install

# 5. 验证安装
python -c "import src; print('Installation successful!')"
```

### 环境配置

创建 `.env` 文件（参考 `.env.example`）：

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置文件，添加你的 API keys
# GOOGLE_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

## 代码风格指南

### Python 代码风格

我们遵循以下规范：

- **PEP 8** - Python 代码风格指南
- **Line Length**: 100 字符（不是 79）
- **Type Hints**: 所有公共函数必须有类型标注
- **Docstrings**: 使用 Google 风格的文档字符串

### 代码格式化工具

我们使用以下工具确保代码质量：

```bash
# 格式化代码
ruff format src/ tests/

# 检查代码风格
ruff check src/ tests/

# 自动修复问题
ruff check --fix src/ tests/

# 类型检查
mypy src/
```

### 代码风格示例

```python
from typing import Any


def example_function(param1: str, param2: int | None = None) -> dict[str, Any]:
    """
    这是一个示例函数，展示代码风格要求。

    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述（可选）

    Returns:
        返回值的描述

    Raises:
        ValueError: 当参数无效时抛出

    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        {'param1': 'test', 'param2': 42}
    """
    if not param1:
        raise ValueError("param1 cannot be empty")

    result: dict[str, Any] = {
        "param1": param1,
        "param2": param2,
    }
    return result
```

### 命名约定

- **变量和函数**: `snake_case`
- **类**: `PascalCase`
- **常量**: `UPPER_SNAKE_CASE`
- **私有成员**: `_leading_underscore`
- **模块**: `lowercase` (不带下划线)

## 测试要求

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/core/test_model_client.py

# 运行特定测试
pytest tests/core/test_model_client.py::TestDirectModelClient::test_context_manager_connects_and_closes

# 运行测试并显示覆盖率
pytest tests/ --cov=src --cov-report=term-missing

# 跳过慢速测试
pytest tests/ -m "not slow"

# 只运行集成测试
pytest tests/ -m integration
```

### 测试覆盖率要求

- **最低覆盖率**: 60% (当前要求)
- **目标覆盖率**: 85%
- 新代码的覆盖率应该至少达到 80%

### 测试最佳实践

1. **每个测试一个断言** - 保持测试简单和专注
2. **使用描述性名称** - 测试名应该说明它在测试什么
3. **测试边界条件** - 不仅测试正常情况，还要测试边缘情况
4. **Mock 外部依赖** - 使用 `unittest.mock` 或 `pytest-mock`
5. **清理资源** - 使用 fixtures 清理测试数据

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_client():
    """Fixture that provides a mock client for testing."""
    client = MagicMock()
    client.aclose = AsyncMock()
    yield client
    # Cleanup if needed


def test_example(mock_client):
    """Test that demonstrates proper testing patterns."""
    # Arrange
    expected = "test_value"

    # Act
    result = some_function(mock_client)

    # Assert
    assert result == expected
    mock_client.some_method.assert_called_once()
```

## 提交代码

### 提交消息格式

我们遵循约定式提交规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**:
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行的变动）
- `refactor`: 重构（既不是新增功能，也不是修改 bug）
- `test`: 增加测试
- `chore`: 构建过程或辅助工具的变动

**示例**:
```
feat(core): add support for Claude 3.5 Sonnet

- Add Claude 3.5 Sonnet model ID to supported models
- Update model context window sizes
- Add tests for new model

Closes #123
```

### Pull Request 流程

1. **确保测试通过** - 所有 CI 检查必须通过
2. **更新文档** - 如果需要，更新 README.md 或其他文档
3. **添加 CHANGELOG 条目** - 在 CHANGELOG.md 中记录你的更改
4. **请求审查** - 等待至少一位维护者的审查
5. **响应反馈** - 及时响应审查意见

### Pull Request 检查清单

- [ ] 代码遵循项目的风格指南
- [ ] 已进行自我审查
- [ ] 代码有适当的注释
- [ ] 文档已更新
- [ ] 没有引入新的警告
- [ ] 添加了测试证明修复有效或功能工作
- [ ] 新的和现有的单元测试通过
- [ ] 任何依赖更改都已被合并和发布

## 报告问题

### 安全问题

如果你发现了安全漏洞，请**不要**通过公开的 GitHub Issues 报告。请发送邮件至安全团队。

### 一般问题

对于一般问题和讨论，请：
1. 查看现有 Issues
2. 查阅文档
3. 创建新 Issue 并使用适当的标签

## 功能建议

我们欢迎功能建议！请：

1. **清晰描述功能** - 它应该做什么？
2. **说明使用场景** - 为什么需要这个功能？
3. **提供示例** - 展示如何使用这个功能
4. **考虑实现** - 如果你有想法，简要说明如何实现

## 获得帮助

- **GitHub Issues**: 用于 bug 报告和功能建议
- **文档**: 查阅项目 README.md 和 docs/ 目录
- **代码注释**: 阅读源代码中的注释

## 许可证

通过贡献代码，你同意你的贡献将根据项目的 MIT 许可证进行许可。

---

再次感谢你对 Doraemon Code 的贡献！🎉
