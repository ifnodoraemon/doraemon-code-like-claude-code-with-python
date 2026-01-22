---
name: Git Workflow
description: Git conventions and workflow for this project
triggers:
  - git
  - commit
  - branch
  - merge
  - pull request
  - PR
priority: 15
files: []
---

## Git Workflow

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation changes

### Commit Messages
Follow conventional commits:

```
type(scope): description

[optional body]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(cli): add /context command for stats
fix(context): correct token estimation for Chinese text
refactor(skills): extract skill loading to separate module
```

### Before Committing
1. Run `ruff check src/` to check linting
2. Run `pytest tests/` to verify tests pass
3. Review changes with `git diff`

### Pull Requests
- Keep PRs focused on single concern
- Include description of changes
- Reference related issues if any
