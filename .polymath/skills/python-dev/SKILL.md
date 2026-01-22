---
name: Python Development
description: Best practices for Python development in this project
triggers:
  - python
  - .py
  - pytest
  - pip
  - def 
  - class
  - import
priority: 20
files: []
---

## Python Development Guidelines

### Code Style
- Use 4 spaces for indentation (not tabs)
- Maximum line length: 100 characters
- Use type hints for all public functions
- Follow PEP 8 naming conventions:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Type Hints
Always include type hints:

```python
def process_data(items: list[str], config: dict[str, Any] | None = None) -> bool:
    """Process items with optional config."""
    ...
```

### Docstrings
Use Google-style docstrings:

```python
def calculate_score(value: int, multiplier: float = 1.0) -> float:
    """
    Calculate the weighted score.
    
    Args:
        value: Base value to calculate from
        multiplier: Optional weight multiplier
    
    Returns:
        The calculated score
    
    Raises:
        ValueError: If value is negative
    """
```

### Error Handling
- Use custom exceptions from `src/core/errors.py`
- Always provide helpful error messages
- Log errors before raising

### Testing
- Write tests in `tests/` mirroring `src/` structure
- Use pytest fixtures for common setup
- Aim for 60%+ coverage on core modules
