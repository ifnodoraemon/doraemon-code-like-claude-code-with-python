"""
Subprocess Utilities

Shared helpers for safe subprocess execution across the codebase.
"""

import os

_DANGEROUS_ENV_VARS = frozenset({
    "PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    "PYTHONPATH", "NODE_OPTIONS", "BASH_ENV", "ENV", "CDPATH",
    "PERL5OPT", "RUBYOPT", "JAVA_TOOL_OPTIONS",
})


def prepare_safe_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Build process env, filtering dangerous overrides."""
    process_env = os.environ.copy()
    if env:
        safe = {k: v for k, v in env.items() if k.upper() not in _DANGEROUS_ENV_VARS}
        process_env.update(safe)
    return process_env
