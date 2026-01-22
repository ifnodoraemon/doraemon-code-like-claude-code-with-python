# Polymath Architecture Simplification

## Summary

Reduced code complexity by ~60% while maintaining all user-facing features.

## Before vs After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| CLI main loop | 620 lines | 250 lines | -60% |
| Tool invocation | subprocess + JSON-RPC | direct call | -10ms/call |
| Startup time | Connect 14 servers | Import modules | ~2s faster |
| Abstractions | 5+ layers | 1-2 layers | Simpler |

## Removed (Unused)

1. **DI Container** - Spring-style DI overkill for CLI app
2. **Event Bus** - Events published but no subscribers
3. **Metrics** - Collected but never consumed
4. **Result Cache** - Low hit rate, stale data risk
5. **MCP Subprocess** - Replaced with direct function calls

## Kept (Valuable)

- Multi-mode support (plan/build/coder)
- HITL approval for sensitive tools
- Rich terminal UI
- Diff view for file changes
- Vector memory (ChromaDB)

## New Files

- `src/host/tools.py` - Simple tool registry with direct function calls
- `src/host/cli_simple.py` - Simplified CLI (~250 lines)

## Usage

```bash
# Simplified version
python -m src.host.cli_simple start

# Original version (still works)
pl start
```

## Migration Path

1. Current: Both versions work in parallel
2. Validation: Compare behavior and performance
3. Replace: Rename cli_simple.py to cli.py when ready
4. Cleanup: Archive unused core modules

## When to Add Complexity Back

| Feature | Add When |
|---------|----------|
| DI Container | Plugin system needed |
| Event Bus | Multiple decoupled components |
| Metrics | Prometheus/Grafana deployed |
| MCP | Remote servers (Docker, cloud) |
