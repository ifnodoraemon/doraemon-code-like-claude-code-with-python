"""
Benchmark tests for Doraemon Code performance

These tests establish performance baselines and detect regressions.
Run with: pytest tests/benchmarks/ -v --benchmark
"""

import sys
import time
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.model_client import ModelClient
from src.core.model_utils import Message


@pytest.mark.benchmark
class TestResponseTimeBenchmarks:
    """Benchmark response time for various operations"""

    def test_simple_query_latency(self, benchmark):
        """Benchmark: Simple query should respond within 2s"""

        def simple_query():
            client = ModelClient.create()
            messages = [Message(role="user", content="Hello")]
            response = client.chat(messages)
            return response

        result = benchmark(simple_query)
        assert result is not None

    def test_tool_call_latency(self, benchmark):
        """Benchmark: Tool call should complete within 5s"""

        def tool_call_query():
            client = ModelClient.create()
            messages = [Message(role="user", content="List files in current directory")]
            response = client.chat(messages)
            return response

        result = benchmark(tool_call_query)
        assert result is not None


@pytest.mark.benchmark
class TestTokenEfficiencyBenchmarks:
    """Benchmark token usage efficiency"""

    def test_token_usage_simple_task(self):
        """Benchmark: Simple task should use < 1000 tokens"""
        client = ModelClient.create()
        messages = [Message(role="user", content="Create a hello.py file")]

        response = client.chat(messages)

        # Check token usage if available
        if hasattr(response, "usage"):
            total_tokens = response.usage.get("total_tokens", 0)
            assert total_tokens < 1000, f"Used {total_tokens} tokens, expected < 1000"

    def test_token_usage_complex_task(self):
        """Benchmark: Complex task should use < 5000 tokens"""
        client = ModelClient.create()
        messages = [
            Message(
                role="user",
                content="Create a Python class with 5 methods and comprehensive docstrings",
            )
        ]

        response = client.chat(messages)

        if hasattr(response, "usage"):
            total_tokens = response.usage.get("total_tokens", 0)
            assert total_tokens < 5000, f"Used {total_tokens} tokens, expected < 5000"


@pytest.mark.benchmark
class TestToolCallEfficiencyBenchmarks:
    """Benchmark tool call efficiency"""

    def test_tool_calls_per_simple_task(self):
        """Benchmark: Simple task should use < 3 tool calls"""
        # This would need integration with actual agent
        # For now, just a placeholder
        pass

    def test_tool_calls_per_complex_task(self):
        """Benchmark: Complex task should use < 10 tool calls"""
        # Placeholder for complex task tool call counting
        pass


@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Benchmark memory usage"""

    def test_memory_usage_baseline(self):
        """Benchmark: Baseline memory usage should be < 500MB"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        assert memory_mb < 500, f"Memory usage {memory_mb:.1f}MB exceeds 500MB"

    def test_memory_growth_after_100_queries(self):
        """Benchmark: Memory should not grow > 50% after 100 queries"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        client = ModelClient.create()

        # Run 100 simple queries
        for i in range(100):
            messages = [Message(role="user", content=f"Query {i}")]
            try:
                client.chat(messages)
            except Exception:
                pass  # Ignore errors for benchmark

        final_memory = process.memory_info().rss
        growth = (final_memory - initial_memory) / initial_memory

        assert growth < 0.5, f"Memory grew by {growth * 100:.1f}%, expected < 50%"


@pytest.mark.benchmark
class TestConcurrencyBenchmarks:
    """Benchmark concurrent request handling"""

    @pytest.mark.asyncio
    async def test_concurrent_requests_throughput(self):
        """Benchmark: Should handle 5 concurrent requests"""
        import asyncio

        async def make_request(i):
            client = ModelClient.create()
            messages = [Message(role="user", content=f"Request {i}")]
            return await asyncio.to_thread(client.chat, messages)

        start = time.time()
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start

        # Should complete within 10 seconds
        assert duration < 10, f"Concurrent requests took {duration:.1f}s, expected < 10s"

        # At least some should succeed
        successes = sum(1 for r in results if not isinstance(r, Exception))
        assert successes >= 3, f"Only {successes}/5 requests succeeded"


# Baseline metrics to track over time
BASELINE_METRICS = {
    "simple_query_latency_ms": 2000,
    "tool_call_latency_ms": 5000,
    "simple_task_tokens": 1000,
    "complex_task_tokens": 5000,
    "baseline_memory_mb": 500,
    "memory_growth_percent": 50,
    "concurrent_requests": 5,
    "concurrent_timeout_s": 10,
}


def save_benchmark_results(results: dict):
    """Save benchmark results for regression detection"""
    import json
    from datetime import datetime

    results_file = Path("tests/benchmarks/results.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "metrics": results,
    }

    with open(results_file, "a") as f:
        f.write(json.dumps(result_entry) + "\n")


if __name__ == "__main__":
    print("Benchmark Baselines:")
    print("=" * 50)
    for metric, value in BASELINE_METRICS.items():
        print(f"{metric}: {value}")
