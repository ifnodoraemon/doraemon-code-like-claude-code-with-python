import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.security import validate_path
from src.servers.memory import save_note, search_notes


# --------------------------
# Security Evals (Red Teaming)
# --------------------------
def test_security_path_traversal():
    """Eval: System should block access to parent directories."""
    # 模拟攻击路径
    attack_path = "../../../etc/passwd"

    with pytest.raises(PermissionError) as excinfo:
        validate_path(attack_path)

    assert "Access Denied" in str(excinfo.value)
    print("✅ Security Eval Passed: Path traversal blocked.")


def test_security_allowed_path():
    """Eval: System should allow access to project files."""
    # 创建一个临时测试文件
    test_file = "test_safe.txt"
    with open(test_file, "w") as f:
        f.write("safe content")

    try:
        path = validate_path(test_file)
        assert path == os.path.abspath(test_file)
        print("✅ Security Eval Passed: Safe path allowed.")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


# --------------------------
# Memory Evals (RAG Accuracy)
# --------------------------
def test_memory_ingestion_and_retrieval():
    """Eval: Saved notes should be retrievable by semantic search."""
    # 1. Save a unique fact
    unique_fact = "Polymath agent was created in 2025 by a visionary developer."
    save_note("History", unique_fact, collection_name="test_eval")

    # 2. Retrieve using semantic query (not exact match)
    query = "When was this AI agent born?"
    result = search_notes(query, collection_name="test_eval")

    # 3. Assertions
    assert "2025" in result
    assert "History" in result
    print("✅ Memory Eval Passed: Semantic retrieval working.")

    # Cleanup (Optional: remove data from vector db if possible, or use mock)


if __name__ == "__main__":
    # Manual run
    try:
        test_security_path_traversal()
        test_security_allowed_path()
        test_memory_ingestion_and_retrieval()
        print("\n🎉 All Manual Evals Passed!")
    except Exception as e:
        print(f"\n❌ Eval Failed: {e}")
