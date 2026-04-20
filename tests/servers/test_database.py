"""Tests for src/servers/database.py"""

import json
import os
import sqlite3

import pytest

from src.servers.database import (
    _contains_multiple_statements,
    db_describe_table,
    db_list_tables,
    db_read_query,
    db_write_query,
)


def _workspace_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def db_path(tmp_path):
    path = _workspace_db_path(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')")
    conn.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT, user_id INTEGER)")
    conn.execute("INSERT INTO posts VALUES (1, 'Hello', 1)")
    conn.commit()
    conn.close()
    return path


class TestContainsMultipleStatements:
    def test_single_statement(self):
        assert _contains_multiple_statements("SELECT * FROM users") is False

    def test_multiple_statements(self):
        assert _contains_multiple_statements("SELECT 1; SELECT 2") is True

    def test_semicolon_in_string(self):
        assert _contains_multiple_statements("SELECT 'a;b'") is False

    def test_comment_stripped(self):
        assert _contains_multiple_statements("SELECT 1 -- comment; SELECT 2") is False

    def test_block_comment(self):
        assert _contains_multiple_statements("SELECT 1 /* ; */ FROM t") is False

    def test_trailing_semicolon(self):
        assert _contains_multiple_statements("SELECT 1;") is False

    def test_whitespace_after_last(self):
        assert _contains_multiple_statements("SELECT 1;  ") is False


class TestDbReadQuery:
    def test_select_all(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_read_query("SELECT * FROM users", db_path)
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"

    def test_select_with_params(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_read_query("SELECT * FROM users WHERE name = ?", db_path, params=["Bob"])
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name"] == "Bob"

    def test_select_empty(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_read_query("SELECT * FROM users WHERE id = 999", db_path)
        assert result == "No results found."

    def test_non_select_rejected(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_read_query("DELETE FROM users", db_path)
        assert "Error" in result
        assert "SELECT" in result

    def test_multiple_statements_rejected(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_read_query("SELECT 1; SELECT 2", db_path)
        assert "Error" in result
        assert "Multiple" in result

    def test_invalid_sql(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_read_query("SELECT * FROM nonexistent_table", db_path)
        assert "Database error" in result


class TestDbWriteQuery:
    def test_insert(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query(
            "INSERT INTO users VALUES (3, 'Charlie', 'charlie@ex.com')", db_path
        )
        assert "successfully" in result
        assert "Rows affected" in result

    def test_update_with_params(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query(
            "UPDATE users SET name = ? WHERE id = ?", db_path, params=["Alicia", 1]
        )
        assert "successfully" in result

    def test_create_table(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", db_path)
        assert "successfully" in result

    def test_drop_blocked(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query("DROP TABLE users", db_path)
        assert "blocked" in result
        assert "DROP" in result

    def test_truncate_blocked(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query("TRUNCATE TABLE users", db_path)
        assert "blocked" in result

    def test_alter_blocked(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query("ALTER TABLE users ADD COLUMN age INT", db_path)
        assert "blocked" in result

    def test_multiple_statements_rejected(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query(
            "INSERT INTO users VALUES (3,'x','y'); INSERT INTO users VALUES (4,'a','b')", db_path
        )
        assert "Multiple" in result

    def test_invalid_sql(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_write_query("INSER INTO users VALUES", db_path)
        assert "Database error" in result


class TestDbListTables:
    def test_list(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_list_tables(db_path)
        data = json.loads(result)
        names = [t["name"] for t in data]
        assert "users" in names
        assert "posts" in names


class TestDbDescribeTable:
    def test_describe(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_describe_table("users", db_path)
        assert "Schema for users" in result
        assert "id" in result
        assert "name" in result
        assert "email" in result

    def test_describe_invalid_name(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_describe_table("bad-name; DROP TABLE", db_path)
        assert "Error" in result
        assert "Invalid table name" in result

    def test_describe_nonexistent_table(self, db_path, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
        result = db_describe_table("nonexistent", db_path)
        assert "not found" in result or "empty" in result
