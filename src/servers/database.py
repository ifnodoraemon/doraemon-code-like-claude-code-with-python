import json
import logging
import os
import re
import sqlite3

from src.core.logger import configure_root_logger
from src.core.security.security import validate_path

_DB_WRITE_MAX_ROWS = int(os.environ.get("AGENT_DB_WRITE_MAX_ROWS", "1000"))

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    # Security check: ensure path is valid
    valid_path = validate_path(db_path)

    conn = sqlite3.connect(valid_path)
    conn.row_factory = sqlite3.Row
    return conn


def _contains_multiple_statements(query: str) -> bool:
    """Check if a query contains multiple SQL statements (semicolon outside strings/comments)."""
    # Remove SQL comments
    cleaned = re.sub(r"--[^\n]*", "", query)  # single-line comments
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)  # block comments
    # Remove string literals
    cleaned = re.sub(r"'[^']*'", "", cleaned)
    cleaned = re.sub(r'"[^"]*"', "", cleaned)
    # Check for semicolons (ignoring trailing whitespace after last statement)
    parts = [p.strip() for p in cleaned.split(";") if p.strip()]
    return len(parts) > 1


def db_read_query(query: str, db_path: str, params: list | None = None) -> str:
    """
    Execute a SELECT query on a SQLite database.

    Args:
        query: The SQL SELECT statement (use ? placeholders for parameters)
        db_path: Path to the SQLite database file
        params: Optional list of query parameters for ? placeholders

    Returns:
        JSON string of the results or error message
    """
    if not query.strip().lower().startswith("select"):
        return "Error: Only SELECT queries are allowed in db_read_query. Use db_write_query for modifications."

    if _contains_multiple_statements(query):
        return "Error: Multiple SQL statements are not allowed. Please execute one statement at a time."

    conn = None
    try:
        conn = _get_connection(db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute(query, params or [])
            rows = cursor.fetchall()

        # Convert rows to dicts
        results = [dict(row) for row in rows]

        if not results:
            return "No results found."

        return json.dumps(results, indent=2, default=str)

    except Exception as e:
        logger.error("Database error: %s", e)
        return f"Database error: {str(e)}"
    finally:
        if conn:
            conn.close()


def db_write_query(query: str, db_path: str, params: list | None = None) -> str:
    """
    Execute an INSERT, UPDATE, DELETE, or CREATE query on a SQLite database.

    Args:
        query: The SQL statement (use ? placeholders for parameters)
        db_path: Path to the SQLite database file
        params: Optional list of query parameters for ? placeholders

    Returns:
        Success message or error
    """
    try:
        validate_path(db_path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    conn = None
    try:
        first_keyword = query.strip().split()[0].upper() if query.strip() else ""
        destructive_keywords = {"DROP", "TRUNCATE", "ALTER"}
        if first_keyword in destructive_keywords:
            return (
                f"Error: '{first_keyword}' operations are blocked for safety. "
                "These operations can cause irreversible data loss. "
                "Use a direct database tool if you need to modify schema."
            )

        if _contains_multiple_statements(query):
            return "Error: Multiple SQL statements are not allowed. Please execute one statement at a time."

        if first_keyword in ("UPDATE", "DELETE") or re.search(r'\b(UPDATE|DELETE)\b', query, re.IGNORECASE):
            upper_q = query.upper()
            update_delete_part = upper_q
            cte_match = re.search(r'\bWITH\b', upper_q)
            if cte_match:
                last_update_delete = None
                for m in re.finditer(r'\b(UPDATE|DELETE)\b', upper_q):
                    last_update_delete = m.end()
                if last_update_delete is not None:
                    update_delete_part = upper_q[last_update_delete:]
            stripped = re.sub(r"--[^\n]*", "", update_delete_part)
            stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
            stripped = re.sub(r"'[^']*'", "", stripped)
            stripped = re.sub(r'"[^"]*"', "", stripped)
            if "WHERE" not in stripped and "LIMIT" not in stripped:
                return (
                    "Error: UPDATE/DELETE without WHERE or LIMIT clause is blocked for safety. "
                    "Add a WHERE clause or LIMIT to restrict affected rows."
                )

        conn = _get_connection(db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute(query, params or [])
            conn.commit()
            row_count = cursor.rowcount

        if row_count > _DB_WRITE_MAX_ROWS:
            return (
                f"Warning: Query affected {row_count} rows (limit: {_DB_WRITE_MAX_ROWS}). "
                "Consider adding a more restrictive WHERE clause."
            )

        return f"Query executed successfully. Rows affected: {row_count}"

    except Exception as e:
        logger.error("Database error: %s", e)
        return f"Database error: {str(e)}"
    finally:
        if conn:
            conn.close()


def db_list_tables(db_path: str) -> str:
    """
    List all tables in the SQLite database.

    Args:
        db_path: Path to the SQLite database file
    """
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    return str(db_read_query(query, db_path))


def db_describe_table(table_name: str, db_path: str) -> str:
    """
    Get the schema of a specific table.

    Args:
        table_name: Name of the table
        db_path: Path to the SQLite database file
    """
    import re

    # Validate table name to prevent injection in PRAGMA
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
        return f"Error: Invalid table name '{table_name}'"

    # PRAGMA does not support parameterized queries; the regex validation above
    # ensures table_name is safe for string interpolation.
    query = f"PRAGMA table_info({table_name});"
    conn = None
    try:
        conn = _get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            return f"Table '{table_name}' not found or empty schema."

        # Format output nicely
        output = [f"Schema for {table_name}:"]
        for row in rows:
            # cid, name, type, notnull, dflt_value, pk
            output.append(f"- {row['name']} ({row['type']}) {'PK' if row['pk'] else ''}")

        return "\n".join(output)
    except Exception as e:
        return f"Error describing table: {e}"
    finally:
        if conn:
            conn.close()

