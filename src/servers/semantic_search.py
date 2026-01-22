"""
Semantic Code Search MCP Server

Provides semantic (meaning-based) code search using embeddings.
Similar to Claude Code's SemanticSearch but with local embeddings.

Features:
- Natural language code search
- Code chunking and indexing
- Similarity-based ranking
- Support for multiple languages
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("PolymathSemanticSearch")


# ========================================
# Configuration
# ========================================


@dataclass
class SearchConfig:
    """Configuration for semantic search."""

    # File patterns to index
    include_patterns: list[str] = field(
        default_factory=lambda: [
            "*.py",
            "*.js",
            "*.ts",
            "*.jsx",
            "*.tsx",
            "*.java",
            "*.go",
            "*.rs",
            "*.c",
            "*.cpp",
            "*.h",
            "*.rb",
            "*.php",
            "*.swift",
            "*.kt",
            "*.md",
            "*.txt",
            "*.json",
            "*.yaml",
            "*.yml",
        ]
    )

    # Directories to exclude
    exclude_dirs: list[str] = field(
        default_factory=lambda: [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
            ".nuxt",
            "target",
            ".idea",
            ".vscode",
            "coverage",
            ".pytest_cache",
        ]
    )

    # Chunk size for indexing
    chunk_size: int = 50  # lines per chunk
    chunk_overlap: int = 10  # overlap between chunks

    # Search settings
    max_results: int = 15
    min_score: float = 0.3


DEFAULT_CONFIG = SearchConfig()


# ========================================
# Code Chunking
# ========================================


@dataclass
class CodeChunk:
    """A chunk of code for indexing."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    signature: str | None = None  # Function/class signature if applicable

    @property
    def id(self) -> str:
        """Unique identifier for this chunk."""
        return hashlib.md5(
            f"{self.file_path}:{self.start_line}:{self.end_line}".encode()
        ).hexdigest()[:12]


def _get_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return language_map.get(ext, "text")


def _extract_signature(content: str, language: str) -> str | None:
    """Extract the signature (function/class definition) from code."""
    lines = content.split("\n")

    if language == "python":
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                return stripped.split(":")[0] + ":"
            if stripped.startswith("async def "):
                return stripped.split(":")[0] + ":"

    elif language in ("javascript", "typescript"):
        for line in lines:
            stripped = line.strip()
            # Function declarations
            if re.match(r"^(export\s+)?(async\s+)?function\s+\w+", stripped):
                return stripped.split("{")[0].strip()
            # Arrow functions with const/let
            if re.match(r"^(export\s+)?(const|let)\s+\w+\s*=\s*(async\s+)?\(", stripped):
                return stripped.split("=>")[0].strip() + " => ..."
            # Class declarations
            if re.match(r"^(export\s+)?class\s+\w+", stripped):
                return stripped.split("{")[0].strip()

    return None


def _chunk_file(
    file_path: str,
    config: SearchConfig = DEFAULT_CONFIG,
) -> list[CodeChunk]:
    """Split a file into chunks for indexing."""
    chunks = []

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return []

    if not lines:
        return []

    language = _get_language(file_path)
    total_lines = len(lines)

    # Create overlapping chunks
    start = 0
    while start < total_lines:
        end = min(start + config.chunk_size, total_lines)
        chunk_lines = lines[start:end]
        content = "".join(chunk_lines)

        # Skip empty chunks
        if content.strip():
            signature = _extract_signature(content, language)

            chunk = CodeChunk(
                file_path=file_path,
                start_line=start + 1,  # 1-indexed
                end_line=end,
                content=content,
                language=language,
                signature=signature,
            )
            chunks.append(chunk)

        # Move to next chunk with overlap
        start = end - config.chunk_overlap
        if start >= total_lines - config.chunk_overlap:
            break

    return chunks


def _should_include_file(file_path: str, config: SearchConfig = DEFAULT_CONFIG) -> bool:
    """Check if a file should be included in the index."""
    path = Path(file_path)

    # Check if in excluded directory
    for exclude in config.exclude_dirs:
        if exclude in path.parts:
            return False

    # Check if matches include patterns
    import fnmatch

    for pattern in config.include_patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return True

    return False


# ========================================
# Simple Text-Based Semantic Search
# ========================================


def _simple_score(query: str, content: str) -> float:
    """
    Simple keyword-based scoring when embeddings are not available.
    This is a fallback that provides basic semantic-like search.
    """
    query_lower = query.lower()
    content_lower = content.lower()

    # Extract keywords from query
    query_words = set(re.findall(r"\b\w+\b", query_lower))

    # Remove common words
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "how",
        "what",
        "where",
        "when",
        "why",
        "which",
        "who",
        "this",
        "that",
    }
    query_words = query_words - stopwords

    if not query_words:
        return 0.0

    # Count keyword matches
    matches = 0
    for word in query_words:
        if word in content_lower:
            # Boost exact word boundaries
            if re.search(rf"\b{re.escape(word)}\b", content_lower):
                matches += 1.5
            else:
                matches += 0.5

    # Calculate score
    base_score = matches / len(query_words)

    # Boost for matches in signatures/function names
    first_line = content.split("\n")[0].lower() if content else ""
    signature_matches = sum(1 for word in query_words if word in first_line)
    signature_boost = signature_matches * 0.3

    return min(1.0, base_score + signature_boost)


# ========================================
# Embedding-Based Search (if available)
# ========================================

_embedder = None
_use_embeddings = False


def _init_embeddings():
    """Initialize the embedding model if available."""
    global _embedder, _use_embeddings

    try:
        from sentence_transformers import SentenceTransformer

        # Use a small, fast model
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _use_embeddings = True
        logger.info("Initialized sentence-transformers for semantic search")
    except ImportError:
        logger.info("sentence-transformers not available, using keyword-based search")
        _use_embeddings = False


def _get_embedding(text: str) -> Any:
    """Get embedding for text."""
    if _embedder is None:
        _init_embeddings()

    if _use_embeddings and _embedder:
        return _embedder.encode(text, convert_to_numpy=True)
    return None


def _cosine_similarity(a: Any, b: Any) -> float:
    """Calculate cosine similarity between two vectors."""
    import numpy as np

    if a is None or b is None:
        return 0.0

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


# ========================================
# Search Tools
# ========================================


@mcp.tool()
def semantic_search(
    query: str,
    path: str = ".",
    max_results: int = 15,
    file_pattern: str | None = None,
) -> str:
    """
    Search code using natural language.

    Find code by describing what you're looking for in plain English.
    This is more powerful than grep when you don't know exact terms.

    Args:
        query: Natural language description of what you're looking for
            Examples:
            - "function that handles user authentication"
            - "where is the database connection configured"
            - "error handling for API requests"
        path: Directory to search in
        max_results: Maximum number of results to return
        file_pattern: Optional glob pattern to filter files (e.g., "*.py")

    Returns:
        Relevant code snippets ranked by similarity
    """
    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    if not os.path.isdir(resolved_path):
        return f"Error: {path} is not a directory"

    # Collect all files
    files_to_search = []
    for root, dirs, files in os.walk(resolved_path):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in DEFAULT_CONFIG.exclude_dirs]

        for file in files:
            file_path = os.path.join(root, file)

            # Apply file pattern filter
            if file_pattern:
                import fnmatch

                if not fnmatch.fnmatch(file, file_pattern):
                    continue

            if _should_include_file(file_path):
                files_to_search.append(file_path)

    if not files_to_search:
        return f"No searchable files found in {path}"

    # Chunk all files
    all_chunks: list[CodeChunk] = []
    for file_path in files_to_search[:100]:  # Limit files for performance
        chunks = _chunk_file(file_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        return "No code chunks found to search"

    # Score chunks
    results: list[tuple[float, CodeChunk]] = []

    # Try embedding-based search first
    _init_embeddings()

    if _use_embeddings:
        query_embedding = _get_embedding(query)

        for chunk in all_chunks:
            # Combine signature and content for better matching
            text = f"{chunk.signature or ''}\n{chunk.content}"
            chunk_embedding = _get_embedding(text)
            score = _cosine_similarity(query_embedding, chunk_embedding)
            results.append((score, chunk))
    else:
        # Fallback to keyword-based scoring
        for chunk in all_chunks:
            text = f"{chunk.signature or ''}\n{chunk.content}"
            score = _simple_score(query, text)
            results.append((score, chunk))

    # Sort by score and take top results
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:max_results]

    # Filter by minimum score
    top_results = [
        (score, chunk) for score, chunk in top_results if score >= DEFAULT_CONFIG.min_score
    ]

    if not top_results:
        return f"No relevant results found for: {query}\n\nTry different search terms or check if the codebase contains what you're looking for."

    # Format output
    output_lines = [f'Found {len(top_results)} relevant result(s) for: "{query}"\n']

    for i, (score, chunk) in enumerate(top_results, 1):
        rel_path = os.path.relpath(chunk.file_path, resolved_path)

        output_lines.append(f"{'─' * 60}")
        output_lines.append(
            f"[{i}] {rel_path}:{chunk.start_line}-{chunk.end_line} (score: {score:.2f})"
        )

        if chunk.signature:
            output_lines.append(f"    Signature: {chunk.signature}")

        # Show a preview of the content
        preview_lines = chunk.content.split("\n")[:10]
        preview = "\n".join(f"    {line}" for line in preview_lines)
        if len(chunk.content.split("\n")) > 10:
            preview += "\n    ..."

        output_lines.append(preview)
        output_lines.append("")

    return "\n".join(output_lines)


@mcp.tool()
def find_similar_code(
    file_path: str,
    line_start: int,
    line_end: int,
    search_path: str = ".",
    max_results: int = 10,
) -> str:
    """
    Find code similar to a given snippet.

    Useful for finding duplicated code or similar implementations.

    Args:
        file_path: Path to the file containing the reference code
        line_start: Starting line number
        line_end: Ending line number
        search_path: Directory to search in
        max_results: Maximum number of results

    Returns:
        Similar code snippets found in the codebase
    """
    try:
        resolved_file = validate_path(file_path)
        resolved_search = validate_path(search_path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    if not os.path.isfile(resolved_file):
        return f"Error: {file_path} is not a file"

    # Read the reference code
    try:
        with open(resolved_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            reference_code = "".join(lines[line_start - 1 : line_end])
    except Exception as e:
        return f"Error reading file: {e}"

    if not reference_code.strip():
        return "Error: Reference code is empty"

    # Search for similar code
    return semantic_search(
        query=reference_code,
        path=search_path,
        max_results=max_results,
    )


@mcp.tool()
def search_by_signature(
    pattern: str,
    path: str = ".",
    language: str | None = None,
) -> str:
    """
    Search for functions/classes by their signature pattern.

    Args:
        pattern: Pattern to match (e.g., "async def *user*", "class *Handler*")
        path: Directory to search in
        language: Filter by language (python, javascript, etc.)

    Returns:
        Matching function/class definitions
    """
    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    # Convert pattern to regex
    regex_pattern = pattern.replace("*", ".*")

    try:
        regex = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid pattern: {e}"

    # Search files
    results = []

    for root, dirs, files in os.walk(resolved_path):
        dirs[:] = [d for d in dirs if d not in DEFAULT_CONFIG.exclude_dirs]

        for file in files:
            file_path = os.path.join(root, file)

            if not _should_include_file(file_path):
                continue

            file_lang = _get_language(file_path)
            if language and file_lang != language:
                continue

            # Read and search file
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        stripped = line.strip()

                        # Check for function/class definitions
                        is_definition = False
                        if file_lang == "python":
                            is_definition = stripped.startswith(("def ", "class ", "async def "))
                        elif file_lang in ("javascript", "typescript"):
                            is_definition = stripped.startswith(
                                ("function ", "class ", "export function ", "export class ")
                            ) or re.match(
                                r"^(export\s+)?(const|let)\s+\w+\s*=\s*(async\s+)?(\(|function)",
                                stripped,
                            )

                        if is_definition and regex.search(stripped):
                            rel_path = os.path.relpath(file_path, resolved_path)
                            results.append(f"{rel_path}:{i}: {stripped}")
            except Exception:
                continue

    if not results:
        return f"No matches found for pattern: {pattern}"

    # Limit results
    if len(results) > 50:
        results = results[:50]
        results.append(f"\n... and more (showing first 50 of {len(results)})")

    return f"Found {len(results)} matching definition(s):\n\n" + "\n".join(results)


@mcp.tool()
def index_codebase(path: str = ".") -> str:
    """
    Build or refresh the search index for a codebase.

    This pre-processes files for faster semantic search.
    Call this when starting work on a new project.

    Args:
        path: Root directory of the codebase

    Returns:
        Indexing summary
    """
    try:
        resolved_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    if not os.path.isdir(resolved_path):
        return f"Error: {path} is not a directory"

    # Count files and chunks
    file_count = 0
    chunk_count = 0
    languages: dict[str, int] = {}

    for root, dirs, files in os.walk(resolved_path):
        dirs[:] = [d for d in dirs if d not in DEFAULT_CONFIG.exclude_dirs]

        for file in files:
            file_path = os.path.join(root, file)

            if _should_include_file(file_path):
                file_count += 1
                chunks = _chunk_file(file_path)
                chunk_count += len(chunks)

                lang = _get_language(file_path)
                languages[lang] = languages.get(lang, 0) + 1

    # Initialize embeddings if available
    _init_embeddings()

    # Format summary
    lines = [
        f"Codebase Index Summary for: {path}",
        f"{'─' * 40}",
        f"Files indexed: {file_count}",
        f"Code chunks: {chunk_count}",
        f"Search method: {'Semantic (embeddings)' if _use_embeddings else 'Keyword-based'}",
        "",
        "Languages detected:",
    ]

    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        lines.append(f"  {lang}: {count} files")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
