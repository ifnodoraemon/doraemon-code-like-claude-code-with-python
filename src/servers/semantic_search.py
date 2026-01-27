"""
Semantic Code Search MCP Server

Provides semantic code search using Vector Database (ChromaDB) and Remote Embeddings.
Uses Google Gemini or OpenAI for embeddings, avoiding local heavy models.
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import chromadb
from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path
from src.services.embeddings import RemoteEmbeddingFunction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonSemanticSearch")

# ========================================
# Configuration
# ========================================

PERSIST_DIR = ".doraemon/chroma_db"
COLLECTION_NAME = "codebase_index_remote"

# Ensure directory exists
os.makedirs(".doraemon", exist_ok=True)

# Initialize ChromaDB
try:
    embedding_fn = RemoteEmbeddingFunction()
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )
    HAS_DB = True
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    HAS_DB = False
    collection = None


@dataclass
class SearchConfig:
    """Configuration for semantic search."""

    # File patterns to index
    include_patterns: list[str] = field(
        default_factory=lambda: [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.go", "*.rs",
            "*.c", "*.cpp", "*.h", "*.rb", "*.php", "*.swift", "*.kt", "*.md",
            "*.txt", "*.json", "*.yaml", "*.yml"
        ]
    )

    # Directories to exclude
    exclude_dirs: list[str] = field(
        default_factory=lambda: [
            "node_modules", ".git", "__pycache__", ".venv", "venv", "dist",
            "build", ".next", ".nuxt", "target", ".idea", ".vscode", "coverage",
            ".pytest_cache"
        ]
    )

    chunk_size: int = 50  # lines per chunk
    chunk_overlap: int = 10  # overlap between chunks
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
    signature: str | None = None  # Function/class signature

    @property
    def id(self) -> str:
        """Unique identifier for this chunk."""
        # Include hash of content to detect changes
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.file_path}:{self.start_line}-{self.end_line}:{content_hash}"


def _get_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
        ".md": "markdown", ".json": "json"
    }
    return mapping.get(ext, "text")


def _chunk_file(file_path: str, config: SearchConfig = DEFAULT_CONFIG) -> list[CodeChunk]:
    """Split a file into chunks for indexing."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return []

    if not lines:
        return []

    language = _get_language(file_path)
    total_lines = len(lines)
    chunks = []
    
    start = 0
    while start < total_lines:
        end = min(start + config.chunk_size, total_lines)
        chunk_lines = lines[start:end]
        content = "".join(chunk_lines)

        if content.strip():
            chunk = CodeChunk(
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                content=content,
                language=language,
                signature=None # Simplified for now, can perform regex extraction if needed
            )
            chunks.append(chunk)

        start = end - config.chunk_overlap
        if start >= total_lines - config.chunk_overlap:
            break

    return chunks


def _should_include_file(file_path: str, config: SearchConfig = DEFAULT_CONFIG) -> bool:
    """Check if a file should be included in the index."""
    path = Path(file_path)
    for exclude in config.exclude_dirs:
        if exclude in path.parts:
            return False
            
    import fnmatch
    for pattern in config.include_patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return True
            
    return False


# ========================================
# Search Tools
# ========================================

@mcp.tool()
def semantic_search(
    query: str,
    path: str = ".",  # Ignored for index search usually, but can be used to filter
    max_results: int = 15,
) -> str:
    """
    Search code using natural language (Vector Search).
    
    Uses remote embeddings to find semantically similar code in the indexed codebase.
    Requires running 'index_codebase' first.
    """
    if not HAS_DB or collection is None:
        return "Error: Database not initialized. Please configure remote embedding keys."

    if collection.count() == 0:
        return "Index is empty. Please run 'index_codebase' first."

    # Validate path for filtering
    try:
        resolved_path = validate_path(path)
        rel_search_path = os.path.relpath(resolved_path, os.getcwd())
    except Exception:
        rel_search_path = "."

    try:
        # Query ChromaDB (it automatically embeds the query using our RemoteEmbeddingFunction)
        results = collection.query(
            query_texts=[query],
            n_results=max_results,
            # We could add a 'where_document' filter for path if needed, 
            # but usually semantic search is global. 
            # Filtering by metadata path would require storing path splits in metadata.
        )
        
        if not results["documents"] or not results["documents"][0]:
            return f"No relevant results found for: {query}"
            
        output_lines = [f'Found {len(results["documents"][0])} relevant result(s) for: "{query}"\n']
        
        for i, (doc, meta, score) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1):
            file_path = meta.get("file_path", "unknown")
            
            # Simple path filtering
            if rel_search_path != "." and not file_path.startswith(rel_search_path):
                continue
                
            line_range = f"{meta.get('start_line')}-{meta.get('end_line')}"
            
            # Score in Chroma is distance (lower is better), but usually we want similarity.
            # Convert to a readable string or just show as is.
            output_lines.append(f"{'─' * 60}")
            output_lines.append(f"[{i}] {file_path}:{line_range} (distance: {score:.4f})")
            
            preview_lines = doc.split("\n")[:10]
            preview = "\n".join(f"    {line}" for line in preview_lines)
            if len(doc.split("\n")) > 10:
                preview += "\n    ..."
            output_lines.append(preview)
            output_lines.append("")
            
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"Search failed: {e}"


@mcp.tool()
def index_codebase(path: str = ".") -> str:
    """
    Build the semantic search index for a codebase.
    
    Scans files, chunks them, and uploads embeddings to the local vector DB.
    WARNING: This consumes API credits for embeddings (Google/OpenAI).
    """
    if not HAS_DB or collection is None:
        return "Error: Database not initialized."
        
    try:
        resolved_path = validate_path(path)
    except Exception as e:
        return f"Error: {e}"

    if not os.path.exists(resolved_path):
        return f"Error: Path not found: {path}"

    # Collect files
    files_to_index = []
    for root, dirs, files in os.walk(resolved_path):
        dirs[:] = [d for d in dirs if d not in DEFAULT_CONFIG.exclude_dirs]
        for file in files:
            file_path = os.path.join(root, file)
            if _should_include_file(file_path):
                files_to_index.append(file_path)

    if not files_to_index:
        return "No files found to index."

    # Chunk files
    all_chunks = []
    for file_path in files_to_index:
        # Use relative path for storage
        rel_path = os.path.relpath(file_path, os.getcwd()) 
        chunks = _chunk_file(file_path)
        for chunk in chunks:
            # We update the chunk object to store rel_path for the DB
            chunk.file_path = rel_path 
            all_chunks.append(chunk)

    if not all_chunks:
        return "No code chunks generated."
        
    # Clear existing index? Or upsert?
    # For simplicity, we'll upsert. But stale chunks might remain.
    # Ideally: collection.delete() might be needed for a clean rebuild.
    # But let's just upsert for now.
    
    # Process in batches to respect API limits
    BATCH_SIZE = 50 
    total_added = 0
    
    import time
    
    logger.info(f"Indexing {len(all_chunks)} chunks...")
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        
        ids = [c.id for c in batch]
        documents = [c.content for c in batch]
        metadatas = [{
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "language": c.language
        } for c in batch]
        
        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            total_added += len(batch)
            # Small sleep to be nice to rate limits
            time.sleep(0.5) 
        except Exception as e:
            logger.error(f"Failed to index batch {i}: {e}")
            
    return f"Indexed {total_added} chunks from {len(files_to_index)} files."


if __name__ == "__main__":
    mcp.run()
