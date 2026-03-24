import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.core.config import load_config
from src.core.logger import configure_root_logger
from src.core.memory_layers import LayeredMemory, MemoryLayer
from src.core.paths import chroma_dir, layered_memory_dir, persona_path, state_dir
from src.services.reranking import NoteReranker

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from the Gemini/httpx SDK
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize FastMCP Server
mcp = FastMCP("AgentMemory")

# Lazy-initialized globals
_initialized = False
embedding_fn = None
client = None
collection = None
COLLECTION_NAME = None
NOTE_DELIMITER = "---"


def _persist_dir() -> str:
    """Resolve the current project's vector store directory."""
    return str(chroma_dir())


def _memory_file() -> Path:
    """Resolve the current project's persona file."""
    return persona_path()


def _structured_memory() -> LayeredMemory:
    """Structured memory facade that reuses the existing persona file."""
    return LayeredMemory(
        storage_dir=layered_memory_dir() / "kv",
        project_id=Path.cwd().name,
        user_id="default",
        layer_paths={MemoryLayer.USER: _memory_file()},
        simple_layers={MemoryLayer.USER},
    )


def _notes_root() -> Path:
    """Resolve the persisted note store root."""
    return layered_memory_dir() / "notes"


def _notes_dir(collection_name: str) -> Path:
    """Resolve the directory for a project's notes."""
    return _notes_root() / collection_name


def _slugify(value: str) -> str:
    """Create a filesystem-safe note slug."""
    collapsed = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return collapsed or "note"


def _note_path(collection_name: str, title: str) -> Path:
    """Resolve a note path from project and title."""
    return _notes_dir(collection_name) / f"{_slugify(title)}.md"


def _load_note_by_title(collection_name: str, title: str) -> dict[str, Any] | None:
    """Load a note by its exact title."""
    expected_title = title.strip()
    for note in _load_project_notes(collection_name):
        note_title = str(note.get("title") or "").strip()
        if note_title == expected_title:
            return note
    return None


def _resolve_note_path(collection_name: str, title: str, *, for_write: bool = False) -> Path:
    """Resolve the canonical path for a note title, avoiding slug collisions."""
    existing_note = _load_note_by_title(collection_name, title)
    if existing_note is not None:
        return Path(existing_note["path"])

    base_path = _note_path(collection_name, title)
    if not for_write or not base_path.exists():
        return base_path

    title_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
    candidate = base_path.with_name(f"{base_path.stem}-{title_hash}{base_path.suffix}")
    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = base_path.with_name(
            f"{base_path.stem}-{title_hash}-{counter}{base_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def _note_id(collection_name: str, title: str, content: str) -> str:
    """Build the vector index ID for a note."""
    return f"{collection_name}_{title}_{hashlib.sha256(content.encode()).hexdigest()[:12]}"


def _serialize_note(title: str, content: str, collection_name: str, tags: list[str]) -> str:
    """Serialize a note to Markdown with JSON front matter."""
    metadata = {
        "title": title,
        "project": collection_name,
        "tags": tags,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    return (
        f"{NOTE_DELIMITER}\n"
        f"{json.dumps(metadata, ensure_ascii=False)}\n"
        f"{NOTE_DELIMITER}\n\n"
        f"{content}"
    )


def _deserialize_note(path: Path) -> dict[str, Any] | None:
    """Parse a persisted note file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    if not raw.startswith(f"{NOTE_DELIMITER}\n"):
        return {
            "title": path.stem,
            "project": path.parent.name,
            "tags": [],
            "content": raw,
            "path": path,
        }

    parts = raw.split(f"\n{NOTE_DELIMITER}\n", 1)
    if len(parts) != 2:
        return None
    metadata_line = parts[0][len(NOTE_DELIMITER) + 1 :]
    content = parts[1].lstrip("\n")

    try:
        metadata = json.loads(metadata_line)
    except json.JSONDecodeError:
        return None

    metadata["content"] = content
    metadata["path"] = path
    metadata["tags"] = metadata.get("tags") or []
    return metadata


def _write_note_file(title: str, content: str, collection_name: str, tags: list[str]) -> Path:
    """Persist a note to the Markdown note store."""
    note_path = _resolve_note_path(collection_name, title, for_write=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        _serialize_note(title=title, content=content, collection_name=collection_name, tags=tags),
        encoding="utf-8",
    )
    return note_path


def _load_project_notes(collection_name: str) -> list[dict[str, Any]]:
    """Load all persisted notes for a project."""
    notes_dir = _notes_dir(collection_name)
    if not notes_dir.exists():
        return []

    notes: list[dict[str, Any]] = []
    for path in sorted(notes_dir.glob("*.md")):
        note = _deserialize_note(path)
        if note is not None:
            notes.append(note)
    return notes


def get_note_file_path(title: str, collection_name: str = "default") -> str:
    """Return the filesystem path for a note."""
    return str(_resolve_note_path(collection_name, title, for_write=True))


@mcp.tool()
def get_note(title: str, collection_name: str = "default") -> str:
    """Read a single note by title."""
    note = _load_note_by_title(collection_name, title)
    if note is None:
        return f"Note '{title}' not found in project {collection_name}."

    rendered_title = note.get("title") or title
    return f"[标题: {rendered_title}]\n{note.get('content', '')}"


@mcp.tool()
def export_notes(collection_name: str = "default", export_format: str = "json") -> str:
    """Export notes for a project as JSON or Markdown."""
    notes = _load_project_notes(collection_name)
    if not notes:
        return f"No notes found in project {collection_name}."

    export_format = export_format.lower()
    if export_format == "json":
        export_payload = [
            {
                "title": note.get("title"),
                "content": note.get("content"),
                "tags": note.get("tags", []),
                "project": note.get("project", collection_name),
            }
            for note in notes
        ]
        return json.dumps(export_payload, indent=2, ensure_ascii=False)

    if export_format == "markdown":
        chunks = []
        for note in notes:
            chunks.append(f"# {note.get('title') or 'Untitled'}\n\n{note.get('content', '')}")
        return f"\n\n{NOTE_DELIMITER}\n\n".join(chunks)

    return f"Unsupported export format: {export_format}"


def _sync_note_to_vector(title: str, content: str, collection_name: str, tags: list[str]) -> None:
    """Best-effort vector sync for a note."""
    if collection is None:
        return

    metadata = {"title": title, "tags": ",".join(tags), "project": collection_name}

    try:
        if hasattr(collection, "get") and hasattr(collection, "delete"):
            existing = collection.get(
                where={"$and": [{"title": title}, {"project": collection_name}]}
            )
            existing_ids = existing.get("ids") if existing else []
            if existing_ids:
                collection.delete(ids=existing_ids)

        collection.add(
            documents=[content],
            metadatas=[metadata],  # type: ignore[arg-type]
            ids=[_note_id(collection_name, title, content)],
        )
    except Exception as e:
        logger.warning("Vector sync failed for note '%s': %s", title, e)


def _keyword_search_notes(
    query: str,
    collection_name: str,
    n_results: int,
) -> list[dict[str, Any]]:
    """Fallback lexical search over persisted note files."""
    query_terms = [term for term in query.lower().split() if term]
    if not query_terms:
        return _load_project_notes(collection_name)[:n_results]

    ranked: list[tuple[int, dict[str, Any]]] = []
    for note in _load_project_notes(collection_name):
        haystack = " ".join(
            [
                str(note.get("title", "")),
                str(note.get("content", "")),
                " ".join(note.get("tags", [])),
            ]
        ).lower()
        score = sum(haystack.count(term) for term in query_terms)
        if score > 0:
            ranked.append((score, note))

    ranked.sort(key=lambda item: (-item[0], str(item[1].get("title", "")).lower()))
    return [note for _score, note in ranked[:n_results]]


def _vector_search_notes(query: str, collection_name: str, n_results: int) -> list[dict[str, Any]]:
    """Collect vector search candidates if the vector index is available."""
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"project": collection_name},
        )
    except Exception as e:
        logger.warning("Vector search failed for project '%s': %s", collection_name, e)
        return []

    docs = results.get("documents")
    metas = results.get("metadatas")
    if not docs or not docs[0] or not metas or not metas[0]:
        return []

    candidates: list[dict[str, Any]] = []
    for rank, (doc, meta) in enumerate(zip(docs[0], metas[0], strict=True), start=1):
        title = meta.get("title") or "Untitled"
        tags_value = meta.get("tags", "")
        tags = [tag for tag in str(tags_value).split(",") if tag]
        file_note = _load_note_by_title(collection_name, title)
        candidates.append(
            {
                "title": title,
                "content": file_note.get("content", doc) if file_note else doc,
                "tags": file_note.get("tags", tags) if file_note else tags,
                "project": collection_name,
                "source": "vector+file" if file_note else "vector",
                "base_score": float(max(n_results - rank + 1, 1)),
            }
        )
    return candidates


def _merge_note_candidates(*candidate_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate note candidates while preserving first-seen order."""
    merged: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for group in candidate_groups:
        for candidate in group:
            title = str(candidate.get("title", "")).strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            merged.append(candidate)
    return merged


def _ensure_initialized():
    """Lazily initialize ChromaDB and embeddings on first use."""
    global _initialized, embedding_fn, client, collection, COLLECTION_NAME
    if collection is not None:
        _initialized = True
        return
    if _initialized:
        return
    _initialized = True

    state_dir().mkdir(parents=True, exist_ok=True)

    try:
        import chromadb

        from src.services.embeddings import RemoteEmbeddingFunction

        embedding_fn = RemoteEmbeddingFunction()
        client = chromadb.PersistentClient(path=_persist_dir())

        provider = embedding_fn.provider
        COLLECTION_NAME = f"agent_notes_{provider}"

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collection: {e}")
        collection = None


@mcp.tool()
def save_note(
    title: str, content: str, collection_name: str = "default", tags: list[str] | None = None
) -> str:
    """Save a note to persistent Markdown memory and sync vector index if available."""
    _ensure_initialized()

    tags = tags or []

    if len(content) > 100000:
        return (
            f"Error: Content too long ({len(content)} chars). Maximum allowed is 100000 characters."
        )

    try:
        _write_note_file(title=title, content=content, collection_name=collection_name, tags=tags)
        _sync_note_to_vector(
            title=title, content=content, collection_name=collection_name, tags=tags
        )
        logger.info(f"Saved note '{title}' to project '{collection_name}'")
        return f"Note '{title}' saved to project {collection_name}."
    except Exception as e:
        logger.error(f"Failed to save note '{title}': {e}")
        return f"Failed to save note: {e}"


@mcp.tool()
def search_notes(query: str, collection_name: str = "default", n_results: int = 3) -> str:
    """Search for notes in the long-term memory."""
    _ensure_initialized()

    try:
        vector_candidates = _vector_search_notes(query, collection_name, n_results)
        keyword_candidates = _keyword_search_notes(
            query, collection_name, max(n_results * 3, n_results)
        )
        candidates = _merge_note_candidates(vector_candidates, keyword_candidates)

        if not candidates:
            return f"Notes not found in project {collection_name}."

        config = load_config(validate=False)
        rerank_model = config.get("rerank_model")
        if rerank_model:
            candidates = NoteReranker(rerank_model).rerank(
                query=query, candidates=candidates, limit=n_results
            )
        else:
            candidates = candidates[:n_results]

        output = [
            f"[标题: {candidate.get('title') or 'Untitled'}]\n{candidate.get('content', '')}"
            for candidate in candidates
        ]
        return "\n---\n".join(output)
    except Exception as e:
        return f"Search failed: {e}"


@mcp.tool()
def delete_note(title: str, collection_name: str = "default") -> str:
    """Delete a note from the long-term memory by title."""
    _ensure_initialized()

    try:
        note = _load_note_by_title(collection_name, title)
        deleted = False
        if note is not None:
            Path(note["path"]).unlink()
            deleted = True

        if collection is not None:
            results = collection.get(
                where={"$and": [{"title": title}, {"project": collection_name}]}
            )
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
                deleted = True

        if not deleted:
            return f"Note '{title}' not found in project {collection_name}."
        logger.info(f"Deleted note '{title}' from project '{collection_name}'")
        return f"Note '{title}' deleted from project {collection_name}."
    except Exception as e:
        logger.error(f"Failed to delete note '{title}': {e}")
        return f"Failed to delete note: {e}"


@mcp.tool()
def list_notes(collection_name: str = "default", limit: int = 20) -> str:
    """List all notes in a collection."""
    _ensure_initialized()

    try:
        output = [f"=== Notes in {collection_name} ==="]

        file_notes = _load_project_notes(collection_name)
        if file_notes:
            for i, note in enumerate(file_notes[:limit], start=1):
                title = note.get("title", "Untitled")
                tags = ", ".join(note.get("tags", []))
                doc = note.get("content", "")
                preview = (doc[:80] + "...") if doc and len(doc) > 80 else (doc or "")
                output.append(f"[{i}] {title}")
                if tags:
                    output.append(f"    Tags: {tags}")
                output.append(f"    {preview}")
            return "\n".join(output)

        if collection is None:
            return f"No notes found in project {collection_name}."

        results = collection.get(where={"project": collection_name}, limit=limit)
        if not results or not results.get("ids"):
            return f"No notes found in project {collection_name}."

        metas = results.get("metadatas", [])
        docs = results.get("documents", [])
        for i, (meta, doc) in enumerate(zip(metas, docs, strict=False), start=1):
            title = meta.get("title", "Untitled") if meta else "Untitled"
            tags = meta.get("tags", "") if meta else ""
            preview = (doc[:80] + "...") if doc and len(doc) > 80 else (doc or "")
            output.append(f"[{i}] {title}")
            if tags:
                output.append(f"    Tags: {tags}")
            output.append(f"    {preview}")

        return "\n".join(output)
    except Exception as e:
        logger.error(f"Failed to list notes: {e}")
        return f"Failed to list notes: {e}"


@mcp.tool()
def update_user_persona(key: str, value: str) -> str:
    """Update user persona (preferences, job, etc.)."""
    memory_file = _memory_file()
    memory_file.parent.mkdir(parents=True, exist_ok=True)

    memory = _structured_memory()
    memory.set(key=key, value=value, layer=MemoryLayer.USER)

    return f"Remembered about you: {key} = {value}"


@mcp.tool()
def get_user_persona() -> str:
    """Read current user persona."""
    memory_file = _memory_file()
    if not memory_file.exists():
        return "No user persona yet."

    memory = _structured_memory()
    entries = memory.list_layer(MemoryLayer.USER)
    data = {entry.key: entry.value for entry in entries}
    return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
