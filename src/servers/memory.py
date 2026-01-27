import logging
import os
import json
from typing import Any, List
from src.services.embeddings import RemoteEmbeddingFunction

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from the Gemini/httpx SDK
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize FastMCP Server
mcp = FastMCP("DoraemonMemory")

# --------------------------
# Core Data Structures
# --------------------------
PERSIST_DIR = ".doraemon/chroma_db"
MEMORY_FILE = ".doraemon/memory.json"

# Ensure directory exists
os.makedirs(".doraemon", exist_ok=True)


# Initialize ChromaDB with Remote Embeddings
embedding_fn = RemoteEmbeddingFunction()
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Note: If the collection was created with a different dimension (e.g. 384 from MiniLM), 
# switching to OpenAI (1536) or Gemini (768) will cause errors.
# We handle this by using a dynamic collection name based on the provider.
PROVIDER = embedding_fn.provider
COLLECTION_NAME = f"doraemon_notes_{PROVIDER}"

try:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=embedding_fn
    )
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB collection: {e}")
    collection = None


@mcp.tool()
def save_note(
    title: str, content: str, collection_name: str = "default", tags: list[str] | None = None
) -> str:
    """Save a note to the long-term memory (Vector DB)."""
    if collection is None:
        return "Memory system is not initialized."
        
    if tags is None:
        tags = []
        
    try:
        # We use a single shared collection for now with metadata filtering if needed, 
        # or we could create potential sub-collections.
        # To keep it simple and avoid model dimension mismatches across collections, we use the main one.
        
        note_id = f"{collection_name}_{title}_{hash(content)}"
        
        # Add metadata for filtering
        metadatas = [{"title": title, "tags": ",".join(tags), "project": collection_name}]
        
        collection.add(
            documents=[content],
            metadatas=metadatas,
            ids=[note_id]
        )
        logger.info(f"Saved note '{title}' to project '{collection_name}'")
        return f"笔记 '{title}' 已保存到项目 {collection_name}。"
    except Exception as e:
        logger.error(f"Failed to save note '{title}': {e}")
        return f"保存笔记失败: {e}"


@mcp.tool()
def search_notes(query: str, collection_name: str = "default", n_results: int = 3) -> str:
    """Search for notes in the long-term memory."""
    if collection is None:
        return "Memory system is not initialized."

    try:
        # Filter by project/collection_name in metadata
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"project": collection_name}
        )

        if not results["documents"] or not results["documents"][0]:
            return f"项目 {collection_name} 中未找到相关笔记。"

        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append(f"[标题: {meta.get('title', 'Untitled')}]\n{doc}")

        return "\n---\n".join(output)
    except Exception as e:
        return f"搜索失败: {e}"


@mcp.tool()
def update_user_persona(key: str, value: str) -> str:
    """Update user persona (preferences, job, etc.)."""
    data = {}
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    data[key] = value

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return f"已记住关于你的事实: {key} = {value}"


@mcp.tool()
def get_user_persona() -> str:
    """Read current user persona."""
    if not os.path.exists(MEMORY_FILE):
        return "暂无用户画像。"
    with open(MEMORY_FILE) as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
