import json
import logging
import os

import chromadb
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastMCP Server
mcp = FastMCP("PolymathMemory")

# --------------------------
# 核心数据结构
# --------------------------
PERSIST_DIR = ".polymath/chroma_db"
MEMORY_FILE = ".polymath/memory.json"

# 确保目录存在
os.makedirs(".polymath", exist_ok=True)

# 初始化 ChromaDB
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name="polymath_notes")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


@mcp.tool()
def save_note(
    title: str, content: str, collection_name: str = "default", tags: list[str] | None = None
) -> str:
    """保存一条笔记到特定项目的长期记忆库。"""
    if tags is None:
        tags = []
    try:
        # 动态获取 collection
        coll = client.get_or_create_collection(name=f"polymath_{collection_name}")
        note_id = f"{title}_{len(content)}"
        embedding = embedding_model.encode(content).tolist()

        coll.add(
            documents=[content],
            metadatas=[{"title": title, "tags": ",".join(tags)}],
            ids=[note_id],
            embeddings=[embedding],
        )
        logger.info(f"Saved note '{title}' to collection '{collection_name}'")
        return f"笔记 '{title}' 已保存到项目 {collection_name}。"
    except Exception as e:
        logger.error(f"Failed to save note '{title}': {e}")
        return f"保存笔记失败: {e}"


@mcp.tool()
def search_notes(query: str, collection_name: str = "default", n_results: int = 3) -> str:
    """从特定项目的长期记忆库中搜索相关笔记。"""
    coll = client.get_or_create_collection(name=f"polymath_{collection_name}")
    query_embedding = embedding_model.encode(query).tolist()
    results = coll.query(query_embeddings=[query_embedding], n_results=n_results)

    if not results["documents"][0]:
        return f"项目 {collection_name} 中未找到相关笔记。"

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0], strict=False):
        output.append(f"[标题: {meta['title']}]\n{doc}")

    return "\n---\n".join(output)


@mcp.tool()
def update_user_persona(key: str, value: str) -> str:
    """更新用户画像（例如：用户的偏好、职业、常用术语）。"""
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
    """读取当前的用户画像。"""
    if not os.path.exists(MEMORY_FILE):
        return "暂无用户画像。"
    with open(MEMORY_FILE) as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
