"""
Shared HTTP Client Pool

Provides a shared httpx.AsyncClient for connection reuse across the codebase.
This reduces connection overhead and improves performance.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx

logger = logging.getLogger(__name__)

_shared_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()

DEFAULT_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
)

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=30.0)


async def get_shared_client() -> httpx.AsyncClient:
    """
    Get or create the shared HTTP client.

    Returns:
        Shared httpx.AsyncClient instance
    """
    global _shared_client

    if _shared_client is None:
        async with _client_lock:
            if _shared_client is None:
                _shared_client = httpx.AsyncClient(
                    limits=DEFAULT_LIMITS,
                    timeout=DEFAULT_TIMEOUT,
                    follow_redirects=True,
                )
                logger.debug("Created shared HTTP client")

    return _shared_client


async def close_shared_client() -> None:
    """Close the shared HTTP client if it exists."""
    global _shared_client

    if _shared_client is not None:
        async with _client_lock:
            if _shared_client is not None:
                await _shared_client.aclose()
                _shared_client = None
                logger.debug("Closed shared HTTP client")


@asynccontextmanager
async def http_request(
    method: str,
    url: str,
    **kwargs,
):
    """
    Context manager for making HTTP requests with the shared client.

    Args:
        method: HTTP method
        url: Request URL
        **kwargs: Additional arguments for httpx.request

    Yields:
        httpx.Response
    """
    client = await get_shared_client()
    response = await client.request(method, url, **kwargs)
    try:
        yield response
    finally:
        pass


async def get(url: str, **kwargs) -> httpx.Response:
    """Convenience method for GET requests."""
    client = await get_shared_client()
    return await client.get(url, **kwargs)


async def post(url: str, **kwargs) -> httpx.Response:
    """Convenience method for POST requests."""
    client = await get_shared_client()
    return await client.post(url, **kwargs)


async def put(url: str, **kwargs) -> httpx.Response:
    """Convenience method for PUT requests."""
    client = await get_shared_client()
    return await client.put(url, **kwargs)


async def delete(url: str, **kwargs) -> httpx.Response:
    """Convenience method for DELETE requests."""
    client = await get_shared_client()
    return await client.delete(url, **kwargs)
