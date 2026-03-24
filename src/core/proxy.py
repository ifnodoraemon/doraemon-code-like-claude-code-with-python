"""
Network Proxy Support

HTTP/HTTPS/SOCKS proxy configuration and management.

Features:
- Multiple proxy protocols
- Authentication support
- Auto-detection
- Per-domain proxy rules
- Proxy rotation
"""

import logging
import os
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProxyType(Enum):
    """Proxy protocol types."""

    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"
    DIRECT = "direct"  # No proxy


@dataclass
class ProxyConfig:
    """Configuration for a proxy server."""

    host: str
    port: int
    proxy_type: ProxyType = ProxyType.HTTP
    username: str | None = None
    password: str | None = None
    name: str = ""  # Optional friendly name

    def to_url(self) -> str:
        """Convert to proxy URL string."""
        scheme = self.proxy_type.value
        if scheme == "direct":
            return ""

        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password}"
            auth += "@"

        return f"{scheme}://{auth}{self.host}:{self.port}"

    @classmethod
    def from_url(cls, url: str, name: str = "") -> "ProxyConfig":
        """Create from proxy URL string."""
        parsed = urllib.parse.urlparse(url)

        proxy_type = ProxyType(parsed.scheme) if parsed.scheme else ProxyType.HTTP
        host = parsed.hostname or ""
        port = parsed.port or (1080 if "socks" in parsed.scheme else 8080)

        return cls(
            host=host,
            port=port,
            proxy_type=proxy_type,
            username=parsed.username,
            password=parsed.password,
            name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "type": self.proxy_type.value,
            "username": self.username,
            "name": self.name,
        }


@dataclass
class ProxyRule:
    """Rule for when to use a proxy."""

    domains: list[str] = field(default_factory=list)  # Domain patterns
    proxy_name: str | None = None  # Proxy to use (None = direct)
    bypass: bool = False  # Bypass proxy for these domains


class ProxyManager:
    """
    Manages proxy configuration and selection.

    Usage:
        proxy_mgr = ProxyManager()

        # Add proxy
        proxy_mgr.add_proxy(ProxyConfig(
            host="proxy.company.com",
            port=8080,
            name="corporate"
        ))

        # Auto-detect from environment
        proxy_mgr.detect_from_environment()

        # Get proxy for URL
        proxy = proxy_mgr.get_proxy_for_url("https://example.com")

        # Apply to requests session
        session.proxies = proxy_mgr.get_requests_proxies()

        # Get environment variables
        env = proxy_mgr.get_env_vars()
    """

    def __init__(self):
        """Initialize proxy manager."""
        self._proxies: dict[str, ProxyConfig] = {}
        self._rules: list[ProxyRule] = []
        self._default_proxy: str | None = None
        self._bypass_list: list[str] = [
            "localhost",
            "127.0.0.1",
            "::1",
            "*.local",
        ]

    def add_proxy(self, config: ProxyConfig) -> str:
        """
        Add a proxy configuration.

        Args:
            config: Proxy configuration

        Returns:
            Proxy name
        """
        name = config.name or f"{config.host}:{config.port}"
        self._proxies[name] = config
        logger.info(f"Added proxy: {name}")
        return name

    def remove_proxy(self, name: str):
        """Remove a proxy."""
        if name in self._proxies:
            del self._proxies[name]

    def set_default(self, name: str | None):
        """Set the default proxy."""
        if name and name not in self._proxies:
            logger.warning(f"Unknown proxy: {name}")
            return
        self._default_proxy = name

    def add_rule(self, rule: ProxyRule):
        """Add a proxy rule."""
        self._rules.append(rule)

    def add_bypass(self, pattern: str):
        """Add a bypass pattern."""
        self._bypass_list.append(pattern)

    def detect_from_environment(self):
        """
        Detect proxy settings from environment variables.

        Checks: HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, NO_PROXY
        """
        # HTTP proxy
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        if http_proxy:
            config = ProxyConfig.from_url(http_proxy, name="env_http")
            self.add_proxy(config)
            logger.info(f"Detected HTTP proxy from environment: {config.host}")

        # HTTPS proxy
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if https_proxy:
            config = ProxyConfig.from_url(https_proxy, name="env_https")
            self.add_proxy(config)
            logger.info(f"Detected HTTPS proxy from environment: {config.host}")

        # All proxy
        all_proxy = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
        if all_proxy:
            config = ProxyConfig.from_url(all_proxy, name="env_all")
            self.add_proxy(config)
            self._default_proxy = "env_all"
            logger.info(f"Detected ALL_PROXY from environment: {config.host}")

        # No proxy
        no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
        if no_proxy:
            patterns = [p.strip() for p in no_proxy.split(",")]
            self._bypass_list.extend(patterns)
            logger.info(f"Detected NO_PROXY patterns: {len(patterns)}")

    def should_bypass(self, hostname: str) -> bool:
        """
        Check if hostname should bypass proxy.

        Args:
            hostname: Hostname to check

        Returns:
            True if should bypass proxy
        """
        import fnmatch

        hostname_lower = hostname.lower()

        for pattern in self._bypass_list:
            pattern_lower = pattern.lower().strip()
            if pattern_lower.startswith("."):
                # .example.com matches *.example.com
                pattern_lower = "*" + pattern_lower

            if fnmatch.fnmatch(hostname_lower, pattern_lower):
                return True

            # Also check if pattern is the hostname
            if hostname_lower == pattern_lower:
                return True

        return False

    def get_proxy_for_url(self, url: str) -> ProxyConfig | None:
        """
        Get the appropriate proxy for a URL.

        Args:
            url: URL to get proxy for

        Returns:
            ProxyConfig or None for direct connection
        """
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        scheme = parsed.scheme or "https"

        # Check bypass list
        if self.should_bypass(hostname):
            return None

        # Check rules
        import fnmatch

        for rule in self._rules:
            for domain in rule.domains:
                if fnmatch.fnmatch(hostname, domain):
                    if rule.bypass:
                        return None
                    if rule.proxy_name and rule.proxy_name in self._proxies:
                        return self._proxies[rule.proxy_name]

        # Check scheme-specific proxies
        if scheme == "http" and "env_http" in self._proxies:
            return self._proxies["env_http"]
        if scheme == "https" and "env_https" in self._proxies:
            return self._proxies["env_https"]

        # Use default
        if self._default_proxy and self._default_proxy in self._proxies:
            return self._proxies[self._default_proxy]

        return None

    def get_requests_proxies(self) -> dict[str, str]:
        """
        Get proxy dict for requests library.

        Returns:
            Dict suitable for requests.Session.proxies
        """
        proxies = {}

        if "env_http" in self._proxies:
            proxies["http"] = self._proxies["env_http"].to_url()

        if "env_https" in self._proxies:
            proxies["https"] = self._proxies["env_https"].to_url()
        elif self._default_proxy and self._default_proxy in self._proxies:
            default = self._proxies[self._default_proxy]
            if "http" not in proxies:
                proxies["http"] = default.to_url()
            if "https" not in proxies:
                proxies["https"] = default.to_url()

        return proxies

    def get_env_vars(self) -> dict[str, str]:
        """
        Get environment variables for subprocess.

        Returns:
            Dict of proxy environment variables
        """
        env = {}

        if "env_http" in self._proxies:
            url = self._proxies["env_http"].to_url()
            env["HTTP_PROXY"] = url
            env["http_proxy"] = url

        if "env_https" in self._proxies:
            url = self._proxies["env_https"].to_url()
            env["HTTPS_PROXY"] = url
            env["https_proxy"] = url

        if self._default_proxy and self._default_proxy in self._proxies:
            url = self._proxies[self._default_proxy].to_url()
            env["ALL_PROXY"] = url
            env["all_proxy"] = url

        if self._bypass_list:
            no_proxy = ",".join(self._bypass_list)
            env["NO_PROXY"] = no_proxy
            env["no_proxy"] = no_proxy

        return env

    def get_aiohttp_proxy(self, url: str) -> str | None:
        """
        Get proxy URL for aiohttp.

        Args:
            url: Target URL

        Returns:
            Proxy URL or None
        """
        proxy = self.get_proxy_for_url(url)
        return proxy.to_url() if proxy else None

    def get_playwright_proxy(self) -> dict | None:
        """
        Get proxy config for Playwright.

        Returns:
            Playwright proxy config dict
        """
        if not self._default_proxy or self._default_proxy not in self._proxies:
            return None

        proxy = self._proxies[self._default_proxy]

        config = {
            "server": f"{proxy.proxy_type.value}://{proxy.host}:{proxy.port}",
        }

        if proxy.username:
            config["username"] = proxy.username
        if proxy.password:
            config["password"] = proxy.password

        bypass = ",".join(self._bypass_list)
        if bypass:
            config["bypass"] = bypass

        return config

    def test_proxy(self, name: str, test_url: str = "https://www.google.com") -> bool:
        """
        Test if a proxy is working.

        Args:
            name: Proxy name
            test_url: URL to test

        Returns:
            True if proxy is working
        """
        if name not in self._proxies:
            return False

        proxy = self._proxies[name]

        try:
            import urllib.request

            proxy_handler = urllib.request.ProxyHandler(
                {
                    "http": proxy.to_url(),
                    "https": proxy.to_url(),
                }
            )

            opener = urllib.request.build_opener(proxy_handler)
            opener.open(test_url, timeout=10)
            return True

        except Exception as e:
            logger.warning(f"Proxy test failed for {name}: {e}")
            return False

    def list_proxies(self) -> list[dict]:
        """List all configured proxies."""
        return [
            {**p.to_dict(), "is_default": name == self._default_proxy}
            for name, p in self._proxies.items()
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get proxy manager summary."""
        return {
            "total_proxies": len(self._proxies),
            "default_proxy": self._default_proxy,
            "rules_count": len(self._rules),
            "bypass_patterns": len(self._bypass_list),
            "proxies": list(self._proxies.keys()),
        }


# Global instance
_proxy_manager: ProxyManager | None = None


def get_proxy_manager() -> ProxyManager:
    """Get the global proxy manager."""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
        _proxy_manager.detect_from_environment()
    return _proxy_manager
