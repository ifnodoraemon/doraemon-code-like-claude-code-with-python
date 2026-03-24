"""
Comprehensive tests for src/core/proxy.py

Tests cover:
- Proxy configuration and initialization
- Request/response interception
- Proxy middleware and filters
- Error handling
- HTTP operations with mocking
"""

import os
from unittest.mock import MagicMock, patch

from src.core.proxy import (
    ProxyConfig,
    ProxyManager,
    ProxyRule,
    ProxyType,
    get_proxy_manager,
)


class TestProxyType:
    """Tests for ProxyType enum."""

    def test_proxy_type_http(self):
        """Test HTTP proxy type."""
        assert ProxyType.HTTP.value == "http"

    def test_proxy_type_https(self):
        """Test HTTPS proxy type."""
        assert ProxyType.HTTPS.value == "https"

    def test_proxy_type_socks4(self):
        """Test SOCKS4 proxy type."""
        assert ProxyType.SOCKS4.value == "socks4"

    def test_proxy_type_socks5(self):
        """Test SOCKS5 proxy type."""
        assert ProxyType.SOCKS5.value == "socks5"

    def test_proxy_type_direct(self):
        """Test DIRECT proxy type (no proxy)."""
        assert ProxyType.DIRECT.value == "direct"

    def test_proxy_type_enum_members(self):
        """Test all proxy type enum members exist."""
        types = [
            ProxyType.HTTP,
            ProxyType.HTTPS,
            ProxyType.SOCKS4,
            ProxyType.SOCKS5,
            ProxyType.DIRECT,
        ]
        assert len(types) == 5


class TestProxyConfig:
    """Tests for ProxyConfig dataclass."""

    def test_proxy_config_initialization(self):
        """Test basic proxy config initialization."""
        config = ProxyConfig(host="proxy.example.com", port=8080)
        assert config.host == "proxy.example.com"
        assert config.port == 8080
        assert config.proxy_type == ProxyType.HTTP
        assert config.username is None
        assert config.password is None

    def test_proxy_config_with_auth(self):
        """Test proxy config with authentication."""
        config = ProxyConfig(host="proxy.example.com", port=8080, username="user", password="pass")
        assert config.username == "user"
        assert config.password == "pass"

    def test_proxy_config_with_name(self):
        """Test proxy config with friendly name."""
        config = ProxyConfig(host="proxy.example.com", port=8080, name="corporate")
        assert config.name == "corporate"

    def test_proxy_config_socks5(self):
        """Test SOCKS5 proxy config."""
        config = ProxyConfig(host="socks.example.com", port=1080, proxy_type=ProxyType.SOCKS5)
        assert config.proxy_type == ProxyType.SOCKS5

    def test_proxy_config_to_url_http(self):
        """Test converting HTTP proxy config to URL."""
        config = ProxyConfig(host="proxy.example.com", port=8080)
        url = config.to_url()
        assert url == "http://proxy.example.com:8080"

    def test_proxy_config_to_url_https(self):
        """Test converting HTTPS proxy config to URL."""
        config = ProxyConfig(host="proxy.example.com", port=8080, proxy_type=ProxyType.HTTPS)
        url = config.to_url()
        assert url == "https://proxy.example.com:8080"

    def test_proxy_config_to_url_socks5(self):
        """Test converting SOCKS5 proxy config to URL."""
        config = ProxyConfig(host="socks.example.com", port=1080, proxy_type=ProxyType.SOCKS5)
        url = config.to_url()
        assert url == "socks5://socks.example.com:1080"

    def test_proxy_config_to_url_with_auth(self):
        """Test converting proxy config with auth to URL."""
        config = ProxyConfig(host="proxy.example.com", port=8080, username="user", password="pass")
        url = config.to_url()
        assert url == "http://user:pass@proxy.example.com:8080"

    def test_proxy_config_to_url_with_username_only(self):
        """Test converting proxy config with username only to URL."""
        config = ProxyConfig(host="proxy.example.com", port=8080, username="user")
        url = config.to_url()
        assert url == "http://user@proxy.example.com:8080"

    def test_proxy_config_to_url_direct(self):
        """Test converting DIRECT proxy config to URL."""
        config = ProxyConfig(host="localhost", port=0, proxy_type=ProxyType.DIRECT)
        url = config.to_url()
        assert url == ""

    def test_proxy_config_from_url_http(self):
        """Test creating proxy config from HTTP URL."""
        config = ProxyConfig.from_url("http://proxy.example.com:8080")
        assert config.host == "proxy.example.com"
        assert config.port == 8080
        assert config.proxy_type == ProxyType.HTTP

    def test_proxy_config_from_url_https(self):
        """Test creating proxy config from HTTPS URL."""
        config = ProxyConfig.from_url("https://proxy.example.com:8443")
        assert config.host == "proxy.example.com"
        assert config.port == 8443
        assert config.proxy_type == ProxyType.HTTPS

    def test_proxy_config_from_url_socks5(self):
        """Test creating proxy config from SOCKS5 URL."""
        config = ProxyConfig.from_url("socks5://socks.example.com:1080")
        assert config.host == "socks.example.com"
        assert config.port == 1080
        assert config.proxy_type == ProxyType.SOCKS5

    def test_proxy_config_from_url_socks4(self):
        """Test creating proxy config from SOCKS4 URL."""
        config = ProxyConfig.from_url("socks4://socks.example.com:1080")
        assert config.host == "socks.example.com"
        assert config.port == 1080
        assert config.proxy_type == ProxyType.SOCKS4

    def test_proxy_config_from_url_with_auth(self):
        """Test creating proxy config from URL with authentication."""
        config = ProxyConfig.from_url("http://user:pass@proxy.example.com:8080")
        assert config.host == "proxy.example.com"
        assert config.port == 8080
        assert config.username == "user"
        assert config.password == "pass"

    def test_proxy_config_from_url_default_port_http(self):
        """Test default port for HTTP proxy."""
        config = ProxyConfig.from_url("http://proxy.example.com")
        assert config.port == 8080

    def test_proxy_config_from_url_default_port_socks(self):
        """Test default port for SOCKS proxy."""
        config = ProxyConfig.from_url("socks5://socks.example.com")
        assert config.port == 1080

    def test_proxy_config_from_url_with_name(self):
        """Test creating proxy config from URL with name."""
        config = ProxyConfig.from_url("http://proxy.example.com:8080", name="corporate")
        assert config.name == "corporate"

    def test_proxy_config_to_dict(self):
        """Test converting proxy config to dictionary."""
        config = ProxyConfig(host="proxy.example.com", port=8080, username="user", name="corporate")
        d = config.to_dict()
        assert d["host"] == "proxy.example.com"
        assert d["port"] == 8080
        assert d["type"] == "http"
        assert d["username"] == "user"
        assert d["name"] == "corporate"

    def test_proxy_config_to_dict_no_username(self):
        """Test converting proxy config without username to dictionary."""
        config = ProxyConfig(host="proxy.example.com", port=8080)
        d = config.to_dict()
        assert d["username"] is None


class TestProxyRule:
    """Tests for ProxyRule dataclass."""

    def test_proxy_rule_initialization(self):
        """Test basic proxy rule initialization."""
        rule = ProxyRule()
        assert rule.domains == []
        assert rule.proxy_name is None
        assert rule.bypass is False

    def test_proxy_rule_with_domains(self):
        """Test proxy rule with domains."""
        rule = ProxyRule(domains=["*.example.com", "test.org"])
        assert rule.domains == ["*.example.com", "test.org"]

    def test_proxy_rule_with_proxy_name(self):
        """Test proxy rule with proxy name."""
        rule = ProxyRule(proxy_name="corporate")
        assert rule.proxy_name == "corporate"

    def test_proxy_rule_bypass(self):
        """Test proxy rule with bypass flag."""
        rule = ProxyRule(domains=["localhost"], bypass=True)
        assert rule.bypass is True

    def test_proxy_rule_complete(self):
        """Test complete proxy rule."""
        rule = ProxyRule(domains=["*.internal.com"], proxy_name="internal_proxy", bypass=False)
        assert rule.domains == ["*.internal.com"]
        assert rule.proxy_name == "internal_proxy"
        assert rule.bypass is False


class TestProxyManager:
    """Tests for ProxyManager class."""

    def test_proxy_manager_initialization(self):
        """Test proxy manager initialization."""
        manager = ProxyManager()
        assert manager._proxies == {}
        assert manager._rules == []
        assert manager._default_proxy is None
        assert len(manager._bypass_list) > 0

    def test_proxy_manager_default_bypass_list(self):
        """Test default bypass list contains localhost."""
        manager = ProxyManager()
        assert "localhost" in manager._bypass_list
        assert "127.0.0.1" in manager._bypass_list
        assert "::1" in manager._bypass_list

    def test_add_proxy(self):
        """Test adding a proxy."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        name = manager.add_proxy(config)
        assert name == "test"
        assert "test" in manager._proxies

    def test_add_proxy_without_name(self):
        """Test adding a proxy without explicit name."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080)
        name = manager.add_proxy(config)
        assert name == "proxy.example.com:8080"

    def test_remove_proxy(self):
        """Test removing a proxy."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.remove_proxy("test")
        assert "test" not in manager._proxies

    def test_remove_nonexistent_proxy(self):
        """Test removing a non-existent proxy."""
        manager = ProxyManager()
        manager.remove_proxy("nonexistent")  # Should not raise

    def test_set_default_proxy(self):
        """Test setting default proxy."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")
        assert manager._default_proxy == "test"

    def test_set_default_proxy_nonexistent(self):
        """Test setting non-existent proxy as default."""
        manager = ProxyManager()
        manager.set_default("nonexistent")
        assert manager._default_proxy is None

    def test_set_default_proxy_none(self):
        """Test setting default proxy to None."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")
        manager.set_default(None)
        assert manager._default_proxy is None

    def test_add_rule(self):
        """Test adding a proxy rule."""
        manager = ProxyManager()
        rule = ProxyRule(domains=["*.example.com"], proxy_name="test")
        manager.add_rule(rule)
        assert len(manager._rules) == 1
        assert manager._rules[0] == rule

    def test_add_bypass(self):
        """Test adding a bypass pattern."""
        manager = ProxyManager()
        initial_count = len(manager._bypass_list)
        manager.add_bypass("*.internal.com")
        assert len(manager._bypass_list) == initial_count + 1
        assert "*.internal.com" in manager._bypass_list

    def test_should_bypass_localhost(self):
        """Test bypass check for localhost."""
        manager = ProxyManager()
        assert manager.should_bypass("localhost") is True

    def test_should_bypass_127_0_0_1(self):
        """Test bypass check for 127.0.0.1."""
        manager = ProxyManager()
        assert manager.should_bypass("127.0.0.1") is True

    def test_should_bypass_ipv6_loopback(self):
        """Test bypass check for IPv6 loopback."""
        manager = ProxyManager()
        assert manager.should_bypass("::1") is True

    def test_should_bypass_wildcard_pattern(self):
        """Test bypass check with wildcard pattern."""
        manager = ProxyManager()
        manager.add_bypass("*.local")
        assert manager.should_bypass("test.local") is True

    def test_should_bypass_dot_pattern(self):
        """Test bypass check with dot pattern."""
        manager = ProxyManager()
        manager.add_bypass(".example.com")
        assert manager.should_bypass("test.example.com") is True

    def test_should_bypass_case_insensitive(self):
        """Test bypass check is case insensitive."""
        manager = ProxyManager()
        manager.add_bypass("EXAMPLE.COM")
        assert manager.should_bypass("example.com") is True

    def test_should_not_bypass(self):
        """Test should not bypass for external domain."""
        manager = ProxyManager()
        assert manager.should_bypass("example.com") is False

    @patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.example.com:8080"})
    def test_detect_from_environment_http(self):
        """Test detecting HTTP proxy from environment."""
        manager = ProxyManager()
        manager.detect_from_environment()
        assert "env_http" in manager._proxies

    @patch.dict(os.environ, {"HTTPS_PROXY": "https://proxy.example.com:8443"})
    def test_detect_from_environment_https(self):
        """Test detecting HTTPS proxy from environment."""
        manager = ProxyManager()
        manager.detect_from_environment()
        assert "env_https" in manager._proxies

    @patch.dict(os.environ, {"ALL_PROXY": "http://proxy.example.com:8080"})
    def test_detect_from_environment_all(self):
        """Test detecting ALL_PROXY from environment."""
        manager = ProxyManager()
        manager.detect_from_environment()
        assert "env_all" in manager._proxies
        assert manager._default_proxy == "env_all"

    @patch.dict(os.environ, {"NO_PROXY": "localhost,127.0.0.1,.local"})
    def test_detect_from_environment_no_proxy(self):
        """Test detecting NO_PROXY from environment."""
        manager = ProxyManager()
        initial_count = len(manager._bypass_list)
        manager.detect_from_environment()
        assert len(manager._bypass_list) > initial_count

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_from_environment_empty(self):
        """Test detecting from empty environment."""
        manager = ProxyManager()
        manager.detect_from_environment()
        # Should not raise and should have default bypass list

    def test_get_proxy_for_url_bypass(self):
        """Test getting proxy for URL that should bypass."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")
        proxy = manager.get_proxy_for_url("http://localhost:8000")
        assert proxy is None

    def test_get_proxy_for_url_default(self):
        """Test getting proxy for URL with default proxy."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")
        proxy = manager.get_proxy_for_url("http://example.com")
        assert proxy == config

    def test_get_proxy_for_url_rule_match(self):
        """Test getting proxy for URL matching a rule."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        rule = ProxyRule(domains=["*.example.com"], proxy_name="test")
        manager.add_rule(rule)
        proxy = manager.get_proxy_for_url("http://api.example.com")
        assert proxy == config

    def test_get_proxy_for_url_rule_bypass(self):
        """Test getting proxy for URL with bypass rule."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")
        rule = ProxyRule(domains=["*.internal.com"], bypass=True)
        manager.add_rule(rule)
        proxy = manager.get_proxy_for_url("http://api.internal.com")
        assert proxy is None

    def test_get_proxy_for_url_no_proxy(self):
        """Test getting proxy for URL when no proxy configured."""
        manager = ProxyManager()
        proxy = manager.get_proxy_for_url("http://example.com")
        assert proxy is None

    def test_get_requests_proxies_empty(self):
        """Test getting requests proxies when empty."""
        manager = ProxyManager()
        proxies = manager.get_requests_proxies()
        assert proxies == {}

    @patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.example.com:8080"})
    def test_get_requests_proxies_http(self):
        """Test getting requests proxies with HTTP proxy."""
        manager = ProxyManager()
        manager.detect_from_environment()
        proxies = manager.get_requests_proxies()
        assert "http" in proxies
        assert proxies["http"] == "http://proxy.example.com:8080"

    @patch.dict(os.environ, {"HTTPS_PROXY": "https://proxy.example.com:8443"})
    def test_get_requests_proxies_https(self):
        """Test getting requests proxies with HTTPS proxy."""
        manager = ProxyManager()
        manager.detect_from_environment()
        proxies = manager.get_requests_proxies()
        assert "https" in proxies
        assert proxies["https"] == "https://proxy.example.com:8443"

    def test_get_env_vars_empty(self):
        """Test getting environment variables when empty."""
        manager = ProxyManager()
        env = manager.get_env_vars()
        assert "NO_PROXY" in env or "no_proxy" in env

    @patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.example.com:8080"})
    def test_get_env_vars_http(self):
        """Test getting environment variables with HTTP proxy."""
        manager = ProxyManager()
        manager.detect_from_environment()
        env = manager.get_env_vars()
        assert env["HTTP_PROXY"] == "http://proxy.example.com:8080"
        assert env["http_proxy"] == "http://proxy.example.com:8080"

    @patch.dict(os.environ, {"ALL_PROXY": "http://proxy.example.com:8080"})
    def test_get_env_vars_all(self):
        """Test getting environment variables with ALL_PROXY."""
        manager = ProxyManager()
        manager.detect_from_environment()
        env = manager.get_env_vars()
        assert env["ALL_PROXY"] == "http://proxy.example.com:8080"
        assert env["all_proxy"] == "http://proxy.example.com:8080"

    def test_get_aiohttp_proxy_none(self):
        """Test getting aiohttp proxy when none configured."""
        manager = ProxyManager()
        proxy = manager.get_aiohttp_proxy("http://example.com")
        assert proxy is None

    def test_get_aiohttp_proxy_configured(self):
        """Test getting aiohttp proxy when configured."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")
        proxy = manager.get_aiohttp_proxy("http://example.com")
        assert proxy == "http://proxy.example.com:8080"

    def test_get_playwright_proxy_none(self):
        """Test getting Playwright proxy when none configured."""
        manager = ProxyManager()
        proxy = manager.get_playwright_proxy()
        assert proxy is None

    def test_get_playwright_proxy_configured(self):
        """Test getting Playwright proxy when configured."""
        manager = ProxyManager()
        config = ProxyConfig(
            host="proxy.example.com", port=8080, name="test", username="user", password="pass"
        )
        manager.add_proxy(config)
        manager.set_default("test")
        proxy = manager.get_playwright_proxy()
        assert proxy is not None
        assert proxy["server"] == "http://proxy.example.com:8080"
        assert proxy["username"] == "user"
        assert proxy["password"] == "pass"

    def test_get_playwright_proxy_no_auth(self):
        """Test getting Playwright proxy without authentication."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")
        proxy = manager.get_playwright_proxy()
        assert "username" not in proxy
        assert "password" not in proxy

    @patch("urllib.request.build_opener")
    def test_test_proxy_success(self, mock_opener):
        """Test proxy testing with successful connection."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)

        mock_opener_instance = MagicMock()
        mock_opener.return_value = mock_opener_instance

        result = manager.test_proxy("test")
        assert result is True

    @patch("urllib.request.build_opener")
    def test_test_proxy_failure(self, mock_opener):
        """Test proxy testing with failed connection."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)

        mock_opener_instance = MagicMock()
        mock_opener_instance.open.side_effect = Exception("Connection failed")
        mock_opener.return_value = mock_opener_instance

        result = manager.test_proxy("test")
        assert result is False

    def test_test_proxy_nonexistent(self):
        """Test proxy testing for non-existent proxy."""
        manager = ProxyManager()
        result = manager.test_proxy("nonexistent")
        assert result is False

    def test_list_proxies_empty(self):
        """Test listing proxies when empty."""
        manager = ProxyManager()
        proxies = manager.list_proxies()
        assert proxies == []

    def test_list_proxies_with_proxies(self):
        """Test listing proxies with configured proxies."""
        manager = ProxyManager()
        config1 = ProxyConfig(host="proxy1.example.com", port=8080, name="test1")
        config2 = ProxyConfig(host="proxy2.example.com", port=8080, name="test2")
        manager.add_proxy(config1)
        manager.add_proxy(config2)
        manager.set_default("test1")

        proxies = manager.list_proxies()
        assert len(proxies) == 2
        assert any(p["is_default"] for p in proxies)

    def test_list_proxies_default_flag(self):
        """Test list proxies includes default flag."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")

        proxies = manager.list_proxies()
        assert proxies[0]["is_default"] is True

    def test_get_summary(self):
        """Test getting proxy manager summary."""
        manager = ProxyManager()
        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        rule = ProxyRule(domains=["*.example.com"])
        manager.add_rule(rule)

        summary = manager.get_summary()
        assert summary["total_proxies"] == 1
        assert summary["rules_count"] == 1
        assert summary["bypass_patterns"] > 0
        assert "test" in summary["proxies"]

    def test_get_summary_empty(self):
        """Test getting summary for empty manager."""
        manager = ProxyManager()
        summary = manager.get_summary()
        assert summary["total_proxies"] == 0
        assert summary["rules_count"] == 0
        assert summary["default_proxy"] is None


class TestGlobalProxyManager:
    """Tests for global proxy manager function."""

    def test_get_proxy_manager_singleton(self):
        """Test get_proxy_manager returns singleton."""
        import src.core.proxy as proxy_module

        # Reset global instance
        proxy_module._proxy_manager = None

        manager1 = get_proxy_manager()
        manager2 = get_proxy_manager()

        assert manager1 is manager2

    def test_get_proxy_manager_initializes(self):
        """Test get_proxy_manager initializes manager."""
        import src.core.proxy as proxy_module

        # Reset global instance
        proxy_module._proxy_manager = None

        manager = get_proxy_manager()
        assert isinstance(manager, ProxyManager)

    @patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.example.com:8080"})
    def test_get_proxy_manager_detects_environment(self):
        """Test get_proxy_manager detects environment on init."""
        import src.core.proxy as proxy_module

        # Reset global instance
        proxy_module._proxy_manager = None

        manager = get_proxy_manager()
        assert "env_http" in manager._proxies


class TestProxyIntegration:
    """Integration tests for proxy functionality."""

    def test_proxy_workflow_complete(self):
        """Test complete proxy workflow."""
        manager = ProxyManager()

        # Add proxies
        config1 = ProxyConfig(host="proxy1.example.com", port=8080, name="proxy1")
        config2 = ProxyConfig(host="proxy2.example.com", port=8080, name="proxy2")
        manager.add_proxy(config1)
        manager.add_proxy(config2)

        # Add rules
        rule = ProxyRule(domains=["*.internal.com"], proxy_name="proxy1")
        manager.add_rule(rule)

        # Set default
        manager.set_default("proxy2")

        # Test URL selection
        internal_proxy = manager.get_proxy_for_url("http://api.internal.com")
        assert internal_proxy == config1

        external_proxy = manager.get_proxy_for_url("http://example.com")
        assert external_proxy == config2

    def test_proxy_url_roundtrip(self):
        """Test proxy URL conversion roundtrip."""
        original_url = "http://user:pass@proxy.example.com:8080"
        config = ProxyConfig.from_url(original_url)
        converted_url = config.to_url()
        assert converted_url == original_url

    def test_proxy_with_special_characters(self):
        """Test proxy with special characters in password."""
        config = ProxyConfig(
            host="proxy.example.com", port=8080, username="user", password="p@ss:word"
        )
        url = config.to_url()
        assert "p@ss:word" in url

    def test_multiple_rules_priority(self):
        """Test multiple rules are checked in order."""
        manager = ProxyManager()

        config1 = ProxyConfig(host="proxy1.example.com", port=8080, name="proxy1")
        config2 = ProxyConfig(host="proxy2.example.com", port=8080, name="proxy2")
        manager.add_proxy(config1)
        manager.add_proxy(config2)

        # Add rules in specific order
        rule1 = ProxyRule(domains=["*.example.com"], proxy_name="proxy1")
        rule2 = ProxyRule(domains=["api.example.com"], proxy_name="proxy2")
        manager.add_rule(rule1)
        manager.add_rule(rule2)

        # First matching rule should be used
        proxy = manager.get_proxy_for_url("http://api.example.com")
        assert proxy == config1

    def test_bypass_takes_precedence(self):
        """Test bypass rules take precedence over proxy rules."""
        manager = ProxyManager()

        config = ProxyConfig(host="proxy.example.com", port=8080, name="test")
        manager.add_proxy(config)
        manager.set_default("test")

        # Add bypass rule first (rules are checked in order)
        bypass_rule = ProxyRule(domains=["internal.example.com"], bypass=True)
        manager.add_rule(bypass_rule)

        # Add rule to use proxy
        rule = ProxyRule(domains=["*.example.com"], proxy_name="test")
        manager.add_rule(rule)

        # Bypass should take precedence when checked first
        proxy = manager.get_proxy_for_url("http://internal.example.com")
        assert proxy is None
