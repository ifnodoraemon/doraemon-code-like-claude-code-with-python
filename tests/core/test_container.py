"""
Unit tests for the Dependency Injection Container.

Tests service registration, resolution, and lifecycle management.
"""

import pytest

from src.core.container import (
    ServiceCollection,
    ServiceLifetime,
    configure_services,
    get_service,
    get_required_service,
)


# Test interfaces and implementations
class ILogger:
    """Logger interface"""

    def log(self, message: str) -> str:
        raise NotImplementedError


class ConsoleLogger(ILogger):
    """Console logger implementation"""

    def __init__(self):
        self.logs: list[str] = []

    def log(self, message: str) -> str:
        self.logs.append(message)
        return f"Console: {message}"


class FileLogger(ILogger):
    """File logger implementation"""

    def __init__(self):
        self.logs: list[str] = []

    def log(self, message: str) -> str:
        self.logs.append(message)
        return f"File: {message}"


class IConfig:
    """Config interface"""

    def get(self, key: str) -> str:
        raise NotImplementedError


class AppConfig(IConfig):
    """App config implementation"""

    def __init__(self):
        self.data = {"app_name": "Polymath"}

    def get(self, key: str) -> str:
        return self.data.get(key, "")


class ServiceWithDependency:
    """Service that depends on ILogger"""

    def __init__(self, logger: ILogger):
        self.logger = logger

    def do_work(self) -> str:
        return self.logger.log("Working...")


# ========================================
# Tests
# ========================================


class TestServiceCollection:
    """Tests for ServiceCollection class"""

    def test_add_singleton_with_class(self):
        """Test registering a singleton service with a class type"""
        services = ServiceCollection()
        services.add_singleton(ILogger, ConsoleLogger)

        provider = services.build_service_provider()
        logger = provider.get_service(ILogger)

        assert logger is not None
        assert isinstance(logger, ConsoleLogger)

    def test_add_singleton_with_instance(self):
        """Test registering a singleton service with an existing instance"""
        services = ServiceCollection()
        instance = ConsoleLogger()
        instance.logs.append("pre-existing")

        services.add_singleton(ILogger, instance)

        provider = services.build_service_provider()
        logger = provider.get_service(ILogger)

        assert logger is instance
        assert "pre-existing" in logger.logs

    def test_add_singleton_with_factory(self):
        """Test registering a singleton service with a factory function"""
        services = ServiceCollection()

        def create_logger():
            logger = ConsoleLogger()
            logger.logs.append("factory-created")
            return logger

        services.add_singleton(ILogger, create_logger)

        provider = services.build_service_provider()
        logger = provider.get_service(ILogger)

        assert isinstance(logger, ConsoleLogger)
        assert "factory-created" in logger.logs

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance every time"""
        services = ServiceCollection()
        services.add_singleton(ILogger, ConsoleLogger)

        provider = services.build_service_provider()
        logger1 = provider.get_service(ILogger)
        logger2 = provider.get_service(ILogger)

        assert logger1 is logger2

    def test_add_transient(self):
        """Test registering a transient service"""
        services = ServiceCollection()
        services.add_transient(ILogger, ConsoleLogger)

        provider = services.build_service_provider()
        logger1 = provider.get_service(ILogger)
        logger2 = provider.get_service(ILogger)

        assert logger1 is not logger2
        assert isinstance(logger1, ConsoleLogger)
        assert isinstance(logger2, ConsoleLogger)

    def test_add_scoped(self):
        """Test registering a scoped service"""
        services = ServiceCollection()
        services.add_scoped(ILogger, ConsoleLogger)

        provider = services.build_service_provider()
        # Currently scoped behaves like transient
        logger = provider.get_service(ILogger)

        assert isinstance(logger, ConsoleLogger)

    def test_fluent_api(self):
        """Test that builder methods return self for chaining"""
        services = ServiceCollection()

        result = (
            services.add_singleton(ILogger, ConsoleLogger)
            .add_transient(IConfig, AppConfig)
            .add_scoped(FileLogger)
        )

        assert result is services


class TestServiceProvider:
    """Tests for DefaultServiceProvider class"""

    def test_get_service_returns_none_for_unregistered(self):
        """Test that get_service returns None for unregistered types"""
        services = ServiceCollection()
        provider = services.build_service_provider()

        result = provider.get_service(ILogger)

        assert result is None

    def test_get_required_service_raises_for_unregistered(self):
        """Test that get_required_service raises for unregistered types"""
        services = ServiceCollection()
        provider = services.build_service_provider()

        with pytest.raises(ValueError, match="Service not registered"):
            provider.get_required_service(ILogger)

    def test_get_required_service_returns_instance(self):
        """Test that get_required_service returns instance for registered types"""
        services = ServiceCollection()
        services.add_singleton(ILogger, ConsoleLogger)

        provider = services.build_service_provider()
        logger = provider.get_required_service(ILogger)

        assert isinstance(logger, ConsoleLogger)

    def test_dependency_injection(self):
        """Test automatic dependency injection"""
        services = ServiceCollection()
        services.add_singleton(ILogger, ConsoleLogger)
        services.add_transient(ServiceWithDependency)

        provider = services.build_service_provider()
        service = provider.get_service(ServiceWithDependency)

        assert service is not None
        result = service.do_work()
        assert "Working..." in result

    def test_thread_safety(self):
        """Test that singleton creation is thread-safe"""
        import threading

        services = ServiceCollection()
        services.add_singleton(ILogger, ConsoleLogger)
        provider = services.build_service_provider()

        instances = []
        errors = []

        def get_instance():
            try:
                instance = provider.get_service(ILogger)
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        # All instances should be the same (singleton)
        assert all(inst is instances[0] for inst in instances)


class TestGlobalFunctions:
    """Tests for global helper functions"""

    def test_configure_services(self):
        """Test configure_services function"""

        def setup(services: ServiceCollection):
            services.add_singleton(ILogger, ConsoleLogger)
            services.add_singleton(IConfig, AppConfig)

        provider = configure_services(setup)

        logger = provider.get_service(ILogger)
        config = provider.get_service(IConfig)

        assert isinstance(logger, ConsoleLogger)
        assert isinstance(config, AppConfig)

    def test_get_service_global(self):
        """Test global get_service function"""

        def setup(services: ServiceCollection):
            services.add_singleton(ILogger, ConsoleLogger)

        configure_services(setup)

        logger = get_service(ILogger)
        assert isinstance(logger, ConsoleLogger)

    def test_get_required_service_global(self):
        """Test global get_required_service function"""

        def setup(services: ServiceCollection):
            services.add_singleton(ILogger, ConsoleLogger)

        configure_services(setup)

        logger = get_required_service(ILogger)
        assert isinstance(logger, ConsoleLogger)
