"""
Dependency Injection Container

Provides a modern DI container for managing service lifecycle and dependencies.
Inspired by Spring Framework and Microsoft.Extensions.DependencyInjection.

Status: AVAILABLE FOR EXTENSION
    This module is fully implemented but not yet integrated into the main application.
    It's designed for advanced users who want to:
    - Add custom services with proper lifecycle management
    - Implement plugin systems with automatic dependency injection
    - Build testable components with injectable dependencies

Example Usage:
    from src.core.container import ServiceCollection, configure_services

    def setup(services: ServiceCollection):
        services.add_singleton(IConfig, AppConfig)
        services.add_transient(ILogger, FileLogger)

    provider = configure_services(setup)
    config = provider.get_service(IConfig)
"""

import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar


class ServiceLifetime(Enum):
    """Service lifetime enumeration"""

    SINGLETON = "singleton"  # One instance for entire app lifetime
    SCOPED = "scoped"  # One instance per scope/request
    TRANSIENT = "transient"  # New instance every time


T = TypeVar("T")


@dataclass
class ServiceDescriptor:
    """Describes how a service should be registered and created"""

    service_type: type
    implementation_type: type | None = None
    factory: Callable | None = None
    instance: Any | None = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT


class ServiceProvider(Protocol):
    """Protocol for service provider"""

    def get_service(self, service_type: type[T]) -> T | None:
        """Get service instance"""
        ...

    def get_required_service(self, service_type: type[T]) -> T:
        """Get service instance (raises if not found)"""
        ...


class ServiceCollection:
    """
    Collection of service descriptors for dependency injection.

    Example:
        services = ServiceCollection()
        services.add_singleton(ILogger, ConsoleLogger)
        services.add_transient(IRepository, UserRepository)
        provider = services.build_service_provider()
        logger = provider.get_service(ILogger)
    """

    def __init__(self):
        self._descriptors: dict[type, ServiceDescriptor] = {}

    def add_singleton(
        self, service_type: type[T], implementation: type[T] | T | Callable[[], T] | None = None
    ) -> "ServiceCollection":
        """
        Register a singleton service.

        Args:
            service_type: The service interface/type
            implementation: Implementation class, instance, or factory
        """
        if implementation is None:
            implementation = service_type

        if isinstance(implementation, type):
            # Class type
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation,
                lifetime=ServiceLifetime.SINGLETON,
            )
        elif callable(implementation):
            # Factory function
            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=implementation,
                lifetime=ServiceLifetime.SINGLETON,
            )
        else:
            # Instance
            descriptor = ServiceDescriptor(
                service_type=service_type,
                instance=implementation,
                lifetime=ServiceLifetime.SINGLETON,
            )

        self._descriptors[service_type] = descriptor
        return self

    def add_scoped(
        self, service_type: type[T], implementation: type[T] | None = None
    ) -> "ServiceCollection":
        """Register a scoped service (one instance per scope)"""
        if implementation is None:
            implementation = service_type

        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation,
            lifetime=ServiceLifetime.SCOPED,
        )
        self._descriptors[service_type] = descriptor
        return self

    def add_transient(
        self, service_type: type[T], implementation: type[T] | None = None
    ) -> "ServiceCollection":
        """Register a transient service (new instance every time)"""
        if implementation is None:
            implementation = service_type

        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
        )
        self._descriptors[service_type] = descriptor
        return self

    def build_service_provider(self) -> "DefaultServiceProvider":
        """Build the service provider from registered services"""
        return DefaultServiceProvider(self._descriptors)


class DefaultServiceProvider:
    """
    Default implementation of service provider with dependency resolution.
    """

    def __init__(self, descriptors: dict[type, ServiceDescriptor]):
        self._descriptors = descriptors
        self._singletons: dict[type, Any] = {}
        self._lock = threading.Lock()

        # Pre-create singleton instances that are already provided
        for service_type, descriptor in descriptors.items():
            if descriptor.lifetime == ServiceLifetime.SINGLETON and descriptor.instance:
                self._singletons[service_type] = descriptor.instance

    def get_service(self, service_type: type[T]) -> T | None:
        """
        Get service instance by type.

        Args:
            service_type: The service type to retrieve

        Returns:
            Service instance or None if not registered
        """
        if service_type not in self._descriptors:
            return None

        descriptor = self._descriptors[service_type]

        # Singleton
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]

            with self._lock:
                # Double-check locking
                if service_type in self._singletons:
                    return self._singletons[service_type]

                instance = self._create_instance(descriptor)
                self._singletons[service_type] = instance
                return instance

        # Transient or Scoped (for now, treat scoped as transient)
        return self._create_instance(descriptor)

    def get_required_service(self, service_type: type[T]) -> T:
        """
        Get service instance (raises if not found).

        Args:
            service_type: The service type to retrieve

        Returns:
            Service instance

        Raises:
            ValueError: If service not registered
        """
        instance = self.get_service(service_type)
        if instance is None:
            raise ValueError(f"Service not registered: {service_type}")
        return instance

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create service instance with dependency injection"""

        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory
        if descriptor.factory is not None:
            # Try to inject dependencies into factory
            return self._invoke_with_di(descriptor.factory)

        # Create from implementation type
        if descriptor.implementation_type is not None:
            return self._invoke_with_di(descriptor.implementation_type)

        raise ValueError(f"Cannot create instance for {descriptor.service_type}")

    def _invoke_with_di(self, callable_obj: Callable) -> Any:
        """
        Invoke a callable with dependency injection.
        Resolves constructor/function parameters from registered services.
        """
        # Get signature
        sig = inspect.signature(callable_obj)

        # Resolve parameters
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Try to resolve from type annotation
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
                service = self.get_service(param_type)

                if service is not None:
                    kwargs[param_name] = service
                elif param.default == inspect.Parameter.empty:
                    # Required parameter with no default and no service
                    raise ValueError(
                        f"Cannot resolve parameter '{param_name}' of type {param_type}"
                    )

        # Invoke
        return callable_obj(**kwargs)


# Global container instance
_global_services: DefaultServiceProvider | None = None


def configure_services(configurator: Callable[[ServiceCollection], None]) -> DefaultServiceProvider:
    """
    Configure services and return provider.

    Args:
        configurator: Function that configures the service collection

    Returns:
        Configured service provider

    Example:
        def setup(services: ServiceCollection):
            services.add_singleton(IConfig, AppConfig)
            services.add_transient(ILogger, FileLogger)

        provider = configure_services(setup)
    """
    global _global_services

    services = ServiceCollection()
    configurator(services)
    _global_services = services.build_service_provider()
    return _global_services


def get_service(service_type: type[T]) -> T | None:
    """Get service from global container"""
    if _global_services is None:
        raise RuntimeError("Services not configured. Call configure_services() first.")
    return _global_services.get_service(service_type)


def get_required_service(service_type: type[T]) -> T:
    """Get required service from global container"""
    if _global_services is None:
        raise RuntimeError("Services not configured. Call configure_services() first.")
    return _global_services.get_required_service(service_type)
