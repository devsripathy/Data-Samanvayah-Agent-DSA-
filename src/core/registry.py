"""
Central registry module for the Data Samanvayah Agent (DSA).

This module provides a flexible, extensible registry system for managing
all pluggable components including agents, models, embeddings, memory backends,
data loaders, visualizations, and metrics. It supports decorator-based registration,
lazy loading, dependency validation, and dynamic plugin discovery.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import Any, Callable, Generic, Literal, TypeVar, get_type_hints
from functools import wraps

logger = logging.getLogger(__name__)

# Type variable for generic registry
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Registry Class
# ---------------------------------------------------------------------------

class Registry(Generic[T]):
    """
    Generic registry for managing pluggable components.
    
    Supports lazy loading, dependency validation, and thread-safe operations.
    
    Attributes:
        name: Human-readable name for this registry (e.g., "agents", "models").
        _registry: Internal storage mapping names to registered items.
        _lock: Thread lock for concurrent access safety.
    """
    
    def __init__(self, name: str) -> None:
        """
        Initializes a new registry instance.
        
        Args:
            name: Identifier for this registry type.
        """
        self.name = name
        self._registry: dict[str, type[T] | Callable[..., T] | str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        
    def register(
        self,
        name: str,
        obj: type[T] | Callable[..., T] | str | None = None,
        *,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        lazy: bool = False,
    ) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]] | None:
        """
        Registers a component with the registry.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: Unique identifier for the component.
            obj: The component class, function, or import path string.
            dependencies: List of required Python package names.
            metadata: Additional metadata about the component.
            lazy: If True, obj should be an import path string for lazy loading.
            
        Returns:
            Decorator function if obj is None, otherwise None.
            
        Raises:
            ValueError: If name is already registered.
            ImportError: If dependencies are not satisfied.
        """
        # Validate dependencies
        if dependencies:
            self._validate_dependencies(name, dependencies)
        
        def decorator(component: type[T] | Callable[..., T]) -> type[T] | Callable[..., T]:
            """Inner decorator for registration."""
            with self._lock:
                if name in self._registry:
                    raise ValueError(
                        f"Component '{name}' already registered in {self.name} registry. "
                        f"Use a different name or unregister the existing one first."
                    )
                
                # Store the component (or import path for lazy loading)
                if lazy and isinstance(component, str):
                    self._registry[name] = component
                else:
                    self._registry[name] = component
                
                # Store metadata
                self._metadata[name] = {
                    "dependencies": dependencies or [],
                    "lazy": lazy,
                    "metadata": metadata or {},
                    "registered_at": importlib.import_module("datetime").datetime.now(),
                }
                
                logger.info(f"Registered {self.name}: {name}")
                
            return component
        
        # If obj is provided, register immediately
        if obj is not None:
            if lazy and isinstance(obj, str):
                # Lazy loading with string path
                with self._lock:
                    if name in self._registry:
                        raise ValueError(f"Component '{name}' already registered.")
                    self._registry[name] = obj
                    self._metadata[name] = {
                        "dependencies": dependencies or [],
                        "lazy": True,
                        "metadata": metadata or {},
                    }
                    logger.info(f"Registered {self.name} (lazy): {name}")
                return None
            else:
                decorator(obj)
                return None
        
        # Otherwise, return decorator
        return decorator
    
    def unregister(self, name: str) -> None:
        """
        Removes a component from the registry.
        
        Args:
            name: The component name to unregister.
            
        Raises:
            KeyError: If component is not registered.
        """
        with self._lock:
            if name not in self._registry:
                raise KeyError(f"Component '{name}' not found in {self.name} registry.")
            
            del self._registry[name]
            if name in self._metadata:
                del self._metadata[name]
            
            logger.info(f"Unregistered {self.name}: {name}")
    
    def get(self, name: str) -> type[T] | Callable[..., T]:
        """
        Retrieves a registered component by name.
        
        Supports lazy loading if the component was registered with a string path.
        
        Args:
            name: The component name.
            
        Returns:
            The registered component (class or function).
            
        Raises:
            KeyError: If component is not registered.
            ImportError: If lazy loading fails.
        """
        with self._lock:
            if name not in self._registry:
                raise KeyError(
                    f"Component '{name}' not found in {self.name} registry. "
                    f"Available: {list(self._registry.keys())}"
                )
            
            component = self._registry[name]
            
            # Handle lazy loading
            if isinstance(component, str):
                logger.debug(f"Lazy loading {self.name}: {name}")
                component = self._load_from_path(component)
                self._registry[name] = component  # Cache the loaded component
            
            return component
    
    def get_metadata(self, name: str) -> dict[str, Any]:
        """
        Retrieves metadata for a registered component.
        
        Args:
            name: The component name.
            
        Returns:
            Dictionary containing component metadata.
            
        Raises:
            KeyError: If component is not registered.
        """
        with self._lock:
            if name not in self._metadata:
                raise KeyError(f"Metadata for '{name}' not found in {self.name} registry.")
            return self._metadata[name].copy()
    
    def list(self) -> list[str]:
        """
        Returns a list of all registered component names.
        
        Returns:
            Sorted list of component names.
        """
        with self._lock:
            return sorted(self._registry.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Checks if a component is registered.
        
        Args:
            name: The component name to check.
            
        Returns:
            True if registered, False otherwise.
        """
        with self._lock:
            return name in self._registry
    
    def _validate_dependencies(self, name: str, dependencies: list[str]) -> None:
        """
        Validates that all required dependencies are available.
        
        Args:
            name: The component name (for error messages).
            dependencies: List of required package names.
            
        Raises:
            ImportError: If any dependency is missing.
        """
        missing = []
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            raise ImportError(
                f"Cannot register '{name}' in {self.name} registry. "
                f"Missing dependencies: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )
    
    def _load_from_path(self, path: str) -> type[T] | Callable[..., T]:
        """
        Dynamically imports a component from a module path string.
        
        Args:
            path: Import path in format "module.path:ClassName" or "module.path.function_name".
            
        Returns:
            The imported component.
            
        Raises:
            ImportError: If the module or attribute cannot be imported.
            ValueError: If the path format is invalid.
        """
        try:
            if ":" in path:
                module_path, attr_name = path.rsplit(":", 1)
            elif "." in path:
                module_path, attr_name = path.rsplit(".", 1)
            else:
                raise ValueError(
                    f"Invalid import path '{path}'. "
                    f"Expected format: 'module.path:ClassName' or 'module.path.function_name'"
                )
            
            module = importlib.import_module(module_path)
            component = getattr(module, attr_name)
            
            return component
            
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(
                f"Failed to lazy load component from path '{path}': {e}"
            ) from e


# ---------------------------------------------------------------------------
# Global Registry Instances
# ---------------------------------------------------------------------------

class RegistryManager:
    """
    Central manager for all DSA component registries.
    
    Provides singleton access to specialized registries for different
    component types (agents, models, embeddings, etc.).
    """
    
    _instance: RegistryManager | None = None
    _lock = threading.Lock()
    
    def __new__(cls) -> RegistryManager:
        """Ensures singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initializes all component registries."""
        if self._initialized:
            return
        
        # Core component registries
        self.agents: Registry[Any] = Registry("agents")
        self.models: Registry[Any] = Registry("models")
        self.embeddings: Registry[Any] = Registry("embeddings")
        self.memory_backends: Registry[Any] = Registry("memory_backends")
        self.data_loaders: Registry[Any] = Registry("data_loaders")
        self.visualizations: Registry[Any] = Registry("visualizations")
        self.metrics: Registry[Any] = Registry("metrics")
        
        self._initialized = True
        logger.info("RegistryManager initialized with all component registries.")
    
    def get_registry(self, component_type: str) -> Registry[Any]:
        """
        Retrieves a specific registry by component type.
        
        Args:
            component_type: One of "agents", "models", "embeddings", "memory_backends",
                          "data_loaders", "visualizations", "metrics".
            
        Returns:
            The corresponding Registry instance.
            
        Raises:
            ValueError: If component_type is invalid.
        """
        registry_map = {
            "agents": self.agents,
            "models": self.models,
            "embeddings": self.embeddings,
            "memory_backends": self.memory_backends,
            "data_loaders": self.data_loaders,
            "visualizations": self.visualizations,
            "metrics": self.metrics,
        }
        
        if component_type not in registry_map:
            raise ValueError(
                f"Invalid component type '{component_type}'. "
                f"Available types: {list(registry_map.keys())}"
            )
        
        return registry_map[component_type]


# Global singleton instance
registry_manager = RegistryManager()


# ---------------------------------------------------------------------------
# Convenience Decorators
# ---------------------------------------------------------------------------

def register_agent(
    name: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register an agent with the global registry.
    
    Args:
        name: Unique agent identifier.
        dependencies: Required Python packages.
        metadata: Additional agent metadata.
        
    Returns:
        Decorator function.
        
    Example:
        @register_agent("custom_agent", dependencies=["numpy"])
        class CustomAgent:
            pass
    """
    return registry_manager.agents.register(
        name, dependencies=dependencies, metadata=metadata
    )


def register_model(
    name: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register a model with the global registry.
    
    Args:
        name: Unique model identifier.
        dependencies: Required Python packages.
        metadata: Additional model metadata.
        
    Returns:
        Decorator function.
        
    Example:
        @register_model("xgboost", dependencies=["xgboost"])
        class XGBoostModel:
            pass
    """
    return registry_manager.models.register(
        name, dependencies=dependencies, metadata=metadata
    )


def register_embedding(
    name: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register an embedding model with the global registry.
    
    Args:
        name: Unique embedding identifier.
        dependencies: Required Python packages.
        metadata: Additional embedding metadata.
        
    Returns:
        Decorator function.
        
    Example:
        @register_embedding("openai_text_3")
        class OpenAIEmbedding:
            pass
    """
    return registry_manager.embeddings.register(
        name, dependencies=dependencies, metadata=metadata
    )


def register_memory_backend(
    name: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register a memory backend with the global registry.
    
    Args:
        name: Unique backend identifier.
        dependencies: Required Python packages.
        metadata: Additional backend metadata.
        
    Returns:
        Decorator function.
        
    Example:
        @register_memory_backend("chroma", dependencies=["chromadb"])
        class ChromaBackend:
            pass
    """
    return registry_manager.memory_backends.register(
        name, dependencies=dependencies, metadata=metadata
    )


def register_data_loader(
    name: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register a data loader with the global registry.
    
    Args:
        name: Unique loader identifier.
        dependencies: Required Python packages.
        metadata: Additional loader metadata.
        
    Returns:
        Decorator function.
        
    Example:
        @register_data_loader("parquet")
        def load_parquet(path: str):
            pass
    """
    return registry_manager.data_loaders.register(
        name, dependencies=dependencies, metadata=metadata
    )


def register_visualization(
    name: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register a visualization with the global registry.
    
    Args:
        name: Unique visualization identifier.
        dependencies: Required Python packages.
        metadata: Additional visualization metadata.
        
    Returns:
        Decorator function.
        
    Example:
        @register_visualization("correlation_heatmap", dependencies=["plotly"])
        def plot_correlation(df):
            pass
    """
    return registry_manager.visualizations.register(
        name, dependencies=dependencies, metadata=metadata
    )


def register_metric(
    name: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register a metric with the global registry.
    
    Args:
        name: Unique metric identifier.
        dependencies: Required Python packages.
        metadata: Additional metric metadata.
        
    Returns:
        Decorator function.
        
    Example:
        @register_metric("f1_score")
        def calculate_f1(y_true, y_pred):
            pass
    """
    return registry_manager.metrics.register(
        name, dependencies=dependencies, metadata=metadata
    )


# ---------------------------------------------------------------------------
# Lazy Registration Helper
# ---------------------------------------------------------------------------

def register_lazy(
    component_type: str,
    name: str,
    import_path: str,
    *,
    dependencies: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Registers a component for lazy loading using an import path string.
    
    The component will only be imported when first accessed via get().
    
    Args:
        component_type: Type of component (e.g., "agents", "models").
        name: Unique component identifier.
        import_path: Import path in format "module.path:ClassName".
        dependencies: Required Python packages.
        metadata: Additional component metadata.
        
    Example:
        register_lazy(
            "models",
            "lightgbm",
            "src.models.lightgbm:LightGBMModel",
            dependencies=["lightgbm"]
        )
    """
    registry = registry_manager.get_registry(component_type)
    registry.register(name, import_path, dependencies=dependencies, metadata=metadata, lazy=True)


# ---------------------------------------------------------------------------
# Plugin Discovery
# ---------------------------------------------------------------------------

def discover_plugins(package_name: str = "dsa_plugins") -> dict[str, list[str]]:
    """
    Dynamically discovers and loads plugins from a specified package.
    
    Plugins should expose a register() function that registers components
    with the global registry.
    
    Args:
        package_name: Name of the package to scan for plugins.
        
    Returns:
        Dictionary mapping plugin names to lists of registered components.
        
    Example:
        discovered = discover_plugins("my_dsa_plugins")
        # Plugins in my_dsa_plugins/ should have register() functions
    """
    discovered = {}
    
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        logger.warning(f"Plugin package '{package_name}' not found.")
        return discovered
    
    if hasattr(package, "__path__"):
        for _, module_name, _ in importlib.pkgutil.iter_modules(package.__path__):
            try:
                full_module_name = f"{package_name}.{module_name}"
                module = importlib.import_module(full_module_name)
                
                if hasattr(module, "register"):
                    register_func = getattr(module, "register")
                    if callable(register_func):
                        # Get components before registration
                        before = {
                            "agents": set(registry_manager.agents.list()),
                            "models": set(registry_manager.models.list()),
                            "embeddings": set(registry_manager.embeddings.list()),
                        }
                        
                        # Call register function
                        register_func()
                        
                        # Get components after registration
                        after = {
                            "agents": set(registry_manager.agents.list()),
                            "models": set(registry_manager.models.list()),
                            "embeddings": set(registry_manager.embeddings.list()),
                        }
                        
                        # Calculate newly registered components
                        newly_registered = []
                        for comp_type in before:
                            new_comps = after[comp_type] - before[comp_type]
                            newly_registered.extend([f"{comp_type}:{c}" for c in new_comps])
                        
                        discovered[module_name] = newly_registered
                        logger.info(f"Loaded plugin: {module_name} ({len(newly_registered)} components)")
                        
            except Exception as e:
                logger.error(f"Failed to load plugin '{module_name}': {e}")
    
    return discovered
