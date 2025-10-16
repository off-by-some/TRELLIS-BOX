"""
State manager for TRELLIS UI.
Reactive observable pattern with automatic Streamlit session state sync.

Usage:
    StateManager.uploaded_image = image  # Automatically reactive
    with StateManager.observe(StateManager.uploaded_image):
        # Code re-runs when uploaded_image changes
        pass
"""

import streamlit as st
from typing import (
    Optional, Dict, List, Type, Union, Iterator,
    TypeVar, Generic, ClassVar
)

# Define a type for session state values
SessionValue = Union[None, str, int, float, bool, Dict[str, 'SessionValue'], List['SessionValue']]

from contextlib import contextmanager
from types import TracebackType

T = TypeVar('T')
U = TypeVar('U')


class Observable(Generic[T]):
    """Observable wrapper for reactive state management."""

    def __init__(self, key: str, initial_value: Optional[T] = None) -> None:
        self.key: str = key
        self._value: Optional[T] = initial_value
        self._active_contexts: List['Observable[T]'] = []  # Track active context managers

    @property
    def value(self) -> Optional[T]:
        """Get the current value."""
        return self._value

    def set(self, value: Optional[T]) -> None:
        """Set the value and trigger re-runs of active contexts."""
        if self._value != value:
            self._value = value
            # Trigger re-runs for any active contexts
            for context in self._active_contexts:
                try:
                    st.rerun()
                except Exception:
                    pass  # st might not be available

    def __enter__(self) -> Optional[T]:
        """Enter context manager - add to active contexts."""
        self._active_contexts.append(self)
        return self._value

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        """Exit context manager - remove from active contexts."""
        if self in self._active_contexts:
            self._active_contexts.remove(self)

    def __bool__(self) -> bool:
        """Boolean conversion returns whether value is truthy."""
        return bool(self._value)

    def __str__(self) -> str:
        """String representation."""
        return str(self._value)

    def __repr__(self) -> str:
        """Representation."""
        return f"Observable({self.key!r}, {self._value!r})"

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, Observable):
            return self._value == other._value
        return self._value == other


# Type alias for subscriptable observables (class variables)
Subscriptable = ClassVar[Observable[Optional[T]]]


class StoreMeta(type):
    """Metaclass for Store to intercept class attribute assignment."""

    def __setattr__(cls, name: str, value: object) -> None:
        """Intercept class attribute assignment for observables."""
        # Check if this is an observable class attribute
        if hasattr(cls, name):
            attr = getattr(cls, name)
            if isinstance(attr, Observable):
                # Set the value on the observable
                attr.set(value)
                return

        # Regular attribute assignment
        super().__setattr__(name, value)


class Store(metaclass=StoreMeta):
    """Base store with observable management and context observation."""

    @classmethod
    @contextmanager
    def observe(cls, *observables: Observable) -> Iterator[None]:
        """Context manager for observing multiple observables."""
        # Add all observables to active contexts
        for obs in observables:
            obs._active_contexts.append(obs)
        try:
            yield
        finally:
            # Remove from active contexts
            for obs in observables:
                if obs in obs._active_contexts:
                    obs._active_contexts.remove(obs)

    def to_dict(self) -> Dict[str, SessionValue]:
        """Serialize observable values to a dictionary."""
        result: Dict[str, SessionValue] = {}
        for attr_name in dir(self.__class__):
            attr_value = getattr(self.__class__, attr_name)
            if isinstance(attr_value, Observable):
                result[attr_value.key] = attr_value.value
        return result

    def load_state(self, state_dict: Dict[str, SessionValue]) -> None:
        """Load state from dictionary into observables."""
        # Get all Observable class attributes from this class and its parents
        for attr_name in dir(self.__class__):
            attr_value = getattr(self.__class__, attr_name)
            if isinstance(attr_value, Observable) and attr_value.key in state_dict:
                # Set the value on the observable
                attr_value.set(state_dict[attr_value.key])


class StreamlitStoreMeta(StoreMeta):
    """Metaclass for StreamlitStore to intercept class attribute assignment with session state sync."""

    def __setattr__(cls, name: str, value: object) -> None:
        """Intercept class attribute assignment for observables with session state sync."""
        # Check if this is an observable class attribute
        if hasattr(cls, name):
            attr = getattr(cls, name)
            if isinstance(attr, Observable):
                # Set the value on the observable
                attr.set(value)
                # Sync to session state
                try:
                    st.session_state[attr.key] = value
                except Exception:
                    pass  # st might not be available
                return

        # Regular attribute assignment
        super().__setattr__(name, value)


class StreamlitStore(Store, metaclass=StreamlitStoreMeta):
    """Store implementation with Streamlit session state synchronization."""

    def __init__(self) -> None:
        if hasattr(self, '_initialized'):
            return  # Already initialized

        # Initialize observables with session state values
        self._sync_with_session_state()
        self._initialized: bool = True

    def _sync_with_session_state(self) -> None:
        """Sync observables with Streamlit session state."""
        # Sync ALL observables
        for attr_name in dir(self.__class__):
            attr_value = getattr(self.__class__, attr_name)
            if isinstance(attr_value, Observable):
                # Sync with session state
                key = attr_value.key
                if key in st.session_state:
                    attr_value.set(st.session_state[key])
                else:
                    # Initialize session state with observable's default
                    st.session_state[key] = attr_value.value

    def _process_state_dict(self, state_dict: Dict[str, SessionValue]) -> Dict[str, SessionValue]:
        """Process state dictionary (placeholder for future processing)."""
        return state_dict

    def load_state(self, state_dict: Dict[str, SessionValue]) -> None:
        """Load state from dictionary into observables with Streamlit-specific handling."""
        # Process state dict through cached function
        processed_state = self._process_state_dict(state_dict)

        # Call parent implementation to load into observables
        super().load_state(processed_state)

        # Ensure session state is properly synced (additional Streamlit logic)
        self._sync_with_session_state()


class StateManager(StreamlitStore):
    """Manages Streamlit session state with type safety and observable pattern."""

    # Observable state properties - automatically managed
    # Note: Type system shows Optional[T] but some are guaranteed non-None due to defaults
    pipeline: Subscriptable[object] = Observable("pipeline")
    refiner: Subscriptable[object] = Observable("refiner")
    uploaded_image: Subscriptable[object] = Observable("uploaded_image")
    processed_preview: Subscriptable[object] = Observable("processed_preview")
    processed_preview_size: Subscriptable[object] = Observable("processed_preview_size")
    current_refinement_setting: Subscriptable[bool] = Observable("current_refinement_setting", False)
    refinement_single_input: Subscriptable[bool] = Observable("refinement_single_input", False)
    generated_video: Subscriptable[object] = Observable("generated_video")
    generated_glb: Subscriptable[object] = Observable("generated_glb")
    generated_state: Subscriptable[object] = Observable("generated_state")
    cleanup_counter: Subscriptable[int] = Observable("cleanup_counter", 0)  # Guaranteed non-None
    is_generating: Subscriptable[bool] = Observable("is_generating", False)  # Guaranteed non-None
    resize_width: Subscriptable[int] = Observable("resize_width", 518)  # Guaranteed non-None
    resize_height: Subscriptable[int] = Observable("resize_height", 518)  # Guaranteed non-None
    multi_images: Subscriptable[object] = Observable("multi_images")
    preserved_multi_images: Subscriptable[object] = Observable("_preserved_multi_images")

    _instance: ClassVar[Optional['StateManager']] = None

    def __new__(cls) -> 'StateManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls) -> None:
        """Initialize the StateManager singleton instance."""
        # This will trigger __init__ if not already initialized
        _ = cls()

    def clear_generated_content(self) -> None:
        """Clear all generated content."""
        # Clear observable values (this also clears session state via subscribers)
        StateManager.generated_video.set(None)
        StateManager.generated_glb.set(None)
        StateManager.generated_state.set(None)
        StateManager.uploaded_image.set(None)
        StateManager.processed_preview.set(None)

    def increment_cleanup_counter(self) -> int:
        """Increment and return the cleanup counter."""
        counter = StateManager.cleanup_counter.value
        counter += 1
        if counter > 1000:
            counter = 0
        StateManager.cleanup_counter.set(counter)
        return counter
