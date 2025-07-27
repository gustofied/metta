"""Base reactive widget class for Jupyter notebooks.

This module provides the base widget class that automatically updates when the
underlying state changes.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Callable
from IPython.display import display, clear_output
import ipywidgets as widgets


class ReactiveWidget(ABC):
    """Base class for reactive widgets that update when state changes."""

    def __init__(self, state: "RunState", title: Optional[str] = None):
        """Initialize reactive widget.

        Args:
            state: The RunState instance to monitor
            title: Optional title for the widget
        """
        self.state = state
        self.title = title
        self.output = widgets.Output()
        self._additional_callbacks: List[Callable] = []

        # Register for updates
        self.state.register_update_callback(self.update)

    @abstractmethod
    def render(self) -> None:
        """Render the widget content. Must be implemented by subclasses."""
        pass

    def update(self) -> None:
        """Update the widget display."""
        with self.output:
            clear_output(wait=True)
            self.render()

        # Trigger any additional callbacks
        for callback in self._additional_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in callback: {e}")

    def display(self) -> None:
        """Display the widget and perform initial render."""
        if self.title:
            display(widgets.HTML(f"<h3>{self.title}</h3>"))
        display(self.output)
        self.update()

    def add_callback(self, callback: Callable) -> None:
        """Add additional callback to be triggered on updates."""
        self._additional_callbacks.append(callback)


class RefreshButton(widgets.Button):
    """Simple refresh button that triggers updates."""

    def __init__(self, callback: Callable, description: str = "Refresh"):
        """Initialize refresh button.

        Args:
            callback: Function to call when clicked
            description: Button text
        """
        super().__init__(description=description, button_style="info", icon="refresh")
        self.on_click(lambda b: callback())
