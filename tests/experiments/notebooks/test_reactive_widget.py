"""Tests for the base ReactiveWidget class."""

import pytest
from unittest.mock import Mock, patch
from experiments.notebooks.widget import ReactiveWidget
from experiments.notebooks.state import RunState
from experiments.skypilot_service import SkypilotService


class ConcreteReactiveWidget(ReactiveWidget):
    """Concrete implementation for testing abstract base class."""
    
    def __init__(self, state, title="Test Widget"):
        super().__init__(state, title)
        self.render_count = 0
        self.last_render_content = None
    
    def render(self):
        """Track render calls."""
        self.render_count += 1
        self.last_render_content = f"Rendered {self.render_count} times"
        print(self.last_render_content)


class TestReactiveWidget:
    """Test the ReactiveWidget base class functionality."""
    
    @pytest.fixture
    def mock_skypilot_service(self):
        """Create a mock SkypilotService."""
        service = Mock(spec=SkypilotService)
        service._tracked_jobs = {}
        service._jobs_by_name = {}
        service.get_tracked_jobs = Mock(return_value=[])
        service.add_job = Mock()
        service.cancel_job = Mock(return_value=True)
        return service
    
    @pytest.fixture
    def mock_state(self, mock_skypilot_service):
        """Create a mock RunState with mocked service."""
        with patch('experiments.notebooks.state.get_skypilot_service', return_value=mock_skypilot_service):
            state = RunState()
            state.service = mock_skypilot_service
            return state
    
    def test_reactive_widget_initialization(self, mock_state):
        """Test that ReactiveWidget initializes correctly."""
        widget = ConcreteReactiveWidget(mock_state, title="Test")
        
        assert widget.state == mock_state
        assert widget.title == "Test"
        assert widget.render_count == 0
        assert hasattr(widget, 'output')
        assert hasattr(widget, '_additional_callbacks')
    
    def test_reactive_widget_registers_callback(self, mock_state):
        """Test that widget registers itself for state updates."""
        # Track registered callbacks
        registered_callbacks = []
        mock_state.register_update_callback = Mock(side_effect=lambda cb: registered_callbacks.append(cb))
        
        widget = ConcreteReactiveWidget(mock_state)
        
        # Should have registered its update method
        assert mock_state.register_update_callback.called
        assert len(registered_callbacks) == 1
        assert registered_callbacks[0] == widget.update
    
    def test_reactive_widget_update_triggers_render(self, mock_state):
        """Test that update() triggers render()."""
        widget = ConcreteReactiveWidget(mock_state)
        
        # Initial state
        assert widget.render_count == 0
        
        # Trigger update
        widget.update()
        
        # Should have rendered
        assert widget.render_count == 1
        assert widget.last_render_content == "Rendered 1 times"
    
    def test_reactive_widget_additional_callbacks(self, mock_state):
        """Test that additional callbacks are triggered on update."""
        widget = ConcreteReactiveWidget(mock_state)
        
        # Add some callbacks
        callback1_called = False
        callback2_called = False
        
        def callback1():
            nonlocal callback1_called
            callback1_called = True
        
        def callback2():
            nonlocal callback2_called
            callback2_called = True
        
        widget.add_callback(callback1)
        widget.add_callback(callback2)
        
        # Trigger update
        widget.update()
        
        # All callbacks should have been called
        assert callback1_called
        assert callback2_called
        assert widget.render_count == 1
    
    def test_reactive_widget_callback_error_handling(self, mock_state):
        """Test that callback errors don't break the update process."""
        widget = ConcreteReactiveWidget(mock_state)
        
        # Add a failing callback
        def bad_callback():
            raise ValueError("Test error")
        
        widget.add_callback(bad_callback)
        
        # Update should not raise
        widget.update()
        
        # Render should still have happened
        assert widget.render_count == 1