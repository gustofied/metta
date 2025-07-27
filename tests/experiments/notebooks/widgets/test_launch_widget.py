"""Tests for the LaunchWidget."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from experiments.notebooks.widgets.launch_widget import LaunchWidget, MultiLaunchWidget
from experiments.notebooks.state import RunState
from experiments.training_job import TrainingJob
import ipywidgets as widgets


class TestLaunchWidget:
    """Test the LaunchWidget functionality."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock RunState."""
        with patch('experiments.notebooks.state.get_skypilot_service'):
            state = RunState()
            state.add_job = Mock()
            return state
    
    def test_launch_widget_initialization(self, mock_state):
        """Test LaunchWidget initialization."""
        widget = LaunchWidget(mock_state)
        
        assert widget.state == mock_state
        assert hasattr(widget, 'run_name')
        assert hasattr(widget, 'curriculum')
        assert hasattr(widget, 'gpus')
        assert hasattr(widget, 'launch_button')
        assert hasattr(widget, 'status_output')
    
    def test_launch_widget_default_values(self, mock_state):
        """Test LaunchWidget has sensible defaults."""
        widget = LaunchWidget(mock_state)
        
        # Check default values
        assert widget.curriculum.value == "env/mettagrid/arena/basic"
        assert widget.gpus.value == 1
        assert widget.nodes.value == 1
        assert widget.spot.value is True
        assert widget.skip_git.value is False
    
    def test_launch_widget_button_enabled(self, mock_state):
        """Test that LaunchWidget launch button is properly configured."""
        widget = LaunchWidget(mock_state)
        
        # Launch button should be enabled
        assert widget.launch_button.disabled is False
        assert widget.launch_button.description == '🚀 Launch Training'
        assert widget.launch_button.button_style == 'primary'


class TestMultiLaunchWidget:
    """Test the MultiLaunchWidget functionality."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock RunState."""
        with patch('experiments.notebooks.state.get_skypilot_service'):
            state = RunState()
            state.add_job = Mock()
            return state
    
    def test_multi_launch_widget_initialization(self, mock_state):
        """Test MultiLaunchWidget initialization."""
        widget = MultiLaunchWidget(mock_state)
        
        assert widget.state == mock_state
        # Should have additional multi-run controls
        assert hasattr(widget, 'num_runs')
        assert hasattr(widget, 'launch_button')
        # Should inherit from LaunchWidget
        assert hasattr(widget, 'run_name')
        assert hasattr(widget, 'curriculum')
    
    def test_multi_launch_widget_num_runs(self, mock_state):
        """Test MultiLaunchWidget num_runs slider."""
        widget = MultiLaunchWidget(mock_state)
        
        # Check num_runs slider configuration
        assert widget.num_runs.value == 1
        assert widget.num_runs.min == 1
        assert widget.num_runs.max == 10