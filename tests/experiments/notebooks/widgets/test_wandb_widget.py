"""Tests for the WandbWidget."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from experiments.notebooks.widgets.wandb_widget import WandbWidget, InteractiveWandbWidget
from experiments.notebooks.state import RunState
from experiments.training_job import TrainingJob
from experiments.wandb_service import WandbService


class TestWandbWidget:
    """Test the WandbWidget functionality."""
    
    @pytest.fixture
    def mock_wandb_service(self):
        """Create a mock WandbService."""
        service = Mock(spec=WandbService)
        
        # Mock run data
        mock_run = Mock()
        mock_run.name = "test.run.1"
        mock_run.config = {"learning_rate": 0.001, "batch_size": 32}
        
        service.get_run = Mock(return_value=mock_run)
        service.get_run_config = Mock(return_value={"learning_rate": 0.001, "batch_size": 32})
        service.fetch_metrics_data = Mock(return_value=[])
        
        return service
    
    @pytest.fixture
    def mock_state_with_runs(self):
        """Create a RunState with some tracked runs."""
        # Create a TrainingJob
        job = TrainingJob(name="test.run.1")
        job.job_id = "sky-123"
        job.launched = True
        job.success = True
        
        with patch('experiments.notebooks.state.get_skypilot_service'):
            state = RunState(training_jobs=[job])
            return state
    
    def test_wandb_widget_initialization(self, mock_state_with_runs):
        """Test WandbWidget initialization."""
        widget = WandbWidget(mock_state_with_runs, plot_type="sps")
        
        assert widget.state == mock_state_with_runs
        assert widget.plot_type == "sps"
        assert hasattr(widget, 'output')
        assert hasattr(widget, 'plot_kwargs')
    
    def test_wandb_widget_empty_state(self):
        """Test WandbWidget with no runs."""
        with patch('experiments.notebooks.state.get_skypilot_service'):
            state = RunState()
            widget = WandbWidget(state)
            
            # Should handle empty state gracefully
            # The render method prints when there are no runs
            with patch('builtins.print') as mock_print:
                widget.render()
                mock_print.assert_called_with("No runs to analyze yet.")
    
    def test_interactive_wandb_widget_initialization(self, mock_state_with_runs):
        """Test InteractiveWandbWidget initialization."""
        widget = InteractiveWandbWidget(mock_state_with_runs)
        
        assert widget.state == mock_state_with_runs
        assert hasattr(widget, 'metric_input')
        assert hasattr(widget, 'run_selector')
        assert hasattr(widget, 'plot_type')
        assert hasattr(widget, 'refresh_btn')
        assert hasattr(widget, 'analyze_btn')


class TestWandbService:
    """Test the WandbService functionality."""
    
    @pytest.fixture
    def wandb_service(self):
        """Create a WandbService instance for testing."""
        return WandbService()
    
    def test_wandb_service_exists(self, wandb_service):
        """Test that WandbService can be instantiated."""
        assert wandb_service is not None
        assert hasattr(wandb_service, 'get_run')
        assert hasattr(wandb_service, 'get_run_config')
        
    def test_wandb_service_methods(self, wandb_service):
        """Test that WandbService has expected methods."""
        # Check key methods exist
        assert hasattr(wandb_service, 'api')
        assert hasattr(wandb_service, 'get_run')
        assert hasattr(wandb_service, 'get_run_config')
        assert hasattr(wandb_service, 'fetch_metrics_data')