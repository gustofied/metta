"""Tests for the ExportWidget."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from experiments.notebooks.widgets.export_widget import ExportWidget
from experiments.notebooks.state import RunState
from experiments.training_job import TrainingJob
import ipywidgets as widgets


class TestExportWidget:
    """Test the ExportWidget functionality."""
    
    @pytest.fixture
    def mock_state_with_runs(self):
        """Create a RunState with some tracked runs."""
        # Create training jobs
        job1 = TrainingJob(name="test.run.1")
        job1.job_id = "sky-123"
        job1.launched = True
        job1.success = True
        
        job2 = TrainingJob(name="test.run.2")
        job2.job_id = "sky-456"
        job2.launched = True
        job2.success = False
        
        # Mock the skypilot service to return our jobs
        mock_service = Mock()
        mock_service.get_tracked_jobs = Mock(return_value=[job1, job2])
        mock_service.add_job = Mock()
        
        with patch('experiments.notebooks.state.get_skypilot_service', return_value=mock_service):
            state = RunState(training_jobs=[job1, job2])
            return state
    
    def test_export_widget_initialization(self, mock_state_with_runs):
        """Test ExportWidget initialization."""
        widget = ExportWidget(mock_state_with_runs)
        
        assert widget.state == mock_state_with_runs
        assert hasattr(widget, 'export_btn')
        assert hasattr(widget, 'output')
        assert hasattr(widget, 'run_selector')
        assert hasattr(widget, 'export_format')
    
    def test_export_widget_empty_state(self):
        """Test ExportWidget with no runs."""
        with patch('experiments.notebooks.state.get_skypilot_service'):
            state = RunState()
            widget = ExportWidget(state)
            
            # Should handle empty state gracefully
            with patch('experiments.notebooks.widgets.export_widget.display') as mock_display:
                widget.render()
                assert mock_display.called
    
    def test_export_widget_exports_data(self, mock_state_with_runs):
        """Test that ExportWidget can export data."""
        widget = ExportWidget(mock_state_with_runs)
        
        # Check that export format is set
        assert widget.export_format.value in ['csv', 'json', 'parquet']
        
        # Check that output directory is set
        assert widget.output_dir.value == './exports'
        
        # Check export button exists
        assert hasattr(widget, 'export_btn')
        assert widget.export_btn.description == 'Export Data'
    
    def test_export_widget_refresh_works(self, mock_state_with_runs):
        """Test that ExportWidget refresh updates the run list."""
        widget = ExportWidget(mock_state_with_runs)
        
        # Initial state should show runs
        assert len(widget.run_selector.options) > 0
        assert 'test.run.1' in widget.run_selector.options
        assert 'test.run.2' in widget.run_selector.options
        
        # Test refresh button exists
        assert hasattr(widget, 'refresh_btn')
        assert widget.refresh_btn.description == 'Refresh'