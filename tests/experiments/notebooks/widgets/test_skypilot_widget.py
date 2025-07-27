"""Tests for the SkypilotWidget."""

import pytest
from unittest.mock import Mock, patch
from experiments.notebooks.widgets.skypilot_widget import SkypilotWidget
from experiments.notebooks.state import RunState
from experiments.training_job import TrainingJob
from experiments.skypilot_service import SkypilotService


class TestSkypilotWidget:
    """Test the SkypilotWidget functionality."""
    
    @pytest.fixture
    def mock_skypilot_service(self):
        """Create a mock SkypilotService with some jobs."""
        service = Mock(spec=SkypilotService)
        
        # Create some test jobs
        job1 = TrainingJob(name="test.run.1")
        job1.job_id = "sky-123"
        job1.launched = True
        job1.success = True
        
        job2 = TrainingJob(name="test.run.2")
        job2.job_id = "sky-456"
        job2.launched = True
        job2.success = True
        job2.notes = "Pre-loaded from experiment"
        
        service._tracked_jobs = {"sky-123": job1, "sky-456": job2}
        service.get_tracked_jobs = Mock(return_value=[job1, job2])
        service.add_job = Mock()
        service.cancel_job = Mock(return_value=True)
        
        return service
    
    @pytest.fixture
    def mock_state_with_jobs(self, mock_skypilot_service):
        """Create a RunState with some tracked jobs."""
        # Ensure we don't create real SkypilotService instances
        with patch('experiments.skypilot_service._skypilot_service', mock_skypilot_service):
            with patch('experiments.notebooks.state.get_skypilot_service', return_value=mock_skypilot_service):
                state = RunState()
                state.service = mock_skypilot_service
                return state
    
    def test_skypilot_widget_initialization(self, mock_state_with_jobs):
        """Test SkypilotWidget initialization."""
        widget = SkypilotWidget(mock_state_with_jobs, show_all_sky_jobs=False)
        
        assert widget.state == mock_state_with_jobs
        assert widget.show_all_sky_jobs is False
        assert hasattr(widget, 'toggle_btn')
        assert hasattr(widget, 'refresh_btn')
        assert hasattr(widget, 'cancel_btn')
        assert hasattr(widget, 'styles')
        assert widget.selected_jobs == []
        assert widget.job_checkboxes == {}
    
    def test_skypilot_widget_empty_state(self):
        """Test SkypilotWidget with no jobs."""
        mock_service = Mock(spec=SkypilotService)
        mock_service.get_tracked_jobs = Mock(return_value=[])
        
        with patch('experiments.skypilot_service._skypilot_service', mock_service):
            with patch('experiments.notebooks.state.get_skypilot_service', return_value=mock_service):
                state = RunState()
                state.service = mock_service
                widget = SkypilotWidget(state)
            
            # Patch display at the module level where it's imported
            with patch('experiments.notebooks.widgets.skypilot_widget.display') as mock_display:
                widget.render()
                
                # Should have called display with HTML widgets
                assert mock_display.called
                
                # Find the HTML widget with the empty state message
                for call in mock_display.call_args_list:
                    args = call[0]
                    if args:
                        widget_obj = args[0]
                        if hasattr(widget_obj, 'value') and isinstance(widget_obj.value, str):
                            if "No Active Training Jobs" in widget_obj.value:
                                return  # Test passes
                
                # If we get here, the expected content wasn't found
                assert False, "Expected 'No Active Training Jobs' in displayed HTML"
    
    def test_skypilot_widget_toggle_functionality(self, mock_state_with_jobs):
        """Test toggle between session and all Sky jobs."""
        widget = SkypilotWidget(mock_state_with_jobs, show_all_sky_jobs=False)
        
        # Initial state
        assert widget.show_all_sky_jobs is False
        
        # Simulate toggle
        widget.toggle_btn.value = True
        widget._on_toggle_change({'new': True})
        
        assert widget.show_all_sky_jobs is True
    
    def test_skypilot_widget_shows_costs(self, mock_state_with_jobs):
        """Test that SkypilotWidget displays cost information."""
        import pandas as pd
        
        # Mock Sky jobs data with costs
        sky_jobs_df = pd.DataFrame({
            'ID': ['sky-123', 'sky-456', 'sky-789'],
            'NAME': ['test.run.1', 'test.run.2', 'other.run'],
            'STATUS': ['RUNNING', 'SUCCEEDED', 'FAILED'],
            'RESOURCES': ['1x A100:8', '2x V100:4', '1x T4:1'],
            'JOB DURATION': ['2h 30m', '1h 15m', '45m'],
            'EST_COST': ['$40.00', '$15.30', '$0.39'],
            'EST_COST_USD': [40.00, 15.30, 0.39]
        })
        
        # Mock the service's get_sky_jobs_data method
        mock_state_with_jobs.service.get_sky_jobs_data = Mock(return_value=sky_jobs_df)
        
        widget = SkypilotWidget(mock_state_with_jobs, show_all_sky_jobs=True)
        
        with patch('experiments.notebooks.widgets.skypilot_widget.display') as mock_display:
            widget.render()
            
            # Check that cost summary is displayed
            display_calls = str(mock_display.call_args_list)
            assert "Total Cost" in display_calls
            assert "$55.69" in display_calls  # Total of all costs


class TestSkypilotWidgetSelection:
    """Test the job selection functionality in SkypilotWidget."""
    
    @pytest.fixture
    def state_with_multiple_jobs(self):
        """Create state with multiple jobs for testing."""
        service = Mock(spec=SkypilotService)
        
        jobs = []
        for i in range(3):
            job = TrainingJob(name=f"test.run.{i}")
            job.job_id = f"sky-{i:03d}"
            job.launched = True
            job.success = True
            jobs.append(job)
        
        service.get_tracked_jobs = Mock(return_value=jobs)
        service.get_job_by_id = Mock(side_effect=lambda jid: next((j for j in jobs if j.job_id == jid), None))
        service.cancel_job = Mock(return_value=True)
        
        with patch('experiments.skypilot_service._skypilot_service', service):
            with patch('experiments.notebooks.state.get_skypilot_service', return_value=service):
                state = RunState()
                state.service = service
                # Manually set the training jobs
                state._training_jobs = jobs
                return state
    
    def test_skypilot_widget_job_selection(self, state_with_multiple_jobs):
        """Test job selection in SkypilotWidget."""
        widget = SkypilotWidget(state_with_multiple_jobs)
        
        # Initially no jobs selected
        assert widget.selected_jobs == []
        assert widget.cancel_btn.disabled is True
        
        # Simulate selecting jobs
        widget.selected_jobs = ["sky-000", "sky-002"]
        widget._on_job_selection_change({'new': True}, "sky-000")
        widget._on_job_selection_change({'new': True}, "sky-002")
        
        assert len(widget.selected_jobs) == 2
        assert "sky-000" in widget.selected_jobs
        assert "sky-002" in widget.selected_jobs
        assert widget.cancel_btn.disabled is False
    
    @pytest.mark.skip(reason="This test is slow due to widget rendering")
    def test_skypilot_widget_cancel_selected(self, state_with_multiple_jobs):
        """Test cancelling selected jobs."""
        widget = SkypilotWidget(state_with_multiple_jobs)
        
        # Select some jobs
        widget.selected_jobs = ["sky-000", "sky-002"]
        widget.cancel_btn.disabled = False
        
        # Call cancel
        widget._cancel_selected(None)
        
        # Should have called cancel for each selected job
        assert state_with_multiple_jobs.service.cancel_job.call_count == 2
        state_with_multiple_jobs.service.cancel_job.assert_any_call("sky-000")
        state_with_multiple_jobs.service.cancel_job.assert_any_call("sky-002")
        
        # Selected jobs should be cleared
        assert widget.selected_jobs == []
        assert widget.cancel_btn.disabled is True