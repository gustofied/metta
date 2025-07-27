"""Tests for the ReplayWidget and replay functions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from IPython.display import IFrame
from experiments.notebooks.widgets.replay_widget import (
    ReplayWidget, show_replay, get_available_replays
)
from experiments.notebooks.state import RunState
from experiments.training_job import TrainingJob
from experiments.wandb_service import WandbService


class TestReplayWidget:
    """Test the ReplayWidget functionality."""
    
    @pytest.fixture
    def mock_wandb_service(self):
        """Create a mock WandbService."""
        service = Mock(spec=WandbService)
        
        # Mock run with replay artifacts
        mock_run = Mock()
        mock_run.name = "test.run.1"
        mock_run.entity = "test_entity"
        mock_run.project = "test_project"
        
        # Mock artifacts
        mock_artifact = Mock()
        mock_artifact.name = "replays-test.run.1:v0"
        mock_artifact.files = Mock(return_value=[
            Mock(name="replay_1000.html"),
            Mock(name="replay_2000.html"),
            Mock(name="replay_3000.html"),
        ])
        
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        service.get_run = Mock(return_value=mock_run)
        
        return service
    
    @pytest.fixture
    def mock_state_with_runs(self):
        """Create a RunState with some tracked runs."""
        job = TrainingJob(name="test.run.1")
        job.job_id = "sky-123"
        job.launched = True
        job.success = True
        
        with patch('experiments.notebooks.state.get_skypilot_service'):
            state = RunState(training_jobs=[job])
            return state
    
    def test_replay_widget_initialization(self, mock_state_with_runs):
        """Test ReplayWidget initialization."""
        widget = ReplayWidget(mock_state_with_runs)
        
        assert widget.state == mock_state_with_runs
        assert hasattr(widget, 'run_selector')
        assert hasattr(widget, 'step_selector')
        assert hasattr(widget, 'width')
        assert hasattr(widget, 'height')
        assert hasattr(widget, 'load_btn')
    
    def test_replay_widget_empty_state(self):
        """Test ReplayWidget with no runs."""
        with patch('experiments.notebooks.state.get_skypilot_service'):
            state = RunState()
            widget = ReplayWidget(state)
            
            # Should show empty state message or have a placeholder
            # If no runs, widget might have a default option
            assert len(widget.run_selector.options) <= 1  # Empty or placeholder
            assert widget.load_btn.disabled is False  # Button might still be enabled


class TestReplayFunctions:
    """Test the standalone replay functions."""
    
    @pytest.fixture
    def mock_wandb_service(self):
        """Create a mock WandbService with replay data."""
        service = Mock(spec=WandbService)
        
        # Mock run
        mock_run = Mock()
        mock_run.name = "test.run.1"
        mock_run.entity = "test_entity"
        mock_run.project = "test_project"
        
        # Mock artifact with replay files
        mock_artifact = Mock()
        mock_artifact.name = "replays-test.run.1:v0"
        mock_artifact.files = Mock(return_value=[
            Mock(name="replay_1000.html"),
            Mock(name="replay_2000.html"),
            Mock(name="replay_last.html"),
        ])
        
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_run.logged_artifacts = Mock(return_value=[mock_artifact])
        
        service.get_run = Mock(return_value=mock_run)
        
        return service
    
    def test_get_available_replays(self, mock_wandb_service):
        """Test getting available replays for a run."""
        # Mock run.files() to return a list
        mock_file1 = Mock()
        mock_file1.name = "media/html/replays/link_1000_.html"
        # Mock the download method to return HTML content
        mock_download_file1 = Mock()
        mock_download_file1.read.return_value = '<a href="http://example.com/replay1">Replay 1</a>'
        mock_download_file1.__enter__ = Mock(return_value=mock_download_file1)
        mock_download_file1.__exit__ = Mock(return_value=None)
        mock_file1.download = Mock(return_value=mock_download_file1)
        
        mock_file2 = Mock()
        mock_file2.name = "media/html/replays/link_2000_.html"
        mock_download_file2 = Mock()
        mock_download_file2.read.return_value = '<a href="http://example.com/replay2">Replay 2</a>'
        mock_download_file2.__enter__ = Mock(return_value=mock_download_file2)
        mock_download_file2.__exit__ = Mock(return_value=None)
        mock_file2.download = Mock(return_value=mock_download_file2)
        
        mock_file3 = Mock()
        mock_file3.name = "media/html/replays/link_3000_.html"
        mock_download_file3 = Mock()
        mock_download_file3.read.return_value = '<a href="http://example.com/replay3">Replay 3</a>'
        mock_download_file3.__enter__ = Mock(return_value=mock_download_file3)
        mock_download_file3.__exit__ = Mock(return_value=None)
        mock_file3.download = Mock(return_value=mock_download_file3)
        
        mock_files = [mock_file1, mock_file2, mock_file3]
        mock_wandb_service.get_run.return_value.files = Mock(return_value=mock_files)
        
        with patch('experiments.notebooks.widgets.replay_widget.WandbService', return_value=mock_wandb_service):
            replays = get_available_replays("test.run.1")
            
            assert len(replays) == 3
            assert replays[0]["step"] == 1000
            assert replays[1]["step"] == 2000
            assert replays[2]["step"] == 3000
    
    def test_get_available_replays_no_artifacts(self, mock_wandb_service):
        """Test getting replays when no artifacts exist."""
        # Mock run.files() to return empty list
        mock_wandb_service.get_run.return_value.files = Mock(return_value=[])
        
        with patch('experiments.notebooks.widgets.replay_widget.WandbService', return_value=mock_wandb_service):
            replays = get_available_replays("test.run.1")
            
            assert replays == []
    
    def test_show_replay(self, mock_wandb_service):
        """Test showing a replay."""
        # Mock run.files() to return a replay file
        mock_file = Mock()
        mock_file.name = "media/html/replays/link_1000_.html"
        # Mock the download method to return HTML content
        mock_download_file = Mock()
        mock_download_file.read.return_value = '<a href="http://example.com/replay1">Replay 1</a>'
        mock_download_file.__enter__ = Mock(return_value=mock_download_file)
        mock_download_file.__exit__ = Mock(return_value=None)
        mock_file.download = Mock(return_value=mock_download_file)
        mock_files = [mock_file]
        mock_wandb_service.get_run.return_value.files = Mock(return_value=mock_files)
        
        with patch('experiments.notebooks.widgets.replay_widget.WandbService', return_value=mock_wandb_service):
            with patch('experiments.notebooks.widgets.replay_widget.display') as mock_display:
                show_replay("test.run.1", step="1000", width=800, height=600)
                
                # Should have called display with an IFrame
                assert mock_display.called
                iframe = mock_display.call_args[0][0]
                assert isinstance(iframe, IFrame)
                assert iframe.width == 800
                assert iframe.height == 600
    
