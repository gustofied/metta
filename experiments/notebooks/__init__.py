"""Notebook utilities for experiments."""

# Import commonly used functions for easy access
from experiments.notebooks.notebook import generate_notebook_from_template, write_notebook
from experiments.notebooks.widgets.replay_widget import show_replay, get_available_replays

# Import from experiments package
from experiments.wandb_service import WandbService

__all__ = [
    # Notebook generation
    "generate_notebook_from_template",
    "write_notebook",
    # Replays
    "show_replay",
    "get_available_replays",
    # Services
    "WandbService",
]
