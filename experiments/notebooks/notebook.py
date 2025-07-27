"""Notebook generation utilities for experiments."""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from experiments.training_job import TrainingJob, TrainingJobConfig
from metta.common.util.config import Config


class AnalysisConfig(Config):
    """Configuration for analysis section of notebook."""
    sps: bool = True


class NotebookConfig(Config):
    """Configuration for notebook generation."""
    setup: bool = True
    state: bool = True
    launch: bool = True
    monitor: bool = True
    analysis: bool = True
    analysis_config: Optional[AnalysisConfig] = None
    replays: bool = True
    scratch: bool = True
    export: bool = True
    simplified: bool = False  # Use simplified notebook with just config & monitor


# Available sections
AVAILABLE_SECTIONS = {
    "setup": "Setup and imports",
    "state": "State management for tracking runs",
    "launch": "Launch training runs",
    "monitor": "Monitor training status",
    "analysis": "Analysis and visualizations",
    "replays": "View MettaScope replays",
    "scratch": "Scratch space for experiments",
    "export": "Export notebook as HTML",
}

# Default sections if none specified
DEFAULT_SECTIONS = ["monitor", "launch", "analysis", "export"]

# Simplified sections for minimal notebook
SIMPLIFIED_SECTIONS = ["setup", "config", "monitor"]


def write_notebook(
    user: str,
    name: str,
    launched_jobs: List[TrainingJob],
    training_job_configs: List[TrainingJobConfig],
    output_dir: Optional[str] = None,
    sections: Optional[List[str]] = None,
) -> str:
    """Write a Jupyter notebook for the experiment using the new API.
    
    Args:
        user: Username running the experiment
        name: Name of the experiment
        launched_jobs: List of launched training jobs
        training_job_configs: List of training job configurations (not yet launched)
        output_dir: Directory to save notebook (defaults to experiments/scratch)
        
    Returns:
        Path to the generated notebook
    """
    # Create metadata
    metadata = {
        "user": user,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "launched_jobs": len(launched_jobs),
        "training_job_configs": len(training_job_configs),
    }
    
    # Add training job configs to metadata if any
    if training_job_configs:
        metadata["training_configs"] = [config.model_dump() for config in training_job_configs]
    
    # Use the existing generate_notebook function with TrainingJob objects
    return generate_notebook(
        name=name,
        description=f"Experiment notebook for {name}",
        sections=sections if sections else DEFAULT_SECTIONS,
        training_jobs=launched_jobs if launched_jobs else None,
        training_job_configs=training_job_configs if training_job_configs else None,
        additional_metadata=metadata,
        output_dir=output_dir,
    )


def generate_notebook(
    name: str,
    description: str = "",
    sections: Optional[List[str]] = None,
    training_jobs: Optional[List[TrainingJob]] = None,
    training_job_configs: Optional[List[TrainingJobConfig]] = None,
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    output_dir: str = "experiments/scratch",
) -> str:
    """Generate a research/experiment notebook.

    Args:
        name: Name for the notebook (will be used in filename)
        description: Optional description of the notebook purpose
        sections: List of sections to include (None = all sections)
        training_jobs: Optional list of TrainingJob objects (preferred)
        wandb_run_names: Optional pre-filled wandb run names (legacy)
        skypilot_job_ids: Optional pre-filled sky job IDs (legacy)
        additional_metadata: Optional metadata to include
        output_dir: Directory to save the notebook

    Returns:
        Path to the generated notebook
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.ipynb"
    filepath = os.path.join(output_dir, filename)

    # Use default sections if none specified
    if sections is None:
        sections = DEFAULT_SECTIONS

    # Create notebook structure
    notebook = {
        "cells": _create_notebook_cells(
            name=name,
            description=description,
            sections=sections,
            training_jobs=training_jobs,
            training_job_configs=training_job_configs,
            wandb_run_names=wandb_run_names,
            skypilot_job_ids=skypilot_job_ids,
            additional_metadata=additional_metadata,
        ),
        "metadata": {
            "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.7",
            },
            "celltoolbar": "Tags",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Write notebook
    with open(filepath, "w") as f:
        json.dump(notebook, f, indent=2)

    if sections != DEFAULT_SECTIONS:
        print(f"Included sections: {', '.join(sections)}")

    return filepath


def generate_notebook_from_template(
    experiment_name: str,
    run_names: List[str],
    sky_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    output_dir: str = "experiments/scratch",
) -> str:
    """Generate a notebook for analyzing experiment runs.

    This is a convenience wrapper for experiments that provides backwards compatibility.

    Args:
        experiment_name: Name of the experiment
        run_names: List of wandb run names
        sky_job_ids: Optional list of corresponding sky job IDs (skypilot_job_ids)
        additional_metadata: Optional additional metadata to include
        output_dir: Directory to save the notebook (default: experiments/scratch)

    Returns:
        Path to the generated notebook
    """
    return generate_notebook(
        name=experiment_name,
        description=f"Analysis notebook for {experiment_name} experiment",
        wandb_run_names=run_names,
        skypilot_job_ids=sky_job_ids,
        additional_metadata=additional_metadata,
        output_dir=output_dir,
    )


def _create_notebook_cells(
    name: str,
    description: str,
    sections: List[str],
    training_jobs: Optional[List[TrainingJob]] = None,
    training_job_configs: Optional[List[TrainingJobConfig]] = None,
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create cells for a notebook based on selected sections."""
    cells = []

    # Title and description (always included)
    title = f"# {name.replace('_', ' ').title()}"
    if description:
        title += f"\n\n{description}"
    title += f"\n\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    cells.append(_create_markdown_cell(title))

    # If we have pre-filled jobs or IDs, add a summary cell
    if training_jobs or wandb_run_names or training_job_configs:
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M')
        user = "Unknown"
        if additional_metadata:
            created_at = additional_metadata.get("created_at", created_at)
            user = additional_metadata.get("user", user)
            
        if training_jobs or training_job_configs:
            # Use TrainingJob objects
            total_jobs = len(training_jobs or []) + len(training_job_configs or [])
            summary = f"""### Experiment Summary

**Experiment**: {name}
**Total Jobs**: {total_jobs}
**Created**: {created_at}
**User**: {user}"""
            
            if training_jobs:
                summary += f"\n\n**Launched Jobs ({len(training_jobs)}):**"
                for job in training_jobs:
                    job_info = f" (job: {job.job_id})" if job.job_id else ""
                    status = "✓" if job.success else "✗" if job.launched else "○"
                    summary += f"\n- {status} {job.name}{job_info}"
                    
            if training_job_configs:
                summary += f"\n\n**Ready to Launch ({len(training_job_configs)}):**"
                for i, config in enumerate(training_job_configs):
                    job_name = f"{name}_job_{i}"
                    summary += f"\n- ⏸️ {job_name}"
        else:
            # Legacy path using wandb_run_names
            summary = f"""### Experiment Summary

**Experiment**: {name}
**Runs**: {len(wandb_run_names)} training runs
**Created**: {created_at}
**User**: {user}"""
            
            if skypilot_job_ids:
                summary += "\n\n**Tracked Jobs:**"
                for i, (job_id, run_name) in enumerate(zip(skypilot_job_ids, wandb_run_names)):
                    summary += f"\n- Job {job_id} → {run_name}"
                    
        cells.append(_create_markdown_cell(summary))

    # Generate the notebook filename we'll use
    notebook_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"

    # Create a single Setup section with all initialization
    cells.append(_create_markdown_cell("## Setup"))

    # Combine setup and state initialization into one section
    setup_cells = _get_setup_section()
    state_cells = _get_state_section(training_jobs, training_job_configs, wandb_run_names, skypilot_job_ids, additional_metadata, name)

    # Mark all setup cells to be in the setup section
    for cell in setup_cells + state_cells:
        if cell["cell_type"] == "code":
            cell["metadata"]["tags"] = cell["metadata"].get("tags", []) + ["setup"]

    cells.extend(setup_cells)
    cells.extend(state_cells)

    # Generate cells for other requested sections
    has_existing_jobs = bool(training_jobs or wandb_run_names)  # True if we have pre-filled jobs
    section_generators = {
        "config": _get_config_section,
        "launch": lambda: _get_launch_section(has_existing_jobs=has_existing_jobs, training_job_configs=training_job_configs),
        "monitor": _get_monitor_section,
        "analysis": _get_analysis_section,
        "replays": _get_replays_section,
        "scratch": _get_scratch_section,
        "export": lambda: _get_export_section(notebook_filename),
    }

    for section in sections:
        # Skip setup and state since we already added them
        if section in section_generators:
            cells.extend(section_generators[section]())

    return cells


def _create_markdown_cell(content: str) -> Dict[str, Any]:
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": content}


def _create_code_cell(content: str, reactive: bool = False) -> Dict[str, Any]:
    """Create a code cell.
    
    Args:
        content: The code content
        reactive: Whether this is a reactive cell (updates automatically)
    """
    metadata = {}
    if reactive:
        metadata["tags"] = ["reactive"]
    return {"cell_type": "code", "execution_count": None, "metadata": metadata, "outputs": [], "source": content}




def _get_setup_section() -> List[Dict[str, Any]]:
    """Generate setup section cells."""
    # Single comprehensive setup cell
    setup_cell = _create_code_cell("""# Initialize notebook
%load_ext autoreload
%autoreload 2

import os
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output

print("✓ Notebook initialized")""")

    return [setup_cell]


def _get_config_section() -> List[Dict[str, Any]]:
    """Generate config section cells."""
    from experiments.notebooks.widgets import config_widget
    return config_widget.generate_cells()


def _get_state_section(
    training_jobs: Optional[List[TrainingJob]] = None,
    training_job_configs: Optional[List[TrainingJobConfig]] = None,
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate state management section cells."""
    # Initialize state with pre-filled data or empty
    if training_jobs or training_job_configs:
        # Convert training jobs to a serializable format for notebook
        jobs_data = []
        if training_jobs:
            for job in training_jobs:
                job_dict = {
                    'name': job.name,
                    'job_id': job.job_id,
                    'launched': job.launched,
                    'success': job.success,
                }
                if hasattr(job, 'notes'):
                    job_dict['notes'] = job.notes
                jobs_data.append(job_dict)
        
        # Convert configs to serializable format
        configs_data = []
        if training_job_configs:
            for config in training_job_configs:
                configs_data.append(config.model_dump())
        
        init_code = f"""# Initialize run tracking
from experiments.notebooks.state import init_state, add_job, add_run, list_runs, kill_all_jobs
from experiments.training_job import TrainingJob, TrainingJobConfig

# Create TrainingJob objects from experiment data
training_jobs = []
for job_data in {jobs_data}:
    job = TrainingJob(name=job_data['name'])
    job.job_id = job_data.get('job_id')
    job.launched = job_data.get('launched', False)
    job.success = job_data.get('success', False)
    if 'notes' in job_data:
        job.notes = job_data['notes']
    training_jobs.append(job)

# Create TrainingJobConfig objects from experiment data
training_job_configs = []
for config_data in {configs_data}:
    config = TrainingJobConfig(**config_data)
    training_job_configs.append(config)

state = init_state(
    training_jobs=training_jobs,
    training_job_configs=training_job_configs,
    metadata={json.dumps(additional_metadata, indent=2) if additional_metadata else "{}"}
)

# These are now dynamic properties that update automatically
wandb_run_names = state.wandb_run_names
skypilot_job_ids = state.skypilot_job_ids
experiments = state.experiments

print(f"✓ Ready. {{len(training_jobs)}} launched jobs, {{len(training_job_configs)}} configs ready to launch")"""    
    else:
        # Legacy initialization or empty state
        init_code = f"""# Initialize run tracking
from experiments.notebooks.state import init_state, add_job, add_run, list_runs, kill_all_jobs
from experiments.training_job import TrainingJob

state = init_state(
    wandb_run_names={wandb_run_names or []},
    skypilot_job_ids={skypilot_job_ids or []},
    metadata={json.dumps(additional_metadata, indent=2) if additional_metadata else "{}"}
)

# These are now dynamic properties that update automatically
wandb_run_names = state.wandb_run_names
skypilot_job_ids = state.skypilot_job_ids
experiments = state.experiments

{f'print("✓ Ready. Tracking {len(wandb_run_names)} runs")' if wandb_run_names else 'print("✓ Ready")'}"""

    return [_create_code_cell(init_code)]


def _get_launch_section(has_existing_jobs: bool = False, training_job_configs: List[TrainingJobConfig] = None) -> List[Dict[str, Any]]:
    """Generate launch section cells."""
    from experiments.notebooks.widgets import launch_widget
    return launch_widget.generate_cells(training_job_configs)


def _get_monitor_section() -> List[Dict[str, Any]]:
    """Generate monitoring section cells."""
    from experiments.notebooks.widgets import skypilot_widget
    return skypilot_widget.generate_cells(state_widget_in_header=True)




def _get_analysis_section() -> List[Dict[str, Any]]:
    """Generate analysis section cells."""
    from experiments.notebooks.widgets import wandb_widget
    return wandb_widget.generate_cells(include_sps=True)


def _get_replays_section() -> List[Dict[str, Any]]:
    """Generate replays section cells."""
    return [
        _create_markdown_cell("## View Replays"),
        _create_code_cell("""# Import replay utilities
from experiments.notebooks.widgets.replay_widget import show_replay, get_available_replays"""),
        _create_code_cell("""# Show last replay for first run
if state.wandb_run_names:
    show_replay(state.wandb_run_names[0], step="last", width=1000, height=600)
else:
    print("No runs tracked yet. Launch some runs first!")"""),
        _create_code_cell("""# Get available replays for first run
if state.wandb_run_names:
    replays = get_available_replays(state.wandb_run_names[0])
    print(f"Available replays for {state.wandb_run_names[0]}:")
    for replay in replays[-10:]:  # Show last 10
        print(f"  {replay['label']} - Step {replay['step']}")"""),
    ]




def _get_scratch_section() -> List[Dict[str, Any]]:
    """Generate scratch space section cells."""
    return [_create_markdown_cell("## Scratch Space"), _create_code_cell("# Quick experiments and one-off analysis\n")]


def _get_export_section(notebook_filename: str) -> List[Dict[str, Any]]:
    """Generate export section cells."""
    from experiments.notebooks.widgets import export_widget
    cells = export_widget.generate_cells()
    
    # Add notebook-specific export cell
    notebook_export_code = f'''# Export this specific notebook
notebook_name = "{notebook_filename}"
print(f"Current notebook: {{notebook_name}}")'''
    
    cells.insert(2, _create_code_cell(notebook_export_code))
    return cells
