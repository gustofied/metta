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
DEFAULT_SECTIONS = ["setup", "state", "monitor", "config"]

# Simplified sections for minimal notebook
SIMPLIFIED_SECTIONS = ["setup", "state", "config", "monitor"]


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
        metadata["training_configs"] = [
            config.model_dump() for config in training_job_configs
        ]

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
            "kernelspec": {
                "display_name": ".venv",
                "language": "python",
                "name": "python3",
            },
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
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
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
                for i, (job_id, run_name) in enumerate(
                    zip(skypilot_job_ids, wandb_run_names)
                ):
                    summary += f"\n- Job {job_id} → {run_name}"

        cells.append(_create_markdown_cell(summary))

    # Generate the notebook filename we'll use
    notebook_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"

    # Create a single Setup section with all initialization
    cells.append(_create_markdown_cell("## Setup"))

    # Get the setup code with data
    setup_code = _get_combined_setup_code(
        training_jobs, training_job_configs, additional_metadata
    )
    setup_cell = _create_code_cell(setup_code)
    setup_cell["metadata"]["tags"] = ["setup"]
    cells.append(setup_cell)

    # Generate cells for other requested sections
    has_existing_jobs = bool(
        training_jobs or wandb_run_names
    )  # True if we have pre-filled jobs
    section_generators = {
        "config": _get_config_section,
        "launch": lambda: _get_launch_section(
            has_existing_jobs=has_existing_jobs,
            training_job_configs=training_job_configs,
        ),
        "monitor": _get_monitor_section,
        "analysis": _get_analysis_section,
        "replays": _get_replays_section,
        "scratch": _get_scratch_section,
        "export": lambda: _get_export_section(notebook_filename),
    }

    for section in sections:
        # Skip setup since we already added it
        if section in section_generators and section not in ["setup", "state"]:
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
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": content,
    }


def _get_combined_setup_code(
    training_jobs: Optional[List[TrainingJob]] = None,
    training_job_configs: Optional[List[TrainingJobConfig]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate combined setup code in a single cell."""
    # Create a JSON representation that can be passed directly
    experiment_data = {"jobs": [], "configs": [], "metadata": additional_metadata or {}}

    if training_jobs:
        for job in training_jobs:
            job_dict = {
                "name": job.name,
                "job_id": job.job_id,
                "launched": job.launched,
                "success": job.success,
            }
            if hasattr(job, "notes"):
                job_dict["notes"] = job.notes
            experiment_data["jobs"].append(job_dict)

    if training_job_configs:
        for config in training_job_configs:
            experiment_data["configs"].append(config.model_dump())

    return f"""from experiments.notebooks.state import (
    setup_notebook, print_configs, print_jobs, launch_all, kill_all,
    print_wandb_runs, plot_sps, plot_metrics, 
    list_replays, show_replay, export_notebook, reset_configs
)
from experiments.training_job import TrainingJobConfig

state, configs = setup_notebook(experiment_data={repr(experiment_data)})"""


def _get_config_section() -> List[Dict[str, Any]]:
    """Generate config section cells."""
    cells = []

    # Section header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 📝 Training Configuration",
        }
    )

    # Cell 1: Print current configs
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "print_configs(configs)",
        }
    )

    # Cell 2: Edit configs and print again
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Edit configs directly in Python
configs[0].gpus = 4
configs[0].curriculum = 'arena'
configs[0].wandb_tags = ['experiment', 'test']

# View updated configs
print_configs(configs)

# Reset to original if needed
# reset_configs(state)""",
        }
    )

    # Cell 3: Launch jobs
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Launch all configured jobs
launch_all(state)""",
        }
    )

    return cells


def _get_launch_section(
    has_existing_jobs: bool = False,
    training_job_configs: List[TrainingJobConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate launch section cells."""
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 🚀 Launch Jobs"
    })
    
    # Launch instructions
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Create and launch a single job
from experiments.training_job import TrainingJob

config = TrainingJobConfig(
    curriculum='env/mettagrid/arena/basic',
    gpus=1,
    nodes=1,
    spot=True,
    wandb_tags=['experiment', 'test']
)

job = TrainingJob(name='my_experiment', config=config)
if job.launch():
    state.add_job(job)
    print(f"✅ Launched {job.name} with ID: {job.job_id}")
else:
    print("❌ Launch failed")"""
    })
    
    # Batch launch
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Or use the config system
configs.append(TrainingJobConfig(curriculum='arena'))
configs.append(TrainingJobConfig(curriculum='harvest', gpus=2))

# Then launch all
launch_all(state)"""
    })
    
    return cells


def _get_monitor_section() -> List[Dict[str, Any]]:
    """Generate monitoring section cells."""
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 🚀 Monitor & Control Training Jobs"
    })
    
    # Status display
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "print_jobs(state)"
    })
    
    # Launch controls
    cells.append({
        "cell_type": "code", 
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Launch all configs
launch_all(state)

# Launch specific config
# job = TrainingJob(name='my_job', config=configs[0])
# job.launch()
# state.add_job(job)"""
    })
    
    # Kill controls
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Kill all running jobs
kill_all(state)

# Kill specific job
# state.service.cancel_job('job_id_here')"""
    })
    
    return cells


def _get_analysis_section() -> List[Dict[str, Any]]:
    """Generate analysis section cells."""
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 📊 Analysis"
    })
    
    # Show available runs
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "print_wandb_runs(state)"
    })
    
    # Plot SPS
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Plot steps per second for all runs
plot_sps(state)

# Plot specific runs
# plot_sps(state, run_indices=[0, 1])"""
    })
    
    # Plot custom metrics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Plot any metric
plot_metrics(state, 'overview/reward')

# Other useful metrics:
# plot_metrics(state, 'trainer/loss')
# plot_metrics(state, 'trainer/explained_variance')
# plot_metrics(state, 'eval/mean_reward', smooth=0.95)"""
    })
    
    return cells


def _get_replays_section() -> List[Dict[str, Any]]:
    """Generate replays section cells."""
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 🎬 Replays"
    })
    
    # List available replays
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# List replays for first run
list_replays(state, run_index=0)

# List for specific run
# list_replays(state, run_index=2)"""
    })
    
    # Show replay
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Show latest replay
show_replay(state, run_index=0, step='last')

# Show specific step
# show_replay(state, run_index=0, step=100000)

# Larger display
# show_replay(state, run_index=0, step='last', width=1200, height=800)"""
    })
    
    return cells


def _get_scratch_section() -> List[Dict[str, Any]]:
    """Generate scratch space section cells."""
    return [
        _create_markdown_cell("## Scratch Space"),
        _create_code_cell("# Quick experiments and one-off analysis\n"),
    ]


def _get_export_section(notebook_filename: str) -> List[Dict[str, Any]]:
    """Generate export section cells."""
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 📤 Export"
    })
    
    # Export instructions
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": f"""# Export this notebook
export_notebook()

# Current notebook: {notebook_filename}"""
    })
    
    return cells
