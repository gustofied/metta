"""Tests for notebook generation with pythonic interface."""

import json
import os
import tempfile

from experiments.notebooks.notebook import generate_notebook, generate_notebook_from_template
from experiments.training_job import TrainingJob, TrainingJobConfig


class TestNotebookGeneration:
    """Test notebook generation produces expected outcomes."""

    def test_generates_valid_jupyter_notebook(self):
        """Test that generated file is a valid Jupyter notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(name="test_notebook", output_dir=tmpdir)

            # File should exist
            assert os.path.exists(filepath)
            assert filepath.endswith(".ipynb")

            # Should be valid JSON
            with open(filepath) as f:
                notebook = json.load(f)

            # Should have notebook structure
            assert "cells" in notebook
            assert "metadata" in notebook
            assert "nbformat" in notebook
            assert notebook["nbformat"] == 4

    def test_setup_returns_state_and_configs(self):
        """Test that setup_notebook returns (state, configs) tuple."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(name="test_setup", output_dir=tmpdir)

            with open(filepath) as f:
                notebook = json.load(f)

            # Find setup cell
            setup_cell = None
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and "setup_notebook" in "".join(cell["source"]):
                    setup_cell = "".join(cell["source"])
                    break

            assert setup_cell is not None
            # Should return state and configs
            assert "state, configs = setup_notebook" in setup_cell

    def test_imports_print_based_functions(self):
        """Test that notebook imports print-based functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(name="test_imports", output_dir=tmpdir)

            with open(filepath) as f:
                notebook = json.load(f)

            # Find import cell
            import_cell = None
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and "from experiments.notebooks.state import" in "".join(cell["source"]):
                    import_cell = "".join(cell["source"])
                    break

            assert import_cell is not None
            # Should import all print-based functions
            assert "print_configs" in import_cell
            assert "print_jobs" in import_cell
            assert "launch_all" in import_cell
            assert "kill_all" in import_cell
            assert "print_wandb_runs" in import_cell
            assert "plot_sps" in import_cell
            assert "plot_metrics" in import_cell
            assert "list_replays" in import_cell
            assert "show_replay" in import_cell
            assert "export_notebook" in import_cell
            assert "reset_configs" in import_cell
            # Should NOT import widget classes
            assert "Widget" not in import_cell

    def test_section_order_monitor_before_config(self):
        """Test that monitor section comes before config section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use default sections
            filepath = generate_notebook(name="test_order", output_dir=tmpdir)

            with open(filepath) as f:
                notebook = json.load(f)

            # Find section positions
            monitor_pos = None
            config_pos = None
            
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "markdown":
                    content = "".join(cell["source"])
                    if "Monitor & Control" in content:
                        monitor_pos = i
                    elif "Training Configuration" in content:
                        config_pos = i

            # Monitor should come before config
            assert monitor_pos is not None
            assert config_pos is not None
            assert monitor_pos < config_pos

    def test_config_section_has_three_cells(self):
        """Test that config section has exactly 3 cells as requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(name="test_config_cells", output_dir=tmpdir)

            with open(filepath) as f:
                notebook = json.load(f)

            # Find config section
            config_section_start = None
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "markdown" and "Training Configuration" in "".join(cell["source"]):
                    config_section_start = i
                    break

            assert config_section_start is not None

            # Next 3 cells should be code cells
            assert notebook["cells"][config_section_start + 1]["cell_type"] == "code"
            assert notebook["cells"][config_section_start + 2]["cell_type"] == "code"
            assert notebook["cells"][config_section_start + 3]["cell_type"] == "code"

            # Check content
            cell1 = "".join(notebook["cells"][config_section_start + 1]["source"])
            cell2 = "".join(notebook["cells"][config_section_start + 2]["source"])
            cell3 = "".join(notebook["cells"][config_section_start + 3]["source"])

            # Cell 1: print configs
            assert "print_configs(configs)" in cell1

            # Cell 2: edit configs and print again
            assert "configs[0]" in cell2
            assert "print_configs(configs)" in cell2
            # Should not have append example
            assert "append" not in cell2

            # Cell 3: launch all
            assert "launch_all(state)" in cell3

    def test_no_widget_references(self):
        """Test that generated notebooks don't reference widgets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(
                name="test_no_widgets",
                sections=["setup", "monitor", "config", "analysis"],
                output_dir=tmpdir
            )

            with open(filepath) as f:
                notebook = json.load(f)

            # Check all code cells
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    content = "".join(cell["source"])
                    # Should not have widget references
                    assert "widget" not in content.lower()
                    assert ".display()" not in content
                    assert "ipywidgets" not in content

    def test_training_jobs_in_setup(self):
        """Test that training jobs are properly passed to setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training jobs
            jobs = [
                TrainingJob(name="job1", config=TrainingJobConfig(curriculum="arena")),
                TrainingJob(name="job2", config=TrainingJobConfig(curriculum="harvest"))
            ]
            jobs[0].job_id = "sky-123"
            jobs[0].launched = True
            jobs[0].success = True

            filepath = generate_notebook(
                name="test_jobs",
                training_jobs=jobs,
                output_dir=tmpdir
            )

            with open(filepath) as f:
                notebook = json.load(f)

            # Find setup cell
            setup_cell = None
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and "experiment_data=" in "".join(cell["source"]):
                    setup_cell = "".join(cell["source"])
                    break

            assert setup_cell is not None
            # Should have job data
            assert "'jobs': [" in setup_cell
            assert "'name': 'job1'" in setup_cell
            assert "'job_id': 'sky-123'" in setup_cell
            assert "'launched': True" in setup_cell

    def test_training_configs_in_setup(self):
        """Test that training configs are properly passed to setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training configs
            configs = [
                TrainingJobConfig(curriculum="arena", gpus=4),
                TrainingJobConfig(curriculum="harvest", gpus=2, wandb_tags=["test"])
            ]

            filepath = generate_notebook(
                name="test_configs",
                training_job_configs=configs,
                output_dir=tmpdir
            )

            with open(filepath) as f:
                notebook = json.load(f)

            # Find setup cell
            setup_cell = None
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and "experiment_data=" in "".join(cell["source"]):
                    setup_cell = "".join(cell["source"])
                    break

            assert setup_cell is not None
            # Should have config data
            assert "'configs': [" in setup_cell
            assert "'curriculum': 'arena'" in setup_cell
            assert "'gpus': 4" in setup_cell
            assert "'wandb_tags': ['test']" in setup_cell

    def test_analysis_section_uses_print_functions(self):
        """Test that analysis section uses print-based functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(
                name="test_analysis",
                sections=["setup", "analysis"],
                output_dir=tmpdir
            )

            with open(filepath) as f:
                notebook = json.load(f)

            # Find analysis cells
            analysis_cells = []
            in_analysis = False
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown" and "Analysis" in "".join(cell["source"]):
                    in_analysis = True
                elif cell["cell_type"] == "markdown" and "##" in "".join(cell["source"]):
                    in_analysis = False
                elif in_analysis and cell["cell_type"] == "code":
                    analysis_cells.append("".join(cell["source"]))

            # Should use print functions
            analysis_content = "\n".join(analysis_cells)
            assert "print_wandb_runs(state)" in analysis_content
            assert "plot_sps(state)" in analysis_content
            assert "plot_metrics(state" in analysis_content

    def test_monitor_section_uses_print_functions(self):
        """Test that monitor section uses print-based functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generate_notebook(
                name="test_monitor",
                sections=["setup", "monitor"],
                output_dir=tmpdir
            )

            with open(filepath) as f:
                notebook = json.load(f)

            # Find monitor cells
            monitor_cells = []
            in_monitor = False
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown" and "Monitor & Control" in "".join(cell["source"]):
                    in_monitor = True
                elif cell["cell_type"] == "markdown" and "##" in "".join(cell["source"]):
                    in_monitor = False
                elif in_monitor and cell["cell_type"] == "code":
                    monitor_cells.append("".join(cell["source"]))

            # Should use print functions
            monitor_content = "\n".join(monitor_cells)
            assert "print_jobs(state)" in monitor_content
            assert "launch_all(state)" in monitor_content
            assert "kill_all(state)" in monitor_content

    def test_simplified_notebook_sections(self):
        """Test that simplified notebooks have the right sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't specify sections - should use DEFAULT_SECTIONS
            filepath = generate_notebook(
                name="test_simplified",
                output_dir=tmpdir
            )

            with open(filepath) as f:
                notebook = json.load(f)

            # Count sections
            sections = []
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown":
                    content = "".join(cell["source"])
                    if content.strip().startswith("## "):
                        sections.append(content.strip())

            # Should have setup, monitor, config sections
            assert any("Setup" in s for s in sections)
            assert any("Monitor" in s for s in sections)
            assert any("Configuration" in s for s in sections)
            # Should NOT have other sections by default
            assert not any("Analysis" in s for s in sections)
            assert not any("Replays" in s for s in sections)
            assert not any("Export" in s for s in sections)