"""Wandb widgets and notebook generation."""

from typing import List, Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

from experiments.notebooks.widget import ReactiveWidget
from experiments.wandb_service import WandbService


class WandbWidget(ReactiveWidget):
    """Reactive widget for displaying wandb analysis plots."""

    def __init__(self, state: "RunState", plot_type: str = "sps", **plot_kwargs):
        """Initialize analysis widget.

        Args:
            state: The RunState instance to monitor
            plot_type: Type of plot to display (e.g., "sps")
            **plot_kwargs: Additional arguments for the plot function
        """
        super().__init__(state, title=f"Analysis: {plot_type.upper()}")
        self.plot_type = plot_type
        self.plot_kwargs = plot_kwargs

    def render(self) -> None:
        """Render the analysis plot."""
        if self.state.wandb_run_names:
            if self.plot_type == "sps":
                # plot_sps is implemented below
                fig = plot_sps(self.state.wandb_run_names, **self.plot_kwargs)
                fig.show()
            else:
                print(f"Unknown plot type: {self.plot_type}")
        else:
            print("No runs to analyze yet.")


class InteractiveWandbWidget(ReactiveWidget):
    """Interactive widget for selecting and displaying different wandb analyses."""

    def __init__(self, state: "RunState"):
        """Initialize interactive analysis widget."""
        super().__init__(state, title=None)

        # Create controls
        self.run_selector = widgets.SelectMultiple(
            options=[],
            value=[],
            description="Select Runs:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px", height="150px"),
        )

        self.plot_type = widgets.Dropdown(
            options=["sps", "reward_curves", "learning_curves"],
            value="sps",
            description="Plot Type:",
            style={"description_width": "initial"},
        )

        self.metric_input = widgets.Text(
            placeholder="overview/reward",
            description="Metric:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        self.refresh_btn = widgets.Button(
            description="Refresh Runs", button_style="info", icon="refresh"
        )
        self.refresh_btn.on_click(lambda b: self._refresh_runs())

        self.analyze_btn = widgets.Button(
            description="Generate Analysis", button_style="primary", icon="chart-line"
        )
        self.analyze_btn.on_click(lambda b: self._generate_analysis())

        # Output area
        self.output = widgets.Output()

        # Layout
        self._create_layout()

        # Initial refresh
        self._refresh_runs()

    def _create_layout(self):
        """Create the widget layout."""
        controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.run_selector,
                        widgets.VBox(
                            [
                                self.plot_type,
                                self.metric_input,
                                widgets.HBox([self.refresh_btn, self.analyze_btn]),
                            ]
                        ),
                    ]
                ),
            ]
        )

        self.container = widgets.VBox(
            [widgets.HTML("<h3>📊 Interactive Analysis</h3>"), controls, self.output],
            layout=widgets.Layout(
                padding="20px", border="1px solid #ddd", margin="10px 0"
            ),
        )

    def _refresh_runs(self):
        """Refresh the list of available runs."""
        run_names = self.state.wandb_run_names
        self.run_selector.options = run_names
        # Select all by default
        self.run_selector.value = run_names

    def _generate_analysis(self):
        """Generate the selected analysis."""
        with self.output:
            clear_output(wait=True)

            selected_runs = list(self.run_selector.value)
            if not selected_runs:
                print("⚠️ Please select at least one run")
                return

            plot_type = self.plot_type.value

            if plot_type == "sps":
                # plot_sps is implemented below
                fig = plot_sps(selected_runs)
                fig.show()
            elif plot_type == "reward_curves":
                # plot_metric_curves is implemented below
                metric = self.metric_input.value or "overview/reward"
                fig = plot_metric_curves(selected_runs, metric=metric)
                fig.show()
            elif plot_type == "learning_curves":
                # plot_learning_curves is implemented below
                fig = plot_learning_curves(selected_runs)
                fig.show()

    def render(self):
        """Render the widget."""
        display(self.container)

    def display(self):
        """Display the widget."""
        self.render()


def generate_cells(include_sps: bool = True) -> List[Dict[str, Any]]:
    """Generate notebook cells for analysis section.

    Args:
        include_sps: Whether to include SPS (steps per second) analysis

    Returns:
        List of cell definitions
    """
    cells = []

    # Section header
    cells.append({"cell_type": "markdown", "metadata": {}, "source": "## 📊 Analysis"})

    # Interactive analysis widget
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Interactive analysis widget
from experiments.notebooks.widgets.wandb_widget import InteractiveWandbWidget

analysis_widget = InteractiveWandbWidget(state)
analysis_widget.display()""",
        }
    )

    # Import analysis functions
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Import analysis utilities
# Plotting functions are implemented below

# Quick access to run names
run_names = state.wandb_run_names
print(f"Available runs: {len(run_names)}")
for name in run_names[:5]:  # Show first 5
    print(f"  - {name}")
if len(run_names) > 5:
    print(f"  ... and {len(run_names) - 5} more")""",
        }
    )

    if include_sps:
        # SPS Analysis
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "### Steps Per Second (SPS) Analysis",
            }
        )

        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": """# Plot SPS comparison
if run_names:
    fig = plot_sps(run_names)
    fig.show()
else:
    print("No runs available for analysis")""",
            }
        )

        # Reactive SPS widget
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": """# Reactive SPS widget (updates when new runs are added)
from experiments.notebooks.widgets.wandb_widget import WandbWidget

sps_widget = WandbWidget(state, plot_type="sps")
sps_widget.display()""",
            }
        )

    # Metric curves
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": "### Training Metrics"}
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Plot reward curves
if run_names:
    fig = plot_metric_curves(run_names, metric='overview/reward')
    fig.show()
else:
    print("No runs available")""",
        }
    )

    # Custom metrics
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Plot custom metric
# metric = 'overview/episode_length'  # Example
# fig = plot_metric_curves(run_names, metric=metric)
# fig.show()""",
        }
    )

    # Fetch raw data
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": "### Fetch Raw Metrics"}
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Fetch metrics data for detailed analysis
if run_names:
    # Fetch data for first run
    df = fetch_run_metrics(run_names[0])
    print(f"Fetched {len(df)} data points")
    print(f"Available metrics: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # Basic statistics
    if 'overview/reward' in df.columns:
        print(f"\\nReward stats:")
        print(f"  Mean: {df['overview/reward'].mean():.4f}")
        print(f"  Max: {df['overview/reward'].max():.4f}")
        print(f"  Final: {df['overview/reward'].iloc[-1]:.4f}")""",
        }
    )

    # Compare runs
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": "### Compare Runs"}
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Compare multiple runs
if len(run_names) >= 2:
    # Compare first two runs
    comparison = compare_runs(run_names[:2])
    display(comparison)
else:
    print("Need at least 2 runs to compare")""",
        }
    )

    return cells


# Plotting function implementations
def plot_sps(run_names: List[str], **kwargs) -> plt.Figure:
    """Plot SPS metrics for the given runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("SPS Analysis")
    ax.set_xlabel("Steps")
    ax.set_ylabel("SPS")
    # Placeholder - actual implementation would fetch and plot data
    return fig


def plot_metric_curves(run_names: List[str], metric: str, **kwargs) -> plt.Figure:
    """Plot metric curves for the given runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{metric} Over Time")
    ax.set_xlabel("Steps")
    ax.set_ylabel(metric)
    # Placeholder - actual implementation would fetch and plot data
    return fig


def plot_learning_curves(run_names: List[str], **kwargs) -> plt.Figure:
    """Plot learning curves for the given runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Learning Curves")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    # Placeholder - actual implementation would fetch and plot data
    return fig


def compare_runs(run_names: List[str], **kwargs) -> plt.Figure:
    """Compare multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Run Comparison")
    # Placeholder - actual implementation would compare runs
    return fig


def fetch_run_metrics(run_name: str) -> pd.DataFrame:
    """Fetch metrics for a given run."""
    try:
        wandb_service = WandbService()
        run = wandb_service.get_run(run_name)
        if not run:
            return pd.DataFrame()
        return run.history()
    except Exception as e:
        print(f"Error fetching metrics for {run_name}: {e}")
        return pd.DataFrame()
