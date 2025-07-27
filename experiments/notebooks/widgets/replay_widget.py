"""Replay widgets and notebook generation."""

import re
from typing import List, Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output, IFrame

from experiments.notebooks.widget import ReactiveWidget
from experiments.wandb_service import WandbService


class ReplayWidget(ReactiveWidget):
    """Interactive widget for viewing MettaScope replays."""

    def __init__(self, state: "RunState"):
        """Initialize replay widget."""
        super().__init__(state, title=None)

        # Create controls
        self.run_selector = widgets.Dropdown(
            options=[],
            description="Select Run:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
        )

        self.step_selector = widgets.SelectionSlider(
            options=["last"],
            value="last",
            description="Step:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
        )

        self.width = widgets.IntSlider(
            value=1000,
            min=400,
            max=1600,
            step=100,
            description="Width:",
            style={"description_width": "initial"},
        )

        self.height = widgets.IntSlider(
            value=600,
            min=300,
            max=1200,
            step=50,
            description="Height:",
            style={"description_width": "initial"},
        )

        self.refresh_btn = widgets.Button(
            description="Refresh Runs", button_style="info", icon="refresh"
        )
        self.refresh_btn.on_click(lambda b: self._refresh_runs())

        self.load_btn = widgets.Button(
            description="Load Replay", button_style="primary", icon="play"
        )
        self.load_btn.on_click(lambda b: self._load_replay())

        # Output area
        self.output = widgets.Output()

        # Layout
        self._create_layout()

        # Initial setup
        self._refresh_runs()

        # Update available steps when run changes
        self.run_selector.observe(self._on_run_change, "value")

    def _create_layout(self):
        """Create the widget layout."""
        controls = widgets.VBox(
            [
                self.run_selector,
                self.step_selector,
                widgets.HBox([self.width, self.height]),
                widgets.HBox([self.refresh_btn, self.load_btn]),
            ]
        )

        self.container = widgets.VBox(
            [widgets.HTML("<h3>🎬 MettaScope Replays</h3>"), controls, self.output],
            layout=widgets.Layout(
                padding="20px", border="1px solid #ddd", margin="10px 0"
            ),
        )

    def _refresh_runs(self):
        """Refresh the list of available runs."""
        run_names = self.state.wandb_run_names
        self.run_selector.options = run_names if run_names else ["No runs available"]
        if run_names:
            self.run_selector.value = run_names[0]

    def _on_run_change(self, change):
        """Handle run selection change."""
        if change["new"] and change["new"] != "No runs available":
            with self.output:
                clear_output(wait=True)
                print("Fetching available replays...")

            # Get available replays for this run
            try:
                replays = get_available_replays(change["new"])
                if replays:
                    # Update step selector with available steps
                    step_options = []
                    for replay in replays:
                        step_options.append(
                            (
                                f"Step {replay['step']} - {replay['label']}",
                                replay["step"],
                            )
                        )
                    step_options.append(("Last Step", "last"))
                    self.step_selector.options = step_options
                    self.step_selector.value = "last"
                else:
                    self.step_selector.options = ["No replays available"]

                with self.output:
                    clear_output(wait=True)
                    print(f"Found {len(replays)} replays for {change['new']}")
            except Exception as e:
                with self.output:
                    clear_output(wait=True)
                    print(f"Error fetching replays: {e}")

    def _load_replay(self):
        """Load the selected replay."""
        with self.output:
            clear_output(wait=True)

            if (
                not self.run_selector.value
                or self.run_selector.value == "No runs available"
            ):
                print("⚠️ Please select a run")
                return

            # Show the replay
            try:
                show_replay(
                    run_name=self.run_selector.value,
                    step=self.step_selector.value,
                    width=self.width.value,
                    height=self.height.value,
                )
            except Exception as e:
                print(f"❌ Error loading replay: {e}")

    def render(self):
        """Render the widget."""
        display(self.container)

    def display(self):
        """Display the widget."""
        self.render()


def generate_cells() -> List[Dict[str, Any]]:
    """Generate notebook cells for replays section.

    Returns:
        List of cell definitions
    """
    cells = []

    # Section header
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": "## 🎬 View Replays"}
    )

    # Interactive replay widget
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Interactive replay viewer
from experiments.notebooks.widgets.replay_widget import ReplayWidget

replay_widget = ReplayWidget(state)
replay_widget.display()""",
        }
    )

    # Import utilities
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Import replay utilities
from experiments.notebooks.widgets.replay_widget import show_replay, get_available_replays""",
        }
    )

    # Manual replay examples
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Manual Replay Examples",
        }
    )

    # Show specific replay
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Show last replay for a specific run
if state.wandb_run_names:
    run_name = state.wandb_run_names[0]
    show_replay(run_name, step="last", width=1000, height=600)
else:
    print("No runs tracked yet. Launch some runs first!")""",
        }
    )

    # List available replays
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Get available replays for a run
if state.wandb_run_names:
    run_name = state.wandb_run_names[0]
    replays = get_available_replays(run_name)
    print(f"Available replays for {run_name}:")
    for replay in replays[-10:]:  # Show last 10
        print(f"  Step {replay['step']:,} - {replay['label']}")
else:
    print("No runs available")""",
        }
    )

    # Show specific step
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Show replay at a specific step
# step = 100000  # Replace with desired step
# show_replay(state.wandb_run_names[0], step=step, width=1200, height=800)""",
        }
    )

    # Compare replays
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Compare replays from different runs side by side
if len(state.wandb_run_names) >= 2:
    from IPython.display import display, HTML
    
    # Create side-by-side layout
    html = '<div style="display: flex; gap: 10px;">'
    for i, run_name in enumerate(state.wandb_run_names[:2]):
        html += f'<div style="flex: 1;"><h4>{run_name}</h4>'
        # Note: You'll need to capture the iframe HTML from show_replay
        # This is a placeholder for the concept
        html += f'<p>Replay for {run_name} would go here</p>'
        html += '</div>'
    html += '</div>'
    
    display(HTML(html))
    print("Note: Implement iframe capture for side-by-side comparison")
else:
    print("Need at least 2 runs for comparison")""",
        }
    )

    return cells


# Replay utility functions (moved from replays.py)


def show_replay(
    run_name: str, step: str | int = "last", width: int = 1000, height: int = 600
) -> None:
    """Show a MettaScope replay for a specific run and step.

    Args:
        run_name: Name of the wandb run
        step: Step number or "first"/"last"
        width: IFrame width in pixels
        height: IFrame height in pixels
    """
    wandb_service = WandbService()
    run = wandb_service.get_run(run_name)
    if run is None:
        return

    replay_urls = fetch_replay_urls_for_run(run)

    if not replay_urls:
        print(f"No replays found for {run_name}")
        return

    # Select the requested replay
    if step == "last":
        selected = replay_urls[-1]
    elif step == "first":
        selected = replay_urls[0]
    else:
        # Find replay closest to requested step
        target_step = int(step)
        selected = min(replay_urls, key=lambda r: abs(r["step"] - target_step))
        if selected["step"] != target_step:
            print(
                f"Note: Requested step {target_step}, showing closest available step {selected['step']}"
            )

    print(f"Loading MettaScope viewer for {run_name} at step {selected['step']:,}...")
    print(f"\nDirect link: {selected['url']}")

    # Try to display in notebook environment
    try:
        display(IFrame(src=selected["url"], width=width, height=height))
    except ImportError:
        # Not in a notebook environment
        print(f"Open in browser: {selected['url']}")


def get_available_replays(run_name: str) -> list[dict]:
    """Get list of available replays for a run.

    Args:
        run_name: Name of the wandb run

    Returns:
        List of dicts with keys: step, url, filename, label
    """
    wandb_service = WandbService()
    run = wandb_service.get_run(run_name)
    if run is None:
        return []

    return fetch_replay_urls_for_run(run)


def fetch_replay_urls_for_run(run) -> list[dict]:
    """Fetch replay URLs from wandb run files.

    Args:
        run: Wandb run object

    Returns:
        List of dicts with keys: step, url, filename, label
    """
    files = run.files()
    replay_urls = []

    # Filter for replay HTML files
    replay_files = [
        f
        for f in files
        if "media/html/replays/link_" in f.name and f.name.endswith(".html")
    ]

    if not replay_files:
        return []

    # Sort by step number
    def get_step_from_filename(file):
        match = re.search(r"link_(\d+)_", file.name)
        return int(match.group(1)) if match else 0

    replay_files.sort(key=get_step_from_filename)

    # Process files (limit to avoid too many)
    max_files = min(20, len(replay_files))
    recent_files = replay_files[-max_files:]

    for file in recent_files:
        try:
            # Download and read the HTML file
            with file.download(replace=True, root="/tmp") as f:
                content = f.read()
            match = re.search(r'<a[^>]+href="([^"]+)"', content)
            if match:
                href = match.group(1)
                if href:
                    step = get_step_from_filename(file)
                    replay_urls.append(
                        {
                            "step": step,
                            "url": href,
                            "filename": file.name,
                            "label": f"Step {step:,}",
                        }
                    )
        except Exception:
            pass

    return replay_urls
