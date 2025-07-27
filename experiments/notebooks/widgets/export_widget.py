"""Export widgets and notebook generation."""

from typing import List, Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import pandas as pd
from datetime import datetime

from experiments.notebooks.widget import ReactiveWidget
from experiments.wandb_service import WandbService


def fetch_run_metrics(run_name: str) -> pd.DataFrame:
    """Fetch metrics data for a given run from W&B.
    
    Args:
        run_name: Name of the W&B run
        
    Returns:
        DataFrame with metrics data
    """
    try:
        wandb_service = WandbService()
        run = wandb_service.get_run(run_name)
        if not run:
            return pd.DataFrame()
        
        # Fetch metrics history
        history = run.history()
        return history
    except Exception as e:
        print(f"Error fetching metrics for {run_name}: {e}")
        return pd.DataFrame()


class ExportWidget(ReactiveWidget):
    """Widget for exporting training data and results."""
    
    def __init__(self, state: 'RunState'):
        """Initialize export widget."""
        super().__init__(state, title=None)
        
        # Create controls
        self.run_selector = widgets.SelectMultiple(
            options=[],
            value=[],
            description='Select Runs:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='150px')
        )
        
        self.export_format = widgets.Dropdown(
            options=['csv', 'json', 'parquet'],
            value='csv',
            description='Format:',
            style={'description_width': 'initial'}
        )
        
        self.include_metrics = widgets.Checkbox(
            value=True,
            description='Include Metrics',
            style={'description_width': 'initial'}
        )
        
        self.include_configs = widgets.Checkbox(
            value=True,
            description='Include Configs',
            style={'description_width': 'initial'}
        )
        
        self.output_dir = widgets.Text(
            value='./exports',
            description='Output Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.refresh_btn = widgets.Button(
            description='Refresh',
            button_style='info',
            icon='refresh'
        )
        self.refresh_btn.on_click(lambda b: self._refresh_runs())
        
        self.export_btn = widgets.Button(
            description='Export Data',
            button_style='primary',
            icon='download'
        )
        self.export_btn.on_click(lambda b: self._export_data())
        
        # Output area
        self.output = widgets.Output()
        
        # Layout
        self._create_layout()
        
        # Initial refresh
        self._refresh_runs()
    
    def _create_layout(self):
        """Create the widget layout."""
        options = widgets.VBox([
            self.export_format,
            self.include_metrics,
            self.include_configs,
            self.output_dir
        ])
        
        controls = widgets.HBox([
            self.run_selector,
            widgets.VBox([
                options,
                widgets.HBox([self.refresh_btn, self.export_btn])
            ])
        ])
        
        self.container = widgets.VBox([
            widgets.HTML('<h3>📦 Export Training Data</h3>'),
            controls,
            self.output
        ], layout=widgets.Layout(
            padding='20px',
            border='1px solid #ddd',
            margin='10px 0'
        ))
    
    def _refresh_runs(self):
        """Refresh the list of available runs."""
        run_names = self.state.wandb_run_names
        self.run_selector.options = run_names
        # Select all by default
        self.run_selector.value = run_names[:5]  # Select first 5
    
    def _export_data(self):
        """Export the selected data."""
        with self.output:
            clear_output(wait=True)
            
            selected_runs = list(self.run_selector.value)
            if not selected_runs:
                print("⚠️ Please select at least one run")
                return
            
            output_dir = self.output_dir.value
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Exporting {len(selected_runs)} runs to {output_dir}/")
            
            for run_name in selected_runs:
                print(f"\nExporting {run_name}...")
                
                try:
                    # Export based on format
                    if self.export_format.value == 'csv':
                        self._export_csv(run_name, output_dir)
                    elif self.export_format.value == 'json':
                        self._export_json(run_name, output_dir)
                    elif self.export_format.value == 'parquet':
                        self._export_parquet(run_name, output_dir)
                    
                    print(f"  ✅ Exported successfully")
                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
            
            print(f"\n✅ Export complete! Files saved to: {output_dir}/")
    
    def _export_csv(self, run_name: str, output_dir: str):
        """Export run data to CSV."""
        # Fetch metrics using WandbService
        from experiments.wandb_service import get_wandb_service
        
        # Export metrics
        if self.include_metrics.value:
            df = fetch_run_metrics(run_name)
            if not df.empty:
                filepath = os.path.join(output_dir, f"{run_name}_metrics.csv")
                df.to_csv(filepath, index=False)
                print(f"  - Metrics: {filepath}")
        
        # Export config
        if self.include_configs.value:
            wandb_service = get_wandb_service()
            config = wandb_service.get_run_config(run_name)
            if config:
                import pandas as pd
                config_df = pd.DataFrame([config])
                filepath = os.path.join(output_dir, f"{run_name}_config.csv")
                config_df.to_csv(filepath, index=False)
                print(f"  - Config: {filepath}")
    
    def _export_json(self, run_name: str, output_dir: str):
        """Export run data to JSON."""
        import json
        # Fetch metrics using WandbService
        from experiments.wandb_service import get_wandb_service
        
        data = {"run_name": run_name}
        
        # Add metrics
        if self.include_metrics.value:
            df = fetch_run_metrics(run_name)
            if not df.empty:
                data["metrics"] = df.to_dict(orient='records')
        
        # Add config
        if self.include_configs.value:
            wandb_service = get_wandb_service()
            config = wandb_service.get_run_config(run_name)
            if config:
                data["config"] = config
        
        # Write to file
        filepath = os.path.join(output_dir, f"{run_name}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  - Data: {filepath}")
    
    def _export_parquet(self, run_name: str, output_dir: str):
        """Export run data to Parquet."""
        # Fetch metrics using WandbService
        
        if self.include_metrics.value:
            df = fetch_run_metrics(run_name)
            if not df.empty:
                filepath = os.path.join(output_dir, f"{run_name}_metrics.parquet")
                df.to_parquet(filepath, index=False)
                print(f"  - Metrics: {filepath}")
    
    def render(self):
        """Render the widget."""
        display(self.container)
    
    def display(self):
        """Display the widget."""
        self.render()


def generate_cells() -> List[Dict[str, Any]]:
    """Generate notebook cells for export section.
    
    Returns:
        List of cell definitions
    """
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 📦 Export Data"
    })
    
    # Export widget
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Interactive export widget
from experiments.notebooks.widgets.export_widget import ExportWidget

export_widget = ExportWidget(state)
export_widget.display()"""
    })
    
    # Manual export examples
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Manual Export Examples"
    })
    
    # Export single run
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Export a single run's metrics
# Removed import - will use WandbService directly

if state.wandb_run_names:
    run_name = state.wandb_run_names[0]
    df = fetch_run_metrics(run_name)
    
    # Save to CSV
    df.to_csv(f"{run_name}_metrics.csv", index=False)
    print(f"Exported {len(df)} rows to {run_name}_metrics.csv")
    
    # Show sample
    print("\\nSample data:")
    display(df.head())"""
    })
    
    # Export configs
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Export run configurations
from experiments.wandb_service import get_wandb_service
import pandas as pd

wandb_service = get_wandb_service()
configs = []

for run_name in state.wandb_run_names[:5]:  # First 5 runs
    config = wandb_service.get_run_config(run_name)
    if config:
        config['run_name'] = run_name
        configs.append(config)

if configs:
    config_df = pd.DataFrame(configs)
    config_df.to_csv("run_configs.csv", index=False)
    print(f"Exported {len(configs)} configurations")
    display(config_df[['run_name', 'trainer.total_timesteps', 'trainer.learning_rate']].head())"""
    })
    
    # Export summary statistics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Export summary statistics
import pandas as pd

summary_data = []

for run_name in state.wandb_run_names:
    try:
        df = fetch_run_metrics(run_name)
        if not df.empty and 'overview/reward' in df.columns:
            summary = {
                'run_name': run_name,
                'num_steps': len(df),
                'final_reward': df['overview/reward'].iloc[-1],
                'max_reward': df['overview/reward'].max(),
                'mean_reward': df['overview/reward'].mean(),
                'std_reward': df['overview/reward'].std()
            }
            summary_data.append(summary)
    except Exception as e:
        print(f"Error processing {run_name}: {e}")

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("run_summary.csv", index=False)
    print(f"Exported summary for {len(summary_data)} runs")
    display(summary_df.head())"""
    })
    
    # Export for external analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Export for External Analysis"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Create a comprehensive dataset for external analysis
import os
import json

export_dir = "./exports/comprehensive"
os.makedirs(export_dir, exist_ok=True)

# Export metadata
metadata = {
    'experiment_name': state.experiment_name if hasattr(state, 'experiment_name') else 'unknown',
    'num_runs': len(state.wandb_run_names),
    'runs': []
}

for job in state.training_jobs:
    run_info = {
        'name': job.name,
        'job_id': job.job_id,
        'launched': job.launched,
        'success': job.success,
        'config': job.config.dict() if hasattr(job.config, 'dict') else {}
    }
    metadata['runs'].append(run_info)

# Save metadata
with open(os.path.join(export_dir, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Exported comprehensive dataset to {export_dir}/")
print(f"  - metadata.json")
print(f"  - Ready for external analysis tools")"""
    })
    
    return cells