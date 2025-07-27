"""Launch widgets and notebook generation."""

from typing import List, Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from experiments.notebooks.widget import ReactiveWidget
from experiments.training_job import TrainingJob, TrainingJobConfig


class LaunchWidget(ReactiveWidget):
    """Interactive widget for launching training runs."""
    
    def __init__(self, state: 'RunState', 
                 default_curriculum: str = "env/mettagrid/arena/basic",
                 default_gpus: int = 1,
                 default_nodes: int = 1,
                 default_spot: bool = True):
        """Initialize launch widget with sensible defaults.
        
        Args:
            state: The RunState instance
            default_curriculum: Default curriculum path
            default_gpus: Default number of GPUs
            default_nodes: Default number of nodes  
            default_spot: Whether to use spot instances by default
        """
        super().__init__(state, title=None)
        
        # Create input widgets
        self.run_name = widgets.Text(
            placeholder='my_experiment',
            description='Run Name:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.curriculum = widgets.Text(
            value=default_curriculum,
            description='Curriculum:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.gpus = widgets.IntSlider(
            value=default_gpus,
            min=1,
            max=8,
            description='GPUs:',
            style={'description_width': 'initial'}
        )
        
        self.nodes = widgets.IntSlider(
            value=default_nodes,
            min=1,
            max=4,
            description='Nodes:',
            style={'description_width': 'initial'}
        )
        
        self.spot = widgets.Checkbox(
            value=default_spot,
            description='Use Spot Instances',
            style={'description_width': 'initial'}
        )
        
        self.skip_git = widgets.Checkbox(
            value=False,
            description='Skip Git Check',
            style={'description_width': 'initial'}
        )
        
        self.wandb_tags = widgets.Text(
            placeholder='tag1, tag2',
            description='W&B Tags:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.additional_args = widgets.Textarea(
            placeholder='trainer.learning_rate=0.001\ntrainer.batch_size=32',
            description='Extra Args:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='80px')
        )
        
        # Launch button
        self.launch_button = widgets.Button(
            description='🚀 Launch Training',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.launch_button.on_click(self._on_launch_click)
        
        # Output area
        self.status_output = widgets.Output()
        
        # Layout
        self._create_layout()
    
    def _create_layout(self):
        """Create the widget layout."""
        # Group related inputs
        basic_inputs = widgets.VBox([
            self.run_name,
            self.curriculum,
            widgets.HBox([self.gpus, self.nodes]),
            widgets.HBox([self.spot, self.skip_git])
        ])
        
        advanced_inputs = widgets.VBox([
            self.wandb_tags,
            self.additional_args
        ])
        
        # Create accordion for advanced options
        accordion = widgets.Accordion(children=[advanced_inputs])
        accordion.set_title(0, 'Advanced Options')
        accordion.selected_index = None  # Start collapsed
        
        # Main layout
        self.container = widgets.VBox([
            widgets.HTML('<h3>🚀 Launch Training Run</h3>'),
            basic_inputs,
            accordion,
            widgets.HBox([self.launch_button]),
            self.status_output
        ], layout=widgets.Layout(
            padding='20px',
            border='1px solid #ddd',
            margin='10px 0'
        ))
    
    def render(self):
        """Render the widget."""
        display(self.container)
    
    def display(self):
        """Display the widget."""
        self.render()
    
    def _on_launch_click(self, button):
        """Handle launch button click."""
        with self.status_output:
            clear_output(wait=True)
            
            # Validate inputs
            if not self.run_name.value:
                print("❌ Please enter a run name")
                return
            
            # Parse tags
            tags = [tag.strip() for tag in self.wandb_tags.value.split(',') if tag.strip()]
            
            # Parse additional args
            additional_args = []
            if self.additional_args.value:
                for line in self.additional_args.value.strip().split('\n'):
                    if line.strip():
                        additional_args.append(line.strip())
            
            # Create config
            config = TrainingJobConfig(
                curriculum=self.curriculum.value,
                gpus=self.gpus.value,
                nodes=self.nodes.value,
                spot=self.spot.value,
                skip_git_check=self.skip_git.value,
                wandb_tags=tags if tags else None,
                additional_args=additional_args if additional_args else None
            )
            
            # Create job
            job = TrainingJob(name=self.run_name.value, config=config)
            
            print(f"Launching {self.run_name.value}...")
            print(f"Config: {self.gpus.value} GPUs, {self.nodes.value} nodes, {'spot' if self.spot.value else 'on-demand'}")
            
            # Launch
            success = job.launch()
            
            if success:
                print(f"✅ Successfully launched! Job ID: {job.job_id}")
                print(f"Run name: {job.name}")
                
                # Add to state
                self.state.add_job(job)
                
                # Clear inputs for next run
                self.run_name.value = ''
            else:
                print(f"❌ Failed to launch training run")


class MultiLaunchWidget(LaunchWidget):
    """Widget for launching multiple training runs."""
    
    def __init__(self, state: 'RunState', **kwargs):
        super().__init__(state, **kwargs)
        
        # Add multi-run controls
        self.num_runs = widgets.IntSlider(
            value=1,
            min=1,
            max=10,
            description='Number of Runs:',
            style={'description_width': 'initial'}
        )
        
        self.vary_seeds = widgets.Checkbox(
            value=True,
            description='Vary Seeds',
            style={'description_width': 'initial'}
        )
        
        # Update button text
        self.launch_button.description = '🚀 Launch Multiple'
        
        # Re-create layout with new controls
        self._create_layout()
    
    def _create_layout(self):
        """Create the widget layout with multi-run controls."""
        # Check if multi-run controls exist (they won't on first parent init)
        if not hasattr(self, 'num_runs'):
            return super()._create_layout()
            
        # Group related inputs
        basic_inputs = widgets.VBox([
            self.run_name,
            self.curriculum,
            widgets.HBox([self.gpus, self.nodes]),
            widgets.HBox([self.spot, self.skip_git]),
            widgets.HBox([self.num_runs, self.vary_seeds])
        ])
        
        advanced_inputs = widgets.VBox([
            self.wandb_tags,
            self.additional_args
        ])
        
        # Create accordion for advanced options
        accordion = widgets.Accordion(children=[advanced_inputs])
        accordion.set_title(0, 'Advanced Options')
        accordion.selected_index = None  # Start collapsed
        
        # Main layout
        self.container = widgets.VBox([
            widgets.HTML('<h3>🚀 Launch Multiple Training Runs</h3>'),
            basic_inputs,
            accordion,
            widgets.HBox([self.launch_button]),
            self.status_output
        ], layout=widgets.Layout(
            padding='20px',
            border='1px solid #ddd',
            margin='10px 0'
        ))
    
    def _on_launch_click(self, button):
        """Handle launch button click for multiple runs."""
        with self.status_output:
            clear_output(wait=True)
            
            # Validate inputs
            if not self.run_name.value:
                print("❌ Please enter a base run name")
                return
            
            # Parse tags
            tags = [tag.strip() for tag in self.wandb_tags.value.split(',') if tag.strip()]
            
            # Parse additional args
            additional_args = []
            if self.additional_args.value:
                for line in self.additional_args.value.strip().split('\n'):
                    if line.strip():
                        additional_args.append(line.strip())
            
            print(f"Launching {self.num_runs.value} runs...")
            
            # Launch multiple runs
            for i in range(self.num_runs.value):
                run_name = f"{self.run_name.value}.{i + 1}" if self.num_runs.value > 1 else self.run_name.value
                
                # Copy additional args and add seed if varying
                run_args = additional_args.copy()
                if self.vary_seeds.value and i > 0:
                    run_args.append(f"trainer.seed={42 + i}")
                
                # Create config
                config = TrainingJobConfig(
                    curriculum=self.curriculum.value,
                    gpus=self.gpus.value,
                    nodes=self.nodes.value,
                    spot=self.spot.value,
                    skip_git_check=self.skip_git.value,
                    wandb_tags=tags if tags else None,
                    additional_args=run_args if run_args else None
                )
                
                # Create and launch job
                job = TrainingJob(name=run_name, config=config)
                
                print(f"\n[{i+1}/{self.num_runs.value}] Launching {run_name}...")
                success = job.launch()
                
                if success:
                    print(f"✅ Successfully launched! Job ID: {job.job_id}")
                    self.state.add_job(job)
                else:
                    print(f"❌ Failed to launch {run_name}")
                    print("Stopping remaining launches")
                    break
            
            # Clear inputs
            self.run_name.value = ''


def generate_cells(training_job_configs: List[TrainingJobConfig] = None) -> List[Dict[str, Any]]:
    """Generate notebook cells for training section.
    
    Args:
        training_job_configs: Optional list of training job configurations
        
    Returns:
        List of cell definitions
    """
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 🚀 Launch Training"
    })
    
    # Single launch widget
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Interactive launch widget
from experiments.notebooks.widgets.training import LaunchWidget

launch_widget = LaunchWidget(state)
launch_widget.display()"""
    })
    
    # Multiple launch widget
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Launch Multiple Runs"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Launch multiple runs with seed variation
from experiments.notebooks.widgets.training import MultiLaunchWidget

multi_launch = MultiLaunchWidget(state)
multi_launch.display()"""
    })
    
    # Pre-configured jobs if any
    if training_job_configs:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": f"### Pre-configured Jobs\n\n{len(training_job_configs)} jobs are configured and ready to launch:"
        })
        
        # Config manager widget
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Interactive config editor
from experiments.notebooks.widgets.config_widget import ConfigManagerWidget

config_manager = ConfigManagerWidget(state)
config_manager.display()"""
        })
        
        # Quick launch option
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Quick launch all configs (alternative to widget)
if state.training_job_configs:
    print(f"Ready to launch {len(state.training_job_configs)} jobs")
    launched = state.launch_all_configs(experiment_name=state.metadata.get('name', 'experiment'))
    print(f"✓ Launched {len(launched)} jobs")
else:
    print("No configs to launch")"""
        })
    
    # Programmatic launch examples
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Programmatic Launch Examples"
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Direct TrainingJob API usage
from experiments.training_job import TrainingJob, TrainingJobConfig

# Create custom config
config = TrainingJobConfig(
    curriculum="env/mettagrid/arena/basic",
    gpus=4,
    nodes=1,
    spot=True,
    wandb_tags=["custom", "experiment"],
    additional_args=[
        "trainer.learning_rate=0.001",
        "trainer.batch_size=32"
    ]
)

# Create and launch job
job = TrainingJob(name="my_custom_experiment", config=config)
if job.launch():
    print(f"✅ Launched! Job ID: {job.job_id}")
    state.add_job(job)"""
    })
    
    return cells