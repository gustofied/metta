"""Training job configuration widgets."""

import ipywidgets as widgets
from IPython.display import display
from typing import Optional

from experiments.notebooks.widget import ReactiveWidget
from experiments.training_job import TrainingJobConfig


class TrainingJobConfigWidget(ReactiveWidget):
    """Widget for editing a single TrainingJobConfig."""
    
    def __init__(self, state: 'RunState', config_index: int, **kwargs):
        """Initialize config widget.
        
        Args:
            state: The RunState instance
            config_index: Index of the config in state.training_job_configs
        """
        super().__init__(state, title=f"Config {config_index}", **kwargs)
        self.config_index = config_index
        
        # Get the config
        if config_index < len(state.training_job_configs):
            self.config = state.training_job_configs[config_index]
        else:
            self.config = TrainingJobConfig()
        
        # Create input widgets for all TrainingJobConfig fields
        self.curriculum_input = widgets.Text(
            value=self.config.curriculum,
            description='Curriculum:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        
        self.gpus_input = widgets.IntSlider(
            value=self.config.gpus,
            min=1,
            max=8,
            description='GPUs:',
            style={'description_width': 'initial'}
        )
        
        self.nodes_input = widgets.IntSlider(
            value=self.config.nodes,
            min=1,
            max=4,
            description='Nodes:',
            style={'description_width': 'initial'}
        )
        
        self.spot_input = widgets.Checkbox(
            value=self.config.spot,
            description='Use Spot Instances',
            style={'description_width': 'initial'}
        )
        
        self.skip_git_check_input = widgets.Checkbox(
            value=self.config.skip_git_check,
            description='Skip Git Check',
            style={'description_width': 'initial'}
        )
        
        self.wandb_tags_input = widgets.Text(
            value=", ".join(self.config.wandb_tags or []),
            placeholder='tag1, tag2',
            description='W&B Tags:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        
        self.additional_args_input = widgets.Textarea(
            value='\n'.join(self.config.additional_args or []),
            placeholder='trainer.learning_rate=0.001\ntrainer.batch_size=32',
            description='Additional Args:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px', height='80px')
        )
        
        # Save button
        self.save_button = widgets.Button(
            description='💾 Save Config',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.save_button.on_click(self._on_save_click)
        
        # Delete button
        self.delete_button = widgets.Button(
            description='🗑️ Delete',
            button_style='danger',
            layout=widgets.Layout(width='150px')
        )
        self.delete_button.on_click(self._on_delete_click)
        
        # Status output
        self.status_output = widgets.Output()
        
        # Layout
        self._create_layout()
    
    def _create_layout(self):
        """Create the widget layout."""
        inputs = widgets.VBox([
            self.curriculum_input,
            widgets.HBox([self.gpus_input, self.nodes_input]),
            widgets.HBox([self.spot_input, self.skip_git_check_input]),
            self.wandb_tags_input,
            self.additional_args_input,
            widgets.HBox([self.save_button, self.delete_button]),
            self.status_output
        ])
        
        self.container = widgets.VBox([
            widgets.HTML(f'<h4>Edit Config {self.config_index}</h4>'),
            inputs
        ], layout=widgets.Layout(
            padding='15px',
            border='1px solid #ddd',
            margin='10px 0'
        ))
    
    def _on_save_click(self, button):
        """Handle save button click."""
        with self.status_output:
            self.status_output.clear_output()
            
            # Parse tags
            tags = [tag.strip() for tag in self.wandb_tags_input.value.split(',') if tag.strip()]
            
            # Parse additional args
            additional_args = []
            if self.additional_args_input.value:
                for line in self.additional_args_input.value.strip().split('\n'):
                    if line.strip():
                        additional_args.append(line.strip())
            
            # Create new config with updated values
            updated_config = TrainingJobConfig(
                curriculum=self.curriculum_input.value,
                gpus=self.gpus_input.value,
                nodes=self.nodes_input.value,
                spot=self.spot_input.value,
                skip_git_check=self.skip_git_check_input.value,
                wandb_tags=tags if tags else None,
                additional_args=additional_args if additional_args else None
            )
            
            # Update in state
            if hasattr(self.state, 'update_config'):
                self.state.update_config(self.config_index, updated_config)
            else:
                # Direct update if method doesn't exist
                if self.config_index < len(self.state.training_job_configs):
                    self.state.training_job_configs[self.config_index] = updated_config
                    self.state._trigger_updates()
            
            print("✅ Config saved")
    
    def _on_delete_click(self, button):
        """Handle delete button click."""
        with self.status_output:
            self.status_output.clear_output()
            
            # Remove from state
            self.state.remove_config(self.config_index)
            print("✓ Config deleted")
            
            # Hide the widget
            self.container.layout.display = 'none'
    
    def display(self):
        """Display the widget."""
        display(self.container)


class ConfigManagerWidget(ReactiveWidget):
    """Widget for managing all training job configs."""
    
    def __init__(self, state: 'RunState', **kwargs):
        """Initialize config manager widget."""
        super().__init__(state, title="Job Configurations", **kwargs)
        
        # Container for individual config widgets
        self.configs_container = widgets.VBox()
        
        # Add new config button
        self.add_button = widgets.Button(
            description='➕ Add Config',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.add_button.on_click(self._on_add_click)
        
        # Launch all button
        self.launch_all_button = widgets.Button(
            description='🚀 Launch All',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.launch_all_button.on_click(self._on_launch_all_click)
        
        # Status output
        self.status_output = widgets.Output()
        
        # Layout
        self._create_layout()
        
        # Initial update
        self._update_configs()
        
        # Register for state updates
        state.register_update_callback(self._update_configs)
    
    def _create_layout(self):
        """Create the widget layout."""
        self.container = widgets.VBox([
            widgets.HTML('<h3>📋 Training Job Configurations</h3>'),
            widgets.HBox([self.add_button, self.launch_all_button]),
            self.configs_container,
            self.status_output
        ], layout=widgets.Layout(
            padding='20px',
            border='1px solid #ddd',
            margin='10px 0'
        ))
    
    def _update_configs(self):
        """Update the configs display."""
        # Clear existing widgets
        self.configs_container.children = []
        
        # Create widget for each config
        config_widgets = []
        for i, config in enumerate(self.state.training_job_configs):
            widget = TrainingJobConfigWidget(self.state, i)
            config_widgets.append(widget.container)
        
        self.configs_container.children = config_widgets
        
        # Update button states
        self.launch_all_button.disabled = len(self.state.training_job_configs) == 0
    
    def _on_add_click(self, button):
        """Handle add config button click."""
        with self.status_output:
            self.status_output.clear_output()
            
            # Add a default config
            new_config = TrainingJobConfig()
            self.state.add_config(new_config)
            print("✓ Added new config")
    
    def _on_launch_all_click(self, button):
        """Handle launch all button click."""
        with self.status_output:
            self.status_output.clear_output()
            
            if not self.state.training_job_configs:
                print("No configs to launch")
                return
            
            print(f"Launching {len(self.state.training_job_configs)} jobs...")
            experiment_name = self.state.metadata.get('name', 'experiment')
            launched = self.state.launch_all_configs(experiment_name)
            print(f"✓ Launched {len(launched)} jobs")
    
    def display(self):
        """Display the widget."""
        display(self.container)


def generate_cells() -> list[dict]:
    """Generate notebook cells for config editing."""
    cells = []
    
    # Section header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "## 📝 Edit Training Configuration"
    })
    
    # Config editor cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# Edit training job configuration
from experiments.notebooks.widgets.config_widget import TrainingJobConfigWidget

# Edit the first (and only) config
if state.training_job_configs:
    config_widget = TrainingJobConfigWidget(state, config_index=0)
    config_widget.display()
else:
    print("No training job configs available")"""
    })
    
    # View current config
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """# View current configuration
if state.training_job_configs:
    config = state.training_job_configs[0]
    print("Current configuration:")
    print(f"  Curriculum: {config.curriculum}")
    print(f"  GPUs: {config.gpus}")
    print(f"  Nodes: {config.nodes}")
    print(f"  Spot: {config.spot}")
    print(f"  Skip Git Check: {config.skip_git_check}")
    print(f"  W&B Tags: {config.wandb_tags}")
    if config.additional_args:
        print(f"  Additional Args:")
        for arg in config.additional_args:
            print(f"    - {arg}")
else:
    print("No configurations available")"""
    })
    
    return cells
