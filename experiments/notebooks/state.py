"""State management utilities for reactive notebook widgets.

This module provides a thin layer over SkypilotService to enable reactive
widgets in Jupyter notebooks that automatically update when jobs are launched.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Callable
from experiments.training_job import TrainingJob, TrainingJobConfig
from experiments.skypilot_service import get_skypilot_service


class ReactiveConfigList(list):
    """A list that triggers state updates when modified."""

    def __init__(self, state: "RunState", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = state

    def append(self, item: TrainingJobConfig) -> None:
        super().append(item)
        self._state._trigger_updates()

    def extend(self, items: List[TrainingJobConfig]) -> None:
        super().extend(items)
        self._state._trigger_updates()

    def insert(self, index: int, item: TrainingJobConfig) -> None:
        super().insert(index, item)
        self._state._trigger_updates()

    def remove(self, item: TrainingJobConfig) -> None:
        super().remove(item)
        self._state._trigger_updates()

    def pop(self, index: int = -1) -> TrainingJobConfig:
        result = super().pop(index)
        self._state._trigger_updates()
        return result

    def clear(self) -> None:
        super().clear()
        self._state._trigger_updates()

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        self._state._trigger_updates()

    def __delitem__(self, index):
        super().__delitem__(index)
        self._state._trigger_updates()


class RunState:
    """Reactive state layer for notebook widgets.

    This class provides a thin wrapper over SkypilotService that enables
    reactive updates in Jupyter notebooks when jobs are launched or modified.
    """

    def __init__(
        self,
        training_jobs: Optional[List[TrainingJob]] = None,
        training_job_configs: Optional[List[TrainingJobConfig]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Legacy parameters for backwards compatibility during migration
        wandb_run_names: Optional[List[str]] = None,
        skypilot_job_ids: Optional[List[str]] = None,
    ):
        """Initialize run state with optional pre-loaded jobs.

        Args:
            training_jobs: List of TrainingJob objects to track
            training_job_configs: List of TrainingJobConfig objects for unlaunched jobs
            metadata: Additional metadata from experiments
            wandb_run_names: (deprecated) Pre-populated wandb run names from experiment
            skypilot_job_ids: (deprecated) Pre-populated sky job IDs from experiment
        """
        self.service = get_skypilot_service()
        self.metadata = metadata or {}
        self._update_callbacks: List[Callable] = []

        # Create reactive configs list
        self._training_job_configs = ReactiveConfigList(self)
        if training_job_configs:
            self._training_job_configs.extend(training_job_configs)

        # Handle new TrainingJob-based initialization
        if training_jobs:
            for job in training_jobs:
                # Ensure job has proper metadata
                if not hasattr(job, "notes"):
                    job.notes = "Pre-loaded from experiment"
                if not hasattr(job, "timestamp"):
                    job.timestamp = self.metadata.get("created_at", datetime.now())

                # Add to service tracking
                self.service.add_job(job)

        # Handle legacy initialization for backwards compatibility
        elif wandb_run_names:
            for i, run_name in enumerate(wandb_run_names):
                job_id = (
                    skypilot_job_ids[i]
                    if skypilot_job_ids and i < len(skypilot_job_ids)
                    else None
                )
                job = TrainingJob(name=run_name)
                job.job_id = job_id
                job.launched = True
                job.success = True
                job.notes = "Pre-loaded from experiment"
                job.timestamp = self.metadata.get("created_at", datetime.now())

                # Add to service tracking
                self.service.add_job(job)

    @property
    def training_jobs(self) -> List[TrainingJob]:
        """Get all tracked training jobs from the service."""
        return self.service.get_tracked_jobs()

    @property
    def training_job_configs(self) -> ReactiveConfigList:
        """Get all training job configurations (unlaunched jobs)."""
        return self._training_job_configs

    @property
    def wandb_run_names(self) -> List[str]:
        """Get list of wandb run names from training jobs."""
        return [job.name for job in self.training_jobs]

    @property
    def skypilot_job_ids(self) -> List[str]:
        """Get list of sky job IDs from training jobs."""
        return [job.job_id for job in self.training_jobs if job.job_id]

    @property
    def all_launched_jobs(self) -> List[Tuple[str, str]]:
        """Get all launched jobs as (name, job_id) tuples."""
        return [
            (job.name, job.job_id)
            for job in self.training_jobs
            if job.launched and job.job_id
        ]

    @property
    def experiments(self) -> Dict[str, Any]:
        """Get experiments dictionary for backward compatibility."""
        exp_dict = {}
        for job in self.training_jobs:
            exp_dict[job.name] = {
                "job_id": job.job_id,
                "config": job.config.model_dump()
                if hasattr(job.config, "model_dump")
                else {},
                "notes": getattr(job, "notes", ""),
                "timestamp": getattr(job, "timestamp", datetime.now()),
            }
        return exp_dict

    def register_update_callback(self, callback: Callable) -> None:
        """Register a callback to be called when state changes."""
        self._update_callbacks.append(callback)

    def _trigger_updates(self) -> None:
        """Trigger all registered update callbacks."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in update callback: {e}")

    def add_job(self, job: TrainingJob) -> None:
        """Add a TrainingJob to track in this session and trigger reactive updates.

        Args:
            job: TrainingJob object to add
        """
        # Ensure job has timestamp
        if not hasattr(job, "timestamp"):
            job.timestamp = datetime.now()

        # Add to service
        self.service.add_job(job)

        print(f"Added job: {job.name}")
        if job.job_id:
            print(f"  Sky job: {job.job_id}")

        # Trigger reactive updates
        self._trigger_updates()

    def add_run(
        self,
        run_name: str,
        job_id: Optional[str] = None,
        config: Optional[Dict] = None,
        notes: str = "",
    ) -> None:
        """Legacy method: Add a run by name. Use add_job() with TrainingJob instead.

        Args:
            run_name: Name of the run
            job_id: Optional Sky job ID
            config: Optional config dictionary
            notes: Optional notes
        """
        # Create TrainingJob
        job = TrainingJob(name=run_name)
        job.job_id = job_id
        if job_id:
            job.launched = True
            job.success = True

        # Store additional metadata
        job.notes = notes
        job.timestamp = datetime.now()

        # If config is provided as dict, try to convert to TrainingJobConfig
        if config:
            from experiments.training_job import TrainingJobConfig

            try:
                job.config = TrainingJobConfig(**config)
            except:
                pass

        # Use the new add_job method
        self.add_job(job)

    def list_runs(self) -> None:
        """List all runs in this session."""
        if not self.training_jobs:
            print("No runs tracked yet")
            return

        print(f"Tracking {len(self.training_jobs)} runs:")
        for i, job in enumerate(self.training_jobs):
            job_info = f" (job: {job.job_id})" if job.job_id else ""
            status = "✓" if job.success else "✗" if job.launched else "○"
            print(f"  {i + 1}. {status} {job.name}{job_info}")

        launched_count = sum(
            1 for job in self.training_jobs if job.launched and job.job_id
        )
        if launched_count > 0:
            print(f"\nTotal jobs launched in this session: {launched_count}")

    def kill_all_jobs(self) -> None:
        """Kill all Sky jobs and trigger reactive updates."""
        # Get jobs with sky IDs
        killable_jobs = [
            job for job in self.training_jobs if job.launched and job.job_id
        ]

        if not killable_jobs:
            print("No jobs to kill")
            return

        print(f"Killing {len(killable_jobs)} jobs...")

        killed = 0
        failed = 0

        for job in killable_jobs:
            try:
                success = self.service.cancel_job(job.job_id)
                if success:
                    print(f"✓ Killed {job.job_id} ({job.name})")
                    killed += 1
                else:
                    print(f"✗ Failed to kill {job.job_id} ({job.name})")
                    failed += 1
            except Exception as e:
                print(f"✗ Error killing {job.job_id} ({job.name}): {e}")
                failed += 1

        print(f"\nSummary: {killed} killed, {failed} failed")

        # Trigger reactive updates
        self._trigger_updates()

    def add_config(self, config: TrainingJobConfig) -> None:
        """Add a TrainingJobConfig for later launching.

        Args:
            config: TrainingJobConfig to add
        """
        self._training_job_configs.append(config)
        print("Added config for future launch")
        self._trigger_updates()

    def remove_config(self, index: int) -> None:
        """Remove a config by index.

        Args:
            index: Index of config to remove
        """
        if 0 <= index < len(self._training_job_configs):
            self._training_job_configs.pop(index)

    def update_config(self, index: int, config: TrainingJobConfig) -> None:
        """Update a config at the given index.

        Args:
            index: Index of config to update
            config: New config
        """
        if 0 <= index < len(self._training_job_configs):
            self._training_job_configs[index] = config

    def launch_all_configs(self, experiment_name: str) -> List[TrainingJob]:
        """Launch all training job configs and move them to launched jobs.

        Args:
            experiment_name: Base name for the jobs

        Returns:
            List of launched TrainingJob objects
        """
        launched_jobs = []

        for i, config in enumerate(self._training_job_configs):
            job_name = f"{experiment_name}_job_{i}"
            job = TrainingJob(name=job_name, config=config)

            # Launch the job
            success = job.launch()
            if success:
                self.add_job(job)
                launched_jobs.append(job)
            else:
                print(f"Failed to launch job {job_name}")

        # Clear configs after launching
        self._training_job_configs.clear()

        return launched_jobs


# Global instance that notebooks will use
_state: Optional[RunState] = None


def init_state(
    training_jobs: Optional[List[TrainingJob]] = None,
    training_job_configs: Optional[List[TrainingJobConfig]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    # Legacy parameters
    wandb_run_names: Optional[List[str]] = None,
    skypilot_job_ids: Optional[List[str]] = None,
) -> RunState:
    """Initialize the global run state.

    This is called automatically by generated notebooks.

    Args:
        training_jobs: List of TrainingJob objects to track
        training_job_configs: List of TrainingJobConfig objects for unlaunched jobs
        metadata: Additional metadata
        wandb_run_names: (deprecated) Legacy parameter
        skypilot_job_ids: (deprecated) Legacy parameter
    """
    global _state
    _state = RunState(
        training_jobs=training_jobs,
        training_job_configs=training_job_configs,
        metadata=metadata,
        wandb_run_names=wandb_run_names,
        skypilot_job_ids=skypilot_job_ids,
    )
    return _state


def get_state() -> RunState:
    """Get the global run state instance."""
    global _state
    if _state is None:
        _state = RunState()
    return _state


# Convenience functions that operate on the global state
def add_job(job: TrainingJob) -> None:
    """Add a TrainingJob to track in this session.

    Args:
        job: TrainingJob object to add
    """
    get_state().add_job(job)


def add_run(
    run_name: str,
    job_id: Optional[str] = None,
    config: Optional[Dict] = None,
    notes: str = "",
) -> None:
    """Legacy function: Add a run to track in this session.

    Use add_job() with a TrainingJob object instead.
    """
    get_state().add_run(run_name, job_id, config, notes)


def list_runs() -> None:
    """List all runs in this session."""
    get_state().list_runs()


def kill_all_jobs() -> None:
    """Kill all Sky jobs launched in this session."""
    get_state().kill_all_jobs()


# Convenience functions for accessing state
def get_training_jobs() -> List[TrainingJob]:
    """Get list of all training jobs."""
    return get_state().training_jobs


def get_wandb_run_names() -> List[str]:
    """Get list of wandb run names."""
    return get_state().wandb_run_names


def get_skypilot_job_ids() -> List[str]:
    """Get list of sky job IDs."""
    return get_state().skypilot_job_ids


def get_experiments() -> Dict[str, Any]:
    """Get experiments dictionary."""
    return get_state().experiments


# Alias for backwards compatibility
def add_training_job(job: TrainingJob) -> None:
    """Alias for add_job - use add_job() instead."""
    add_job(job)


# Simplified one-liner setup function
def print_configs(configs: List[TrainingJobConfig]) -> None:
    """Print a nice tree representation of training job configs.

    Args:
        configs: List of TrainingJobConfig objects to display
    """
    if not configs:
        print("No configs defined")
        return

    print(f"📋 Training Job Configs ({len(configs)} total)\n")

    for i, config in enumerate(configs):
        print(f"[{i}] Job Config")
        print(f"    ├─ curriculum: {config.curriculum}")
        print(f"    ├─ gpus: {config.gpus}")
        print(f"    ├─ nodes: {config.nodes}")
        print(f"    ├─ spot: {config.spot}")
        print(f"    ├─ git_check: {config.git_check}")

        if config.wandb_tags:
            tags_str = ", ".join(config.wandb_tags)
            print(f"    ├─ wandb_tags: [{tags_str}]")
        else:
            print("    ├─ wandb_tags: []")

        if config.additional_args:
            print("    └─ additional_args:")
            for j, arg in enumerate(config.additional_args):
                is_last = j == len(config.additional_args) - 1
                prefix = "       └─" if is_last else "       ├─"
                print(f"{prefix} {arg}")
        else:
            print("    └─ additional_args: []")

        if i < len(configs) - 1:
            print()  # Empty line between configs


def print_jobs(state: 'RunState') -> None:
    """Print a formatted view of all jobs and configs."""
    from datetime import datetime
    
    # Print header
    print("\n" + "="*80)
    print("🚀 TRAINING JOBS STATUS")
    print("="*80 + "\n")
    
    # Count jobs
    total_jobs = len(state.training_jobs)
    running_jobs = [j for j in state.training_jobs if j.job_id and not getattr(j, 'cancelled', False)]
    completed_jobs = [j for j in state.training_jobs if getattr(j, 'success', False)]
    
    # Summary line
    print(f"📊 Summary: {total_jobs} total jobs | {len(running_jobs)} running | {len(completed_jobs)} completed\n")
    
    # Print configs first if any
    if state.training_job_configs:
        print("📋 CONFIGS READY TO LAUNCH:")
        print("-" * 60)
        for i, config in enumerate(state.training_job_configs):
            tags = f" [{', '.join(config.wandb_tags)}]" if config.wandb_tags else ""
            spot = " (spot)" if config.spot else ""
            print(f"  [{i}] {config.curriculum}")
            print(f"      └─ {config.gpus} GPU × {config.nodes} node{spot}{tags}")
        print()
    
    # Print active jobs
    if state.training_jobs:
        print("🏃 ACTIVE JOBS:")
        print("-" * 60)
        for job in state.training_jobs:
            # Status icon
            if getattr(job, 'cancelled', False):
                status = "🛑"
                color = "\033[90m"  # Gray
            elif job.job_id and not getattr(job, 'cancelled', False):
                status = "🟢"
                color = "\033[92m"  # Green
            elif job.success:
                status = "✅"
                color = "\033[94m"  # Blue
            else:
                status = "❌"
                color = "\033[91m"  # Red
            
            # Print job info
            job_id = f"({job.job_id})" if job.job_id else "(no ID)"
            timestamp = getattr(job, 'timestamp', None)
            time_str = timestamp.strftime("%H:%M") if timestamp and hasattr(timestamp, 'strftime') else ""
            
            print(f"  {status} {color}{job.name}\033[0m {job_id}")
            if time_str:
                print(f"      └─ Started: {time_str}")
    else:
        print("💤 No active jobs\n")
    
    # Footer with commands
    print("\n" + "-"*80)
    print("💡 Commands:")
    print("  • launch_all()     - Launch all configs")
    print("  • kill_all()       - Kill all running jobs")
    print("  • print_jobs(state) - Refresh this view")
    print("="*80 + "\n")


def launch_all(state: 'RunState') -> None:
    """Launch all training job configs."""
    if not state.training_job_configs:
        print("❌ No configs to launch!")
        print("   Add configs first: configs.append(TrainingJobConfig(curriculum='arena'))")
        return
    
    from experiments.training_job import TrainingJob
    from datetime import datetime
    
    print(f"\n🚀 Starting launch of {len(state.training_job_configs)} training jobs...")
    print("=" * 60)
    
    experiment_name = state.metadata.get('name', 'experiment')
    launched_jobs = []
    failed_jobs = []
    
    for i, config in enumerate(state.training_job_configs):
        job_name = f"{experiment_name}_job_{i}"
        job = TrainingJob(name=job_name, config=config)
        
        # Print config details
        print(f"\n[{i + 1}/{len(state.training_job_configs)}] Launching: {job_name}")
        print(f"  📋 Curriculum: {config.curriculum}")
        print(f"  🖥️  Resources: {config.gpus} GPU(s) × {config.nodes} node(s)")
        if config.spot:
            print(f"  💰 Using spot instances")
        if config.wandb_tags:
            print(f"  🏷️  Tags: {', '.join(config.wandb_tags)}")
        if config.additional_args:
            print(f"  ⚙️  Additional args: {len(config.additional_args)} custom settings")
        
        print(f"\n  ⏳ Submitting to Skypilot...")
        start_time = datetime.now()
        
        try:
            if job.launch():
                elapsed = (datetime.now() - start_time).total_seconds()
                state.add_job(job)
                launched_jobs.append(job)
                print(f"  ✅ SUCCESS! Launched in {elapsed:.1f}s")
                print(f"  📍 Job ID: {job.job_id}")
                print(f"  🔗 Run name: {job.name}")
            else:
                elapsed = (datetime.now() - start_time).total_seconds()
                failed_jobs.append((job_name, "Launch returned False"))
                print(f"  ❌ FAILED after {elapsed:.1f}s")
                print(f"  ⚠️  Check your Skypilot setup and credentials")
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            failed_jobs.append((job_name, str(e)))
            print(f"  ❌ ERROR after {elapsed:.1f}s: {str(e)}")
            print(f"  ⚠️  This might be a configuration or permission issue")
        
        print("-" * 60)
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 LAUNCH SUMMARY:")
    print(f"  ✅ Successful: {len(launched_jobs)}/{len(state.training_job_configs)}")
    print(f"  ❌ Failed: {len(failed_jobs)}/{len(state.training_job_configs)}")
    
    if launched_jobs:
        print(f"\n✅ Successfully launched jobs:")
        for job in launched_jobs:
            print(f"  - {job.name} (ID: {job.job_id})")
    
    if failed_jobs:
        print(f"\n❌ Failed jobs:")
        for job_name, error in failed_jobs:
            print(f"  - {job_name}: {error}")
    
    # Clear configs if all successful
    if len(launched_jobs) == len(state.training_job_configs):
        state.training_job_configs.clear()
        print(f"\n✨ All configs launched successfully and cleared")
    elif launched_jobs:
        print(f"\n⚠️  Some configs failed - successful ones have been cleared")
        # Remove only successful configs
        remaining_configs = []
        for i, config in enumerate(state.training_job_configs):
            job_name = f"{experiment_name}_job_{i}"
            if not any(job.name == job_name for job in launched_jobs):
                remaining_configs.append(config)
        state.training_job_configs.clear()
        state.training_job_configs.extend(remaining_configs)
    
    print(f"\n💡 Use 'print_jobs(state)' to see current status")
    print(f"{'=' * 60}\n")


def kill_all(state: 'RunState') -> None:
    """Kill all running jobs."""
    running = [j for j in state.training_jobs if j.job_id and not getattr(j, 'cancelled', False)]
    
    if not running:
        print("❌ No running jobs to kill")
        return
    
    print(f"\n🛑 Killing {len(running)} jobs...")
    print("="*60)
    
    killed = 0
    failed = 0
    
    for job in running:
        print(f"\n• {job.name} ({job.job_id})")
        try:
            if state.service.cancel_job(job.job_id):
                job.cancelled = True
                killed += 1
                print("  ✅ Killed")
            else:
                failed += 1
                print("  ❌ Failed")
        except Exception as e:
            failed += 1
            print(f"  ❌ Error: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"✅ Killed: {killed} | ❌ Failed: {failed}")
    print()


def setup_notebook(experiment_data: Optional[Dict[str, Any]] = None) -> tuple:
    """One-liner setup for notebooks that initializes everything.

    This function initializes everything needed for a notebook in one line:
    - Sets up autoreload
    - Imports necessary modules
    - Initializes the state
    - Creates and displays the main widgets

    Args:
        experiment_data: Dict with 'jobs', 'configs', and 'metadata' keys

    Returns:
        Tuple of (state, configs)
    """
    # Enable autoreload
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")

    # Parse experiment data
    training_jobs = None
    training_job_configs = None
    metadata = {}

    if experiment_data:
        # Reconstruct training jobs
        if experiment_data.get("jobs"):
            training_jobs = []
            for job_data in experiment_data["jobs"]:
                job = TrainingJob(name=job_data["name"])
                job.job_id = job_data.get("job_id")
                job.launched = job_data.get("launched", False)
                job.success = job_data.get("success", False)
                if "notes" in job_data:
                    job.notes = job_data["notes"]
                training_jobs.append(job)

        # Reconstruct configs
        if experiment_data.get("configs"):
            training_job_configs = []
            for config_data in experiment_data["configs"]:
                training_job_configs.append(TrainingJobConfig(**config_data))

        metadata = experiment_data.get("metadata", {})

    # Initialize state
    state = init_state(
        training_jobs=training_jobs,
        training_job_configs=training_job_configs,
        metadata=metadata,
    )

    # Store original configs for reset functionality
    state._original_configs = [TrainingJobConfig(**config.model_dump()) for config in training_job_configs]
    
    # Create a reference to the configs list that's connected to state
    configs = state.training_job_configs

    # Display summary
    job_count = len(state.training_jobs)
    config_count = len(configs)
    print(f"✓ Notebook initialized: {job_count} jobs, {config_count} configs")
    print("\n📝 Edit configs directly in Python:")
    print("  configs[0].gpus = 4")
    print("  configs[0].curriculum = 'arena'")
    print("  reset_configs(state)  # Reset to original")
    print("\n🔍 View your jobs and configs:")
    print("  print_jobs(state)")
    print("  print_configs(configs)")
    print("\n🚀 Launch and control:")
    print("  launch_all(state)")
    print("  kill_all(state)")

    return state, configs


def print_wandb_runs(state: 'RunState') -> None:
    """Print available W&B runs for analysis."""
    if not state.wandb_run_names:
        print("❌ No W&B runs available")
        print("   Launch some training jobs first!")
        return
    
    print("\n" + "="*60)
    print("📊 WANDB RUNS AVAILABLE FOR ANALYSIS")
    print("="*60)
    
    for i, run_name in enumerate(state.wandb_run_names):
        job = None
        for j in state.training_jobs:
            if j.name == run_name:
                job = j
                break
        
        status = "🟢 Active" if job and job.job_id else "✅ Complete"
        print(f"\n[{i}] {run_name}")
        print(f"    └─ Status: {status}")
        if job and job.job_id:
            print(f"    └─ Job ID: {job.job_id}")
    
    print("\n" + "-"*60)
    print("💡 Analysis commands:")
    print("  • plot_sps(state, run_indices=[0,1,2])  - Plot steps per second")
    print("  • plot_metrics(state, 'overview/reward') - Plot any metric")
    print("  • get_wandb_data(state, run_index=0)    - Get raw data")
    print("="*60 + "\n")


def plot_sps(state: 'RunState', run_indices: List[int] = None, last_n_hours: int = 24) -> None:
    """Plot steps per second for selected runs.
    
    Args:
        state: The run state
        run_indices: List of run indices to plot (None = all)
        last_n_hours: How many hours of data to show
    """
    if not state.wandb_run_names:
        print("❌ No runs to analyze")
        return
    
    # Select runs
    if run_indices is None:
        run_indices = list(range(len(state.wandb_run_names)))
    
    selected_runs = [state.wandb_run_names[i] for i in run_indices if i < len(state.wandb_run_names)]
    
    if not selected_runs:
        print("❌ No valid run indices selected")
        return
    
    print(f"\n📈 Plotting SPS for {len(selected_runs)} runs...")
    
    # Import here to avoid issues if wandb not installed
    try:
        from experiments.wandb_service import get_wandb_service
        import matplotlib.pyplot as plt
        
        service = get_wandb_service()
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        for run_name in selected_runs:
            try:
                df = service.get_run_data(run_name, metrics=['trainer/fps'])
                if df is not None and not df.empty:
                    plt.plot(df.index, df['trainer/fps'], label=run_name, linewidth=2)
                else:
                    print(f"  ⚠️  No data for {run_name}")
            except Exception as e:
                print(f"  ❌ Error loading {run_name}: {str(e)}")
        
        plt.xlabel('Time')
        plt.ylabel('Steps per Second')
        plt.title(f'Training Speed (last {last_n_hours}h)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("✅ Plot complete!")
        
    except ImportError:
        print("❌ W&B not installed. Run: pip install wandb")
    except Exception as e:
        print(f"❌ Error creating plot: {str(e)}")


def plot_metrics(state: 'RunState', metric: str, run_indices: List[int] = None, smooth: float = 0.8) -> None:
    """Plot any W&B metric for selected runs.
    
    Args:
        state: The run state
        metric: Metric name (e.g. 'overview/reward', 'trainer/loss')
        run_indices: List of run indices to plot (None = all)
        smooth: Smoothing factor (0-1)
    """
    if not state.wandb_run_names:
        print("❌ No runs to analyze")
        return
    
    # Select runs
    if run_indices is None:
        run_indices = list(range(len(state.wandb_run_names)))
    
    selected_runs = [state.wandb_run_names[i] for i in run_indices if i < len(state.wandb_run_names)]
    
    print(f"\n📈 Plotting {metric} for {len(selected_runs)} runs...")
    
    try:
        from experiments.wandb_service import get_wandb_service
        import matplotlib.pyplot as plt
        import numpy as np
        
        service = get_wandb_service()
        
        plt.figure(figsize=(12, 6))
        
        for run_name in selected_runs:
            try:
                df = service.get_run_data(run_name, metrics=[metric])
                if df is not None and metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        # Apply smoothing
                        if smooth > 0:
                            smoothed = values.ewm(alpha=1-smooth).mean()
                            plt.plot(df.index, smoothed, label=run_name, linewidth=2)
                        else:
                            plt.plot(df.index, values, label=run_name, linewidth=2)
                else:
                    print(f"  ⚠️  No {metric} data for {run_name}")
            except Exception as e:
                print(f"  ❌ Error loading {run_name}: {str(e)}")
        
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.title(f'{metric} over time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("✅ Plot complete!")
        
    except ImportError:
        print("❌ Required packages not installed. Run: pip install wandb matplotlib")
    except Exception as e:
        print(f"❌ Error creating plot: {str(e)}")


def list_replays(state: 'RunState', run_index: int = 0) -> None:
    """List available replays for a run.
    
    Args:
        state: The run state
        run_index: Index of the run to check
    """
    if not state.wandb_run_names:
        print("❌ No runs available")
        return
    
    if run_index >= len(state.wandb_run_names):
        print(f"❌ Invalid run index. Available: 0-{len(state.wandb_run_names)-1}")
        return
    
    run_name = state.wandb_run_names[run_index]
    print(f"\n🎬 Checking replays for: {run_name}")
    print("="*60)
    
    try:
        from experiments.notebooks.widgets.replay_widget import get_available_replays
        
        replays = get_available_replays(run_name)
        
        if not replays:
            print("  ❌ No replays found")
            print("  💡 Replays are saved during evaluation phases")
        else:
            print(f"  ✅ Found {len(replays)} replays:\n")
            
            # Show last 10 replays
            for replay in replays[-10:]:
                print(f"  [{replay['step']}] {replay['label']}")
                if 'size' in replay:
                    size_mb = replay['size'] / (1024 * 1024)
                    print(f"       └─ Size: {size_mb:.1f} MB")
            
            if len(replays) > 10:
                print(f"\n  ... and {len(replays) - 10} more")
            
            print("\n" + "-"*60)
            print("💡 Commands:")
            print(f"  • show_replay(state, {run_index}, step='last')     - Show latest")
            print(f"  • show_replay(state, {run_index}, step={replays[-1]['step']})  - Show specific step")
            
    except Exception as e:
        print(f"❌ Error checking replays: {str(e)}")
    
    print("="*60 + "\n")


def show_replay(state: 'RunState', run_index: int = 0, step: str = 'last', width: int = 800, height: int = 600) -> None:
    """Show a replay for a specific run and step.
    
    Args:
        state: The run state
        run_index: Index of the run
        step: Step number or 'last' for most recent
        width: Display width
        height: Display height
    """
    if not state.wandb_run_names:
        print("❌ No runs available")
        return
    
    if run_index >= len(state.wandb_run_names):
        print(f"❌ Invalid run index. Available: 0-{len(state.wandb_run_names)-1}")
        return
    
    run_name = state.wandb_run_names[run_index]
    
    try:
        from experiments.notebooks.widgets.replay_widget import show_replay as _show_replay
        
        print(f"🎬 Loading replay for {run_name} at step {step}...")
        _show_replay(run_name, step=step, width=width, height=height)
        
    except Exception as e:
        print(f"❌ Error showing replay: {str(e)}")


def export_notebook() -> None:
    """Export the current notebook to HTML."""
    print("\n📤 EXPORT NOTEBOOK TO HTML")
    print("="*60)
    
    print("\nTo export this notebook:")
    print("1. Save the notebook (Cmd/Ctrl + S)")
    print("2. Run one of these commands in terminal:\n")
    
    print("   # Basic export:")
    print("   jupyter nbconvert --to html notebook.ipynb")
    print()
    print("   # With code cells hidden:")
    print("   jupyter nbconvert --to html --no-input notebook.ipynb")
    print()
    print("   # Custom template:")
    print("   jupyter nbconvert --to html --template lab notebook.ipynb")
    
    print("\n" + "-"*60)
    print("💡 You can also export from Jupyter:")
    print("   File → Download as → HTML")
    print("="*60 + "\n")


def reset_configs(state):
    """Reset configs to their original values from setup_notebook.
    
    Args:
        state: The RunState object
    """
    if hasattr(state, '_original_configs'):
        # Clear current configs
        state.training_job_configs.clear()
        
        # Restore original configs
        for config in state._original_configs:
            state.training_job_configs.append(TrainingJobConfig(**config.model_dump()))
        
        print("✅ Configs reset to original values")
        print_configs(state.training_job_configs)
    else:
        print("⚠️  No original configs found to reset to")
