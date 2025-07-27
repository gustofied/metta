"""State management utilities for reactive notebook widgets.

This module provides a thin layer over SkypilotService to enable reactive
widgets in Jupyter notebooks that automatically update when jobs are launched.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Callable
from experiments.training_job import TrainingJob, TrainingJobConfig
from experiments.skypilot_service import get_skypilot_service


class RunState:
    """Reactive state layer for notebook widgets.
    
    This class provides a thin wrapper over SkypilotService that enables
    reactive updates in Jupyter notebooks when jobs are launched or modified.
    """
    
    def __init__(self, 
                 training_jobs: Optional[List[TrainingJob]] = None,
                 training_job_configs: Optional[List[TrainingJobConfig]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 # Legacy parameters for backwards compatibility during migration
                 wandb_run_names: Optional[List[str]] = None,
                 skypilot_job_ids: Optional[List[str]] = None):
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
        self._training_job_configs: List[TrainingJobConfig] = training_job_configs or []
        
        # Handle new TrainingJob-based initialization
        if training_jobs:
            for job in training_jobs:
                # Ensure job has proper metadata
                if not hasattr(job, 'notes'):
                    job.notes = 'Pre-loaded from experiment'
                if not hasattr(job, 'timestamp'):
                    job.timestamp = self.metadata.get('created_at', datetime.now())
                
                # Add to service tracking
                self.service.add_job(job)
        
        # Handle legacy initialization for backwards compatibility
        elif wandb_run_names:
            for i, run_name in enumerate(wandb_run_names):
                job_id = skypilot_job_ids[i] if skypilot_job_ids and i < len(skypilot_job_ids) else None
                job = TrainingJob(name=run_name)
                job.job_id = job_id
                job.launched = True
                job.success = True
                job.notes = 'Pre-loaded from experiment'
                job.timestamp = self.metadata.get('created_at', datetime.now())
                
                # Add to service tracking
                self.service.add_job(job)
    
    @property
    def training_jobs(self) -> List[TrainingJob]:
        """Get all tracked training jobs from the service."""
        return self.service.get_tracked_jobs()
    
    @property
    def training_job_configs(self) -> List[TrainingJobConfig]:
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
        return [(job.name, job.job_id) for job in self.training_jobs if job.launched and job.job_id]
    
    @property
    def experiments(self) -> Dict[str, Any]:
        """Get experiments dictionary for backward compatibility."""
        exp_dict = {}
        for job in self.training_jobs:
            exp_dict[job.name] = {
                'job_id': job.job_id,
                'config': job.config.model_dump() if hasattr(job.config, 'model_dump') else {},
                'notes': getattr(job, 'notes', ''),
                'timestamp': getattr(job, 'timestamp', datetime.now())
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
        if not hasattr(job, 'timestamp'):
            job.timestamp = datetime.now()
        
        # Add to service
        self.service.add_job(job)
        
        print(f"Added job: {job.name}")
        if job.job_id:
            print(f"  Sky job: {job.job_id}")
            
        # Trigger reactive updates
        self._trigger_updates()
    
    def add_run(self, run_name: str, job_id: Optional[str] = None, 
                config: Optional[Dict] = None, notes: str = "") -> None:
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
            print(f"  {i+1}. {status} {job.name}{job_info}")
        
        launched_count = sum(1 for job in self.training_jobs if job.launched and job.job_id)
        if launched_count > 0:
            print(f"\nTotal jobs launched in this session: {launched_count}")
    
    def kill_all_jobs(self) -> None:
        """Kill all Sky jobs and trigger reactive updates."""
        # Get jobs with sky IDs
        killable_jobs = [job for job in self.training_jobs if job.launched and job.job_id]
        
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
        print(f"Added config for future launch")
        self._trigger_updates()
    
    def remove_config(self, index: int) -> None:
        """Remove a config by index.
        
        Args:
            index: Index of config to remove
        """
        if 0 <= index < len(self._training_job_configs):
            self._training_job_configs.pop(index)
            self._trigger_updates()
    
    def update_config(self, index: int, config: TrainingJobConfig) -> None:
        """Update a config at the given index.
        
        Args:
            index: Index of config to update
            config: New config
        """
        if 0 <= index < len(self._training_job_configs):
            self._training_job_configs[index] = config
            self._trigger_updates()
    
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
        self._training_job_configs = []
        self._trigger_updates()
        
        return launched_jobs


# Global instance that notebooks will use
_state: Optional[RunState] = None


def init_state(training_jobs: Optional[List[TrainingJob]] = None,
               training_job_configs: Optional[List[TrainingJobConfig]] = None,
               metadata: Optional[Dict[str, Any]] = None,
               # Legacy parameters
               wandb_run_names: Optional[List[str]] = None,
               skypilot_job_ids: Optional[List[str]] = None) -> RunState:
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
    _state = RunState(training_jobs=training_jobs, training_job_configs=training_job_configs,
                      metadata=metadata, wandb_run_names=wandb_run_names, skypilot_job_ids=skypilot_job_ids)
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


def add_run(run_name: str, job_id: Optional[str] = None, 
            config: Optional[Dict] = None, notes: str = "") -> None:
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