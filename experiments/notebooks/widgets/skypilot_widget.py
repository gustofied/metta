"""Skypilot widgets and notebook generation."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display
import pandas as pd

from experiments.notebooks.widget import ReactiveWidget
from experiments.training_job import TrainingJob


class SkypilotWidget(ReactiveWidget):
    """Reactive widget for displaying training job status with Sky history."""

    def __init__(
        self,
        state: "RunState",
        show_metrics: Optional[List[str]] = None,
        show_all_sky_jobs: bool = False,
    ):
        """Initialize status widget.

        Args:
            state: The RunState instance to monitor
            show_metrics: Optional list of metrics to display
            show_all_sky_jobs: Show all Sky jobs, not just current session
        """
        # Don't use title from parent since we'll create custom header
        super().__init__(state, title=None)
        self.show_metrics = show_metrics
        self.show_all_sky_jobs = show_all_sky_jobs
        self._setup_controls()
        self._setup_styles()

    def _setup_styles(self) -> None:
        """Set up custom CSS styles."""
        self.styles = widgets.HTML("""
        <style>
        .status-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px 10px 0 0;
            margin-bottom: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-header h2 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }
        .status-header p {
            margin: 5px 0 0 0;
            opacity: 0.9;
            font-size: 14px;
        }
        .status-container {
            border: 1px solid #e1e4e8;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 20px;
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .status-empty {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        .status-empty h3 {
            font-size: 20px;
            margin-bottom: 10px;
            color: #495057;
        }
        .status-controls {
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .job-count-badge {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }
        </style>
        """)

    def _setup_controls(self) -> None:
        """Set up control widgets."""
        self.toggle_btn = widgets.ToggleButton(
            value=self.show_all_sky_jobs,
            description="Show All Sky Jobs",
            disabled=False,
            button_style="info",
            tooltip="Toggle between session jobs and all Sky jobs",
            icon="history",
            layout=widgets.Layout(width="200px"),
        )
        self.toggle_btn.observe(self._on_toggle_change, "value")

        self.refresh_btn = widgets.Button(
            description="Refresh",
            button_style="primary",
            icon="refresh",
            layout=widgets.Layout(width="100px"),
        )
        self.refresh_btn.on_click(lambda b: self.update())

        # Job selection tracking
        self.selected_jobs = []
        self.job_checkboxes = {}  # job_id -> checkbox widget

        # Cancel button
        self.cancel_btn = widgets.Button(
            description="Cancel Selected",
            button_style="danger",
            icon="stop",
            layout=widgets.Layout(width="150px"),
            disabled=True,
        )
        self.cancel_btn.on_click(self._cancel_selected)

        # Launch all button
        self.launch_all_btn = widgets.Button(
            description="Launch All Configs",
            button_style="success",
            icon="rocket",
            layout=widgets.Layout(width="180px"),
        )
        self.launch_all_btn.on_click(self._launch_all)

        # Kill all button
        self.kill_all_btn = widgets.Button(
            description="Kill All Jobs",
            button_style="danger",
            icon="times-circle",
            layout=widgets.Layout(width="150px"),
        )
        self.kill_all_btn.on_click(self._kill_all)

    def _on_toggle_change(self, change) -> None:
        """Handle toggle button change."""
        self.show_all_sky_jobs = change["new"]
        self.update()

    def _on_job_selection_change(self, change, job_id: str) -> None:
        """Handle job checkbox selection change."""
        if change["new"]:
            if job_id not in self.selected_jobs:
                self.selected_jobs.append(job_id)
        else:
            if job_id in self.selected_jobs:
                self.selected_jobs.remove(job_id)

        # Enable/disable cancel button based on selection
        self.cancel_btn.disabled = len(self.selected_jobs) == 0

    def _cancel_selected(self, button) -> None:
        """Cancel all selected jobs."""
        if not self.selected_jobs:
            return

        # Disable button during operation
        self.cancel_btn.disabled = True
        original_text = self.cancel_btn.description
        self.cancel_btn.description = "Cancelling..."
        self.cancel_btn.button_style = "warning"
        
        # Create output widget
        cancel_output = widgets.Output()
        display(cancel_output)
        
        with cancel_output:
            print(f"🛑 Cancelling {len(self.selected_jobs)} selected jobs...\n")
            
            cancelled_count = 0
            failed_count = 0
            
            for job_id in self.selected_jobs:
                # Find job name
                job_name = "Unknown"
                for job in self.state.training_jobs:
                    if job.job_id == job_id:
                        job_name = job.name
                        break
                
                try:
                    if self.state.service.cancel_job(job_id):
                        cancelled_count += 1
                        print(f"✅ Cancelled {job_name} ({job_id})")
                    else:
                        failed_count += 1
                        print(f"❌ Failed to cancel {job_name} ({job_id})")
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Error cancelling {job_name}: {e}")

            print(f"\nSummary: {cancelled_count} cancelled, {failed_count} failed")

        # Clear selection and restore button
        self.selected_jobs = []
        self.cancel_btn.disabled = True
        self.cancel_btn.description = original_text
        self.cancel_btn.button_style = "danger"
        
        self.update()

    def _launch_all(self, button) -> None:
        """Launch all training job configs."""
        if not self.state.training_job_configs:
            print("No training job configs to launch")
            return

        # Disable all launch-related buttons
        self.launch_all_btn.disabled = True
        self.launch_all_btn.description = "Launching..."
        self.launch_all_btn.button_style = "warning"
        
        # Create output widget for launch progress
        launch_output = widgets.Output()
        display(launch_output)
        
        with launch_output:
            print(f"🚀 Starting launch of {len(self.state.training_job_configs)} training jobs...\n")
            print("=" * 60)
            
            from experiments.training_job import TrainingJob
            from datetime import datetime

            experiment_name = self.state.metadata.get("name", "experiment")
            
            launched_jobs = []
            failed_jobs = []
            
            for i, config in enumerate(self.state.training_job_configs):
                job_name = f"{experiment_name}_job_{i}"
                job = TrainingJob(name=job_name, config=config)
                
                # Print config details
                print(f"\n[{i + 1}/{len(self.state.training_job_configs)}] Launching: {job_name}")
                print(f"  📋 Curriculum: {config.curriculum}")
                print(f"  🖥️  Resources: {config.gpus} GPU(s) × {config.nodes} node(s)")
                if config.spot:
                    print(f"  💰 Using spot instances")
                if config.wandb_tags:
                    print(f"  🏷️  Tags: {', '.join(config.wandb_tags)}")
                
                print(f"\n  ⏳ Submitting to Skypilot...")
                start_time = datetime.now()
                
                try:
                    if job.launch():
                        elapsed = (datetime.now() - start_time).total_seconds()
                        self.state.add_job(job)
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
            print(f"  ✅ Successful: {len(launched_jobs)}/{len(self.state.training_job_configs)}")
            print(f"  ❌ Failed: {len(failed_jobs)}/{len(self.state.training_job_configs)}")
            
            if launched_jobs:
                print(f"\n✅ Successfully launched jobs:")
                for job in launched_jobs:
                    print(f"  - {job.name} (ID: {job.job_id})")
            
            if failed_jobs:
                print(f"\n❌ Failed jobs:")
                for job_name, error in failed_jobs:
                    print(f"  - {job_name}: {error}")
            
            print(f"\n💡 Use 'sky status' in terminal to check job status")
            print(f"{'=' * 60}\n")
        
        # Re-enable button and update UI
        self.launch_all_btn.disabled = False
        self.launch_all_btn.description = "Launch All Configs"
        self.launch_all_btn.button_style = "success"
        
        # Clear configs if all successful
        if len(launched_jobs) == len(self.state.training_job_configs):
            self.state.training_job_configs.clear()
        
        # Update display
        self.update()

    def _kill_all(self, button) -> None:
        """Kill all running jobs."""
        running_jobs = [
            j for j in self.state.training_jobs if j.job_id and not j.cancelled
        ]

        if not running_jobs:
            print("No running jobs to kill")
            return

        # Disable button during operation
        self.kill_all_btn.disabled = True
        self.kill_all_btn.description = "Killing..."
        self.kill_all_btn.button_style = "warning"
        
        # Create output widget for kill progress
        kill_output = widgets.Output()
        display(kill_output)
        
        with kill_output:
            print(f"🛑 Killing {len(running_jobs)} running jobs...\n")
            print("=" * 60)

            killed_count = 0
            failed_count = 0
            
            for i, job in enumerate(running_jobs):
                print(f"\n[{i + 1}/{len(running_jobs)}] Killing: {job.name}")
                print(f"  📍 Job ID: {job.job_id}")
                
                try:
                    if self.state.service.cancel_job(job.job_id):
                        killed_count += 1
                        job.cancelled = True
                        print(f"  ✅ Successfully killed")
                    else:
                        failed_count += 1
                        print(f"  ❌ Failed to kill - job may have already finished")
                except Exception as e:
                    failed_count += 1
                    print(f"  ❌ Error: {str(e)}")
                
                print("-" * 60)
            
            # Summary
            print(f"\n{'=' * 60}")
            print(f"📊 KILL SUMMARY:")
            print(f"  ✅ Killed: {killed_count}/{len(running_jobs)}")
            print(f"  ❌ Failed: {failed_count}/{len(running_jobs)}")
            print(f"{'=' * 60}\n")
        
        # Re-enable button
        self.kill_all_btn.disabled = False
        self.kill_all_btn.description = "Kill All Jobs"
        self.kill_all_btn.button_style = "danger"
        
        self.update()

    def render(self) -> None:
        """Render the status widget with enhanced visuals."""
        # Count jobs
        total_jobs = len(self.state.training_jobs)
        preloaded_jobs = sum(
            1
            for job in self.state.training_jobs
            if hasattr(job, "notes") and "Pre-loaded" in str(job.notes)
        )
        session_jobs = total_jobs - preloaded_jobs
        
        # Update launch button state
        self.launch_all_btn.disabled = len(self.state.training_job_configs) == 0

        if total_jobs == 0 and not self.show_all_sky_jobs:
            # Empty state - minimal display
            self._render_empty_state()
            return

        # Full display with header
        display(self.styles)

        # Create header with job counts
        badges = []
        if total_jobs > 0:
            badges.append(f'<span class="job-count-badge">{total_jobs} total</span>')
        if preloaded_jobs > 0:
            badges.append(
                f'<span class="job-count-badge" style="background: #6c757d;">{preloaded_jobs} from experiment</span>'
            )
        if session_jobs > 0:
            badges.append(
                f'<span class="job-count-badge" style="background: #28a745;">{session_jobs} this session</span>'
            )

        badges_html = (
            " ".join(badges)
            if badges
            else '<span class="job-count-badge">0 active</span>'
        )

        header_html = f"""
        <div class="status-header">
            <h2>🚀 Training Jobs Dashboard {badges_html}</h2>
            <p>Monitor your training runs and Sky cluster usage</p>
        </div>
        """
        display(widgets.HTML(header_html))

        # Container for content
        container = widgets.VBox(
            layout=widgets.Layout(padding="0", margin="0", width="100%")
        )

        # Controls
        controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.toggle_btn,
                        self.refresh_btn,
                        widgets.HTML(
                            f'<span style="margin-left: 20px; color: #6c757d;">Last updated: {datetime.now().strftime("%H:%M:%S")}</span>'
                        ),
                    ]
                ),
                widgets.HBox([self.launch_all_btn, self.cancel_btn, self.kill_all_btn]),
            ],
            layout=widgets.Layout(padding="10px"),
        )

        # Content area
        content = widgets.Output()
        with content:
            if self.show_all_sky_jobs:
                self._render_all_sky_jobs()
            else:
                self._render_session_jobs()

        # Assemble container
        container.children = [
            widgets.HTML('<div class="status-container">'),
            controls,
            content,
            widgets.HTML("</div>"),
        ]

        display(container)

    def _render_empty_state(self) -> None:
        """Render minimal empty state with launch controls if configs exist."""
        display(self.styles)
        
        # Check if we have configs to launch
        has_configs = len(self.state.training_job_configs) > 0
        
        # Empty state message
        empty_html = """
        <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 20px; margin-bottom: 20px; background: #f8f9fa;">
            <div style="text-align: center;">
                <h3 style="margin: 0; color: #6c757d; font-size: 18px;">📊 No Active Training Jobs</h3>
                <p style="margin: 10px 0 0 0; color: #868e96; font-size: 14px;">Launch a training run to see status here</p>
            </div>
        </div>
        """
        display(widgets.HTML(empty_html))
        
        # Show launch controls if we have configs
        if has_configs:
            # Create config summary
            config_summary_html = self._get_config_summary_html()
            
            controls = widgets.VBox([
                widgets.HTML(f'<div style="margin-top: 10px; text-align: center;"><p style="color: #495057; margin-bottom: 10px;">You have {len(self.state.training_job_configs)} config(s) ready to launch:</p></div>'),
                widgets.HTML(config_summary_html),
                widgets.HBox([self.launch_all_btn], layout=widgets.Layout(justify_content='center', margin='10px 0'))
            ])
            display(controls)

    def _render_job_with_checkbox(self, job: TrainingJob) -> widgets.HBox:
        """Render a job with a checkbox for selection."""
        # Create checkbox for this job
        checkbox = widgets.Checkbox(
            value=job.job_id in self.selected_jobs,
            description="",
            layout=widgets.Layout(width="30px"),
        )
        checkbox.observe(
            lambda change: self._on_job_selection_change(change, job.job_id), "value"
        )
        self.job_checkboxes[job.job_id] = checkbox

        # Get job status display
        job_display = job_status([job])

        # Combine checkbox and job display
        return widgets.HBox(
            [checkbox, job_display], layout=widgets.Layout(align_items="center")
        )

    def _render_session_jobs(self) -> None:
        """Render current session jobs with clear distinction."""
        # Clear checkbox tracking
        self.job_checkboxes = {}
        
        # Show configs if any
        if self.state.training_job_configs:
            config_summary_html = self._get_config_summary_html()
            display(widgets.HTML(config_summary_html))

        if self.state.training_jobs:
            # Group jobs by type
            preloaded = [
                j
                for j in self.state.training_jobs
                if hasattr(j, "notes") and "Pre-loaded" in str(j.notes)
            ]
            session = [
                j
                for j in self.state.training_jobs
                if not (hasattr(j, "notes") and "Pre-loaded" in str(j.notes))
            ]

            # Display preloaded jobs if any
            if preloaded:
                display(
                    widgets.HTML(
                        '<h4 style="margin: 15px 0 10px 0; color: #495057;">📦 Experiment Jobs (Pre-loaded)</h4>'
                    )
                )
                for job in preloaded:
                    display(self._render_job_with_checkbox(job))

            # Display session jobs if any
            if session:
                if preloaded:  # Add spacing if we showed preloaded jobs
                    display(widgets.HTML('<div style="margin-top: 20px;"></div>'))
                display(
                    widgets.HTML(
                        '<h4 style="margin: 15px 0 10px 0; color: #495057;">🌟 Current Session Jobs</h4>'
                    )
                )
                for job in session:
                    display(self._render_job_with_checkbox(job))

            # If no session jobs but have preloaded
            if preloaded and not session:
                display(
                    widgets.HTML(
                        '<div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; text-align: center;">'
                        + '<p style="margin: 0; color: #6c757d;">No new jobs launched in this session. Launch training runs to add more.</p>'
                        + "</div>"
                    )
                )
        else:
            display(
                widgets.HTML(
                    '<div class="status-empty">'
                    + "<h3>No jobs tracked</h3>"
                    + "<p>Launch training runs or toggle to see all Sky jobs</p>"
                    + "</div>"
                )
            )

    def _render_all_sky_jobs(self) -> None:
        """Render all Sky jobs with enhanced display."""
        sky_df = self.state.service.get_sky_jobs_data(include_all=True)

        if sky_df.empty:
            display(
                widgets.HTML(
                    '<div class="status-empty"><h3>No Sky jobs found</h3></div>'
                )
            )
            return

        # Get job IDs by type
        all_tracked_ids = [j.job_id for j in self.state.training_jobs if j.job_id]
        preloaded_ids = [
            j.job_id
            for j in self.state.training_jobs
            if j.job_id and hasattr(j, "notes") and "Pre-loaded" in str(j.notes)
        ]
        session_ids = [
            j.job_id
            for j in self.state.training_jobs
            if j.job_id and not (hasattr(j, "notes") and "Pre-loaded" in str(j.notes))
        ]

        # Create enhanced table
        html_parts = ['<div style="overflow-x: auto;">']
        html_parts.append("""
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
        <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
        """)

        # Headers
        headers = ["Job ID", "Name", "Status", "Resources", "Duration", "Cost"]
        for header in headers:
            html_parts.append(
                f'<th style="padding: 12px; text-align: left; font-weight: 600;">{header}</th>'
            )
        html_parts.append("</tr></thead><tbody>")

        # Rows
        for _, row in sky_df.iterrows():
            job_id_str = row.get("ID", "-")
            is_tracked = job_id_str in all_tracked_ids
            is_preloaded = job_id_str in preloaded_ids
            is_session = job_id_str in session_ids

            # Different styles for different job types
            if is_session:
                row_style = "background-color: #d4edda;"  # Green for session
            elif is_preloaded:
                row_style = "background-color: #e2e3e5;"  # Gray for preloaded
            else:
                row_style = ""

            html_parts.append(
                f'<tr style="border-bottom: 1px solid #e9ecef; {row_style}">'
            )

            # Job ID with indicator
            job_id = job_id_str
            if is_session:
                job_id = (
                    f'<span style="color: #155724; font-weight: 600;">✓ {job_id}</span>'
                )
            elif is_preloaded:
                job_id = f'<span style="color: #383d41; font-weight: 600;">📦 {job_id}</span>'
            html_parts.append(f'<td style="padding: 10px;">{job_id}</td>')

            # Name
            name = row.get("NAME", "-")
            html_parts.append(
                f'<td style="padding: 10px; font-weight: 500;">{name}</td>'
            )

            # Status with color
            status = row.get("STATUS", "-")
            status_color = {
                "RUNNING": "#28a745",
                "SUCCEEDED": "#007bff",
                "FAILED": "#dc3545",
                "CANCELLED": "#6c757d",
            }.get(status, "#000")
            html_parts.append(
                f'<td style="padding: 10px; color: {status_color}; font-weight: 600;">{status}</td>'
            )

            # Resources
            resources = row.get("RESOURCES", "-")
            html_parts.append(f'<td style="padding: 10px;">{resources}</td>')

            # Duration
            duration = row.get("JOB DURATION", "-")
            html_parts.append(f'<td style="padding: 10px;">{duration}</td>')

            # Cost
            cost = row.get("EST_COST", "-")
            cost_style = ""
            if cost != "-":
                try:
                    cost_val = float(cost.replace("$", ""))
                    if cost_val > 10:
                        cost_style = "color: #dc3545; font-weight: 600;"
                    elif cost_val > 5:
                        cost_style = "color: #ffc107;"
                except:
                    pass
            html_parts.append(f'<td style="padding: 10px; {cost_style}">{cost}</td>')

            html_parts.append("</tr>")

        html_parts.append("</tbody></table></div>")

        # Legend
        html_parts.append("""
        <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
            <span style="margin-right: 20px;"><span style="color: #155724; font-weight: 600;">✓</span> Current Session</span>
            <span style="margin-right: 20px;"><span style="color: #383d41; font-weight: 600;">📦</span> Experiment Pre-loaded</span>
            <span>Other Sky Jobs</span>
        </div>
        """)

        # Summary
        if "EST_COST_USD" in sky_df.columns:
            total_cost = sky_df["EST_COST_USD"].sum()
            tracked_cost = (
                sky_df[sky_df["ID"].isin(all_tracked_ids)]["EST_COST_USD"].sum()
                if all_tracked_ids
                else 0
            )
            session_cost = (
                sky_df[sky_df["ID"].isin(session_ids)]["EST_COST_USD"].sum()
                if session_ids
                else 0
            )

            html_parts.append(f"""
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <div style="font-size: 24px; font-weight: 600; color: #495057;">${total_cost:.2f}</div>
                        <div style="font-size: 14px; color: #6c757d;">Total Cost</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: 600; color: #1976d2;">${tracked_cost:.2f}</div>
                        <div style="font-size: 14px; color: #6c757d;">Tracked Cost</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: 600; color: #28a745;">${session_cost:.2f}</div>
                        <div style="font-size: 14px; color: #6c757d;">Session Cost</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: 600; color: #495057;">{len(all_tracked_ids)}</div>
                        <div style="font-size: 14px; color: #6c757d;">Tracked Jobs</div>
                    </div>
                </div>
            </div>
            """)

        display(widgets.HTML("".join(html_parts)))

    def _get_config_summary_html(self) -> str:
        """Get a compact HTML summary of training job configs."""
        if not self.state.training_job_configs:
            return ""
        
        html_parts = ['<div style="margin: 10px 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">']
        html_parts.append('<h4 style="margin: 0 0 10px 0; color: #495057; font-size: 16px;">📋 Configs Ready to Launch:</h4>')
        html_parts.append('<div style="font-family: monospace; font-size: 13px;">')
        
        for i, config in enumerate(self.state.training_job_configs):
            # Create a short description
            tags = f" [{', '.join(config.wandb_tags)}]" if config.wandb_tags else ""
            spot = " (spot)" if config.spot else ""
            
            html_parts.append(
                f'<div style="margin: 5px 0; padding: 8px; background: white; border: 1px solid #e1e4e8; border-radius: 4px;">'
                f'<span style="color: #0969da; font-weight: 600;">[{i}]</span> '
                f'<span style="color: #24292e;">{config.curriculum}</span> '
                f'<span style="color: #6f42c1;">({config.gpus}×GPU, {config.nodes} node{"s" if config.nodes > 1 else ""}){spot}</span>'
                f'<span style="color: #586069;">{tags}</span>'
                f'</div>'
            )
        
        html_parts.append('</div>')
        html_parts.append('</div>')
        
        return ''.join(html_parts)


def job_status(
    training_jobs: List[TrainingJob],
    entity: str = "metta-research",
    project: str = "metta",
    show_costs: bool = True,
) -> Any:
    """Get status for a list of TrainingJob objects.

    Args:
        training_jobs: List of TrainingJob objects
        entity: Wandb entity
        project: Wandb project
        show_costs: Include cost estimates

    Returns:
        HTML widget with enhanced job status display
    """
    if not training_jobs:
        try:
            import ipywidgets as widgets

            return widgets.HTML("<i>No jobs to display</i>")
        except:
            return pd.DataFrame()

    # Extract run names and job IDs
    wandb_run_names = [job.name for job in training_jobs]
    skypilot_job_ids = [job.job_id for job in training_jobs if job.job_id]

    # Get status from service
    from experiments.skypilot_service import get_skypilot_service

    service = get_skypilot_service()
    df = service.get_training_status(
        wandb_run_names=wandb_run_names,
        skypilot_job_ids=skypilot_job_ids if skypilot_job_ids else None,
        entity=entity,
        project=project,
        include_costs=show_costs,
    )

    # Return enhanced widget
    return create_enhanced_status_widget(df)


def create_enhanced_status_widget(df: pd.DataFrame) -> Any:
    """Create an enhanced HTML widget for status display with cost info.

    Args:
        df: DataFrame with status information

    Returns:
        HTML widget with styled table
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        return df

    if df.empty:
        return widgets.HTML("<i>No data to display</i>")

    # Start building HTML
    html_parts = ["<style>"]
    html_parts.append("""
    .enhanced-status-table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 13px;
        margin-bottom: 10px;
    }
    .enhanced-status-table th {
        background-color: #f0f0f0;
        padding: 10px;
        text-align: left;
        border-bottom: 2px solid #ddd;
        font-weight: 600;
    }
    .enhanced-status-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #eee;
    }
    .enhanced-status-table tr:hover {
        background-color: #f9f9f9;
    }
    .status-running { color: #28a745; font-weight: bold; }
    .status-finished { color: #007bff; }
    .status-failed { color: #dc3545; font-weight: bold; }
    .cost-high { color: #dc3545; font-weight: bold; }
    .cost-medium { color: #ffc107; }
    .cost-low { color: #28a745; }
    .wandb-link { color: #007bff; text-decoration: none; }
    .wandb-link:hover { text-decoration: underline; }
    """)
    html_parts.append("</style>")

    # Build table
    html_parts.append('<table class="enhanced-status-table">')

    # Header - customize column display names
    column_names = {
        "run_name": "Run Name",
        "state": "W&B State",
        "sky_status": "Sky Status",
        "sky_duration": "Duration",
        "cost": "Est. Cost",
        "created": "Created",
        "url": "Link",
        "_step": "Steps",
        "sky_job_id": "Sky Job ID",
    }

    html_parts.append("<thead><tr>")
    for col in df.columns:
        display_name = column_names.get(col, col.replace("_", " ").title())
        html_parts.append(f"<th>{display_name}</th>")
    html_parts.append("</tr></thead>")

    # Body
    html_parts.append("<tbody>")
    for _, row in df.iterrows():
        html_parts.append("<tr>")
        for col in df.columns:
            value = row[col]
            cell_class = ""

            # Apply styling based on column and value
            if col == "state" and pd.notna(value):
                if "running" in str(value).lower():
                    cell_class = "status-running"
                elif "finished" in str(value).lower():
                    cell_class = "status-finished"
                elif "failed" in str(value).lower():
                    cell_class = "status-failed"

            elif col == "sky_status" and pd.notna(value):
                if "RUNNING" in str(value):
                    cell_class = "status-running"
                elif "SUCCEEDED" in str(value):
                    cell_class = "status-finished"
                elif "FAILED" in str(value):
                    cell_class = "status-failed"

            elif col == "cost" and pd.notna(value) and value != "-":
                # Parse cost value
                try:
                    cost_val = float(str(value).replace("$", ""))
                    if cost_val > 10:
                        cell_class = "cost-high"
                    elif cost_val > 5:
                        cell_class = "cost-medium"
                    else:
                        cell_class = "cost-low"
                except:
                    pass

            elif col == "url" and pd.notna(value) and str(value).startswith("http"):
                value = (
                    f'<a href="{value}" target="_blank" class="wandb-link">W&B →</a>'
                )

            # Handle None/NaN values
            if pd.isna(value) or value is None:
                value = "-"

            html_parts.append(f'<td class="{cell_class}">{value}</td>')
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    # Add cost summary if available
    if "cost" in df.columns:
        total_cost = 0
        for _, row in df.iterrows():
            if pd.notna(row["cost"]) and row["cost"] != "-":
                try:
                    cost_val = float(str(row["cost"]).replace("$", ""))
                    total_cost += cost_val
                except:
                    pass

        if total_cost > 0:
            html_parts.append('<div style="margin-top: 10px; font-size: 14px;">')
            html_parts.append(f"<b>Total Estimated Cost:</b> ${total_cost:.2f}")
            html_parts.append("</div>")

    return widgets.HTML("".join(html_parts))


def generate_cells(state_widget_in_header: bool = True) -> List[Dict[str, Any]]:
    """Generate notebook cells for monitoring section.

    Args:
        state_widget_in_header: If True, SkypilotWidget is assumed to be in the header

    Returns:
        List of cell definitions
    """
    cells = []

    # Always add section header for simplified notebook
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 🚀 Monitor & Control Training Jobs",
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """status_widget.display()""",
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """status_widget.update()""",
        }
    )

    return cells
