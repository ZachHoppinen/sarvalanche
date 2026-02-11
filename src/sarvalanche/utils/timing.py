"""
Simple timing utilities for the SARvalanche pipeline.

This module provides clean timing without cluttering code with context managers.
"""

import time
import logging
from functools import wraps

log = logging.getLogger(__name__)


class PipelineTimer:
    """
    Simple timer to track pipeline steps.

    Usage:
        timer = PipelineTimer()

        timer.step('validation')
        # do validation work

        timer.step('data_loading')
        # do data loading

        timer.summary()
    """

    def __init__(self):
        self.steps = {}
        self.current_step = None
        self.current_start = None
        self.total_start = time.time()
        log.info("⏱️  Pipeline timer started")

    def step(self, step_name):
        """
        End the previous step (if any) and start timing a new step.

        This is the main method you'll use - just call it at the start
        of each major step in your pipeline.
        """
        # Stop and record the previous step
        if self.current_step is not None:
            duration = time.time() - self.current_start
            self.steps[self.current_step] = duration
            log.info(f"✅ Completed: {self.current_step} ({self._format_time(duration)})")

        # Start the new step
        self.current_step = step_name
        self.current_start = time.time()
        log.info(f"▶️  Starting: {step_name}")

    def summary(self):
        """Print a summary of all timed steps."""
        # Record the final step if one is active
        if self.current_step is not None:
            duration = time.time() - self.current_start
            self.steps[self.current_step] = duration
            log.info(f"✅ Completed: {self.current_step} ({self._format_time(duration)})")
            self.current_step = None

        if not self.steps:
            log.info("No timing data recorded")
            return

        log.info("\n" + "="*70)
        log.info("⏱️  PIPELINE TIMING SUMMARY")
        log.info("="*70)

        # Calculate total time
        total_time = time.time() - self.total_start

        # Sort by duration (longest first)
        sorted_steps = sorted(self.steps.items(), key=lambda x: x[1], reverse=True)

        for step_name, duration in sorted_steps:
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            log.info(f"  {step_name:45s} {self._format_time(duration):>12s} ({percentage:5.1f}%)")

        log.info("-"*70)
        log.info(f"  {'TOTAL TIME':45s} {self._format_time(total_time):>12s}")
        log.info("="*70 + "\n")

        return self.steps

    @staticmethod
    def _format_time(seconds):
        """Format seconds into human-readable time."""
        if seconds < 1:
            return f"{seconds*1000:.0f} ms"
        elif seconds < 60:
            return f"{seconds:.1f} sec"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"