"""Operational Resource & Software Condition Monitor for E.D.I.T.H. V5.

Observes software and host conditions (CPU, memory, latency, retry counts)
without anthropomorphic body simulation, and translates operational condition
into computational reasoning budgets.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional

from .cognitive_state import CognitiveState, OperationalResourceState


class OperationalResourceMonitor:
    """Monitors local computing conditions and budgets computational depth."""

    def __init__(self):
        self.last_sample_time = 0.0

    def sample_system_metrics(self) -> Dict[str, float]:
        """Query host CPU, RAM, and Disk metrics safely."""
        cpu = 0.0
        mem = 0.0
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
        except Exception:
            pass

        return {
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(mem, 1)
        }

    def update_operational_condition(
        self,
        state: CognitiveState,
        last_latency_ms: float = 0.0,
        action_failed: bool = False
    ) -> OperationalResourceState:
        """Update operational resource state and compute available compute budget."""
        metrics = self.sample_system_metrics()

        oper = state.operational
        oper.cpu_percent = metrics["cpu_percent"]
        oper.memory_percent = metrics["memory_percent"]
        oper.recent_latency_ms = round(last_latency_ms, 2)

        if action_failed:
            oper.retry_count += 1
            oper.consecutive_failures += 1
        else:
            oper.consecutive_failures = 0
            if oper.retry_count > 0:
                oper.retry_count -= 1

        # Calculate available compute budget [0.0 (throttled) to 1.0 (free)]
        # High CPU/RAM pressure or high latency lowers compute budget to preserve host responsiveness
        cpu_load_factor = min(1.0, oper.cpu_percent / 100.0)
        mem_load_factor = min(1.0, oper.memory_percent / 100.0)
        latency_factor = min(1.0, oper.recent_latency_ms / 2000.0)

        pressure = 0.4 * cpu_load_factor + 0.4 * mem_load_factor + 0.2 * latency_factor
        oper.available_compute_budget = round(max(0.1, 1.0 - pressure), 3)

        return oper
