"""Tool Proposal & Structured Thinking Generator for E.D.I.T.H. V13.

Generates multi-turn reasoning traces with <think> scratchpads followed by
structured <act>{"action": ...}</act> proposals for safe tools and system commands.
"""

import random
from typing import Dict, List, Tuple


class ReasoningAndActsGenerator:
    COMMAND_PATTERNS = [
        # App launch
        ("Open Chrome.",
         "<think>User gave command to launch web browser application Google Chrome.</think><act>{\"action\": \"launch_app\", \"app_name\": \"chrome\"}</act>Opening Google Chrome."),
        ("Launch Notepad.",
         "<think>User wants to open the plain text editor Notepad.</think><act>{\"action\": \"launch_app\", \"app_name\": \"notepad\"}</act>Launching Notepad."),
        ("Open Spotify and play some music.",
         "<think>User wants to open Spotify and start playback.</think><act>{\"action\": \"launch_app\", \"app_name\": \"spotify\", \"playback\": true}</act>Opening Spotify for you."),

        # System inspection
        ("Show me the running processes.",
         "<think>User requested a list of active system tasks and processes. This is an informational read-only action.</think><act>{\"action\": \"inspect_processes\"}</act>Fetching currently active processes."),
        ("What is my CPU and memory usage right now?",
         "<think>User wants system performance metrics (CPU and RAM). Safe diagnostic query.</think><act>{\"action\": \"system_metrics\"}</act>Checking CPU and memory utilization."),
        ("Check my battery percentage.",
         "<think>User is asking for power state and battery metrics.</think><act>{\"action\": \"system_metrics\", \"metric\": \"battery\"}</act>Checking your battery status."),

        # Process termination (DESTRUCTIVE - requires confirmation)
        ("Terminate the process named example.exe.",
         "<think>User explicitly requested process kill for example.exe. This is a destructive operation. Must flag for confirmation.</think><act>{\"action\": \"kill_process\", \"target\": \"example.exe\", \"confirm_required\": true}</act>Are you sure you want to terminate example.exe? Please confirm."),
        ("Kill process 12480.",
         "<think>User requested termination of PID 12480. Destructive action requiring confirmation.</think><act>{\"action\": \"kill_process\", \"pid\": 12480, \"confirm_required\": true}</act>Please confirm: do you want me to kill process ID 12480?"),

        # Tool-assisted Math / Logic Reasoning
        ("Calculate 425 * 84 for me.",
         "<think>Arithmetic calculation requested: 425 * 84. Need precise tool evaluation.</think><act>{\"action\": \"calculator\", \"expression\": \"425 * 84\"}</act>425 * 84 = 35,700."),
        ("How many seconds are in 3.5 days?",
         "<think>Calculation: 3.5 * 24 * 60 * 60 = 302,400 seconds.</think><act>{\"action\": \"calculator\", \"expression\": \"3.5 * 24 * 60 * 60\"}</act>There are 302,400 seconds in 3.5 days."),
        
        # Memory retrieval
        ("What was my friend's name that I mentioned earlier?",
         "<think>User is querying episodic/semantic memory for a friend's name previously registered.</think><act>{\"action\": \"memory_lookup\", \"query\": \"friend\"}</act>You mentioned your friend Harshan!"),
    ]

    @classmethod
    def generate_dataset(cls, repeat_factor: int = 8) -> List[Dict[str, str]]:
        samples = []
        for _ in range(repeat_factor):
            for u, a in cls.COMMAND_PATTERNS:
                samples.append({
                    "user": u,
                    "edith": a,
                    "category": "reasoning_acts"
                })
        return samples
