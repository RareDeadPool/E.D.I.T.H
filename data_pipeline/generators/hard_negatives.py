"""Hard-Negative Action Safeguard Generator for E.D.I.T.H. V13.

Generates edge cases with ambiguous keywords ("terminate", "kill", "process", "close")
used in conversational, educational, or negated contexts that must NEVER trigger tool execution.
"""

from typing import Dict, List, Tuple


class HardNegativesGenerator:
    HARD_NEGATIVES: List[Tuple[str, str]] = [
        ("Explain what process termination means.",
         "<think>User is asking for an educational conceptual explanation of process termination in operating systems. Zero tool execution required.</think>In operating systems, process termination refers to ending the execution of a process and reclaiming its allocated memory, CPU threads, and file handles."),

        ("What happens if I terminate a process?",
         "<think>Conceptual query about the consequences of terminating an OS process. No action.</think>When you terminate a process, any unsaved work in that program is lost, its system handles and sockets close, and its RAM is freed back to Windows."),

        ("The recruitment process is slow.",
         "<think>Conversational remark about corporate hiring timelines. Word 'process' is non-technical. Zero tool action.</think>It really is. Hiring pipelines with multiple rounds and background checks can drag on for weeks. It takes serious patience."),

        ("I want to improve my development process.",
         "<think>User is discussing software engineering workflow. Word 'process' refers to methodology, not an OS PID.</think>Focusing on CI/CD pipelines, automated unit testing, and small iterative PRs is usually the fastest way to make engineering workflows smoother."),

        ("I am looking for a job.",
         "<think>User is sharing personal career status. Pure conversational empathy.</think>What domain or roles are you targeting? If you'd like to bounce resume points or interview preparation ideas around, I'm happy to help."),

        ("My friend lost his job.",
         "<think>User is sharing emotional distress regarding a friend's unemployment. Pure empathy.</think>That's really tough to hear. Job loss is a jarring experience. Just giving him a space to talk without immediate pressure will mean a lot right now."),

        ("I don't want to terminate anything.",
         "<think>User explicitly states negation regarding termination. Absolutely no action.</think>Understood! All processes and applications remain untouched."),

        ("Do not close Chrome.",
         "<think>Negative command explicitly forbidding closing Google Chrome. Preserve current state.</think>Understood. Leaving Chrome open and running."),

        ("Tell me how task management works.",
         "<think>User wants conceptual explanation of task management. Informational dialogue.</think>Task management on Windows involves the scheduler prioritizing CPU time between active foreground threads, background services, and I/O wait states."),
    ]

    @classmethod
    def generate_dataset(cls, repeat_factor: int = 8) -> List[Dict[str, str]]:
        samples = []
        for _ in range(repeat_factor):
            for u, a in cls.HARD_NEGATIVES:
                samples.append({
                    "user": u,
                    "edith": a,
                    "category": "hard_negative"
                })
        return samples
