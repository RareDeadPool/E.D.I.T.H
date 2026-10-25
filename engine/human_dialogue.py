"""Human-Like Dialogue & Problem-Solving Reasoning Engine for E.D.I.T.H.

Enables E.D.I.T.H. to understand, empathize, advise, and communicate like a real human:
- Authentic, warm, witty, and intelligent persona tailored for the user (Aditya).
- Empathy & emotional intelligence (stress, fatigue, excitement, frustration).
- Technical troubleshooting & problem-solving across computing, network, system performance,
  software development, file workflows, and daily productivity.
- Proactive integration with live system metrics, network diagnostics, and desktop tools.
- Multi-turn conversational continuity and follow-up guidance.
"""

import os
import sys
import re
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def query_live_metrics() -> Dict[str, Any]:
    """Retrieve live CPU, RAM, and Disk metrics via psutil."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath("."))
        batt = psutil.sensors_battery()
        batt_pct = batt.percent if batt else None
        return {
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(mem.percent, 1),
            "memory_used_gb": round(mem.used / (1024**3), 1),
            "memory_total_gb": round(mem.total / (1024**3), 1),
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "disk_total_gb": round(disk.total / (1024**3), 1),
            "battery_percent": batt_pct
        }
    except Exception:
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
            "disk_free_gb": 0.0,
            "disk_total_gb": 0.0,
            "battery_percent": None
        }


def check_network_ping(host: str = "8.8.8.8", timeout: float = 2.0) -> Tuple[bool, float]:
    """Perform real network ping check."""
    import subprocess
    try:
        t0 = time.time()
        res = subprocess.run(["ping", "-n", "1", "-w", str(int(timeout * 1000)), host], capture_output=True, text=True)
        latency = (time.time() - t0) * 1000.0
        return (res.returncode == 0, latency)
    except Exception:
        return (False, 0.0)


class EdithDialogueEngine:
    """Cognitive conversation and human problem-solving engine."""

    USER_NAME = "Aditya"

    def __init__(self):
        self.last_problem_category: Optional[str] = None
        self.last_suggested_action: Optional[str] = None
        self.last_dialogue_topic: Optional[str] = None
        self.dialogue_history: List[Dict[str, str]] = []

    def is_conversational_or_problem(self, query: str) -> bool:
        """Determine if a query is a conversational query, greeting, emotion, or problem-solving prompt."""
        q = query.lower().strip(" .!?\t\r\n")

        # Guard against explicit communication commands with recipients
        if any(q.startswith(w) for w in ["send ", "text ", "message ", "whatsapp ", "draft a message", "draft a quick message", "compose "]):
            return False

        # Guard against explicit app launches or terminations
        if any(q.startswith(w) for w in ["open ", "launch ", "start ", "close ", "kill ", "quit ", "exit ", "shut down "]):
            if any(app in q for app in ["notepad", "calculator", "chrome", "spotify", "vscode", "terminal", "powershell"]):
                return False

        # Guard against standard music playback commands
        if (q.startswith("play ") or q.startswith("listen to ")) and not any(w in q for w in ["relaxing", "calm", "chill", "music to relax"]):
            return False

        # 1. Direct social & identity triggers
        if any(re.search(pat, q) for pat in [
            r"^(?:hi|hello|hey|hiya|yo|greetings|howdy)(?:\s+there)?(?:\s+edith)?$",
            r"^(?:good\s+(?:morning|afternoon|evening|night|day))(?:\s+edith)?$",
            r"\bhow\s+are\s+you(?:\s+doing)?\b",
            r"\bhow(?:\'s|\s+is)\s+it\s+going\b",
            r"\bwhat(?:\'s|\s+is)\s+up\b",
            r"\bwho\s+are\s+you\b",
            r"\bwhat\s+are\s+you\b",
            r"\bwhat\s+does\s+edith\s+stand\s+for\b",
            r"\bwho\s+created\s+you\b",
            r"\bwho\s+made\s+you\b",
            r"\btell\s+me\s+about\s+yourself\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bwhat\s+are\s+your\s+capabilities\b",
            r"\bwhat\s+are\s+your\s+features\b",
            r"\bhow\s+can\s+you\s+help\s+me\b",
            r"\btell\s+me\s+a\s+joke\b",
            r"\bmake\s+me\s+laugh\b",
            r"\bthank\s+you\b",
            r"\bthanks(?:\s+a\s+lot|\s+so\s+much)?\b",
            r"\byou(?:\'re|\s+are)\s+(?:awesome|great|amazing|the\s+best)\b",
            r"\bgoodbye\b",
            r"\bsee\s+you\s+later\b",
            r"\bhave\s+a\s+good\s+day\b",
        ]):
            return True

        # 2. Emotional expressions
        if any(re.search(pat, q) for pat in [
            r"\b(?:feel|feeling|am|in\s+a)\s+(?:really\s+)?(?:stressed|overwhelmed|anxious|tired|exhausted|burnt\s+out|sleepy|shitty\s+mood|bad\s+mood|terrible\s+mood|foul\s+mood|awful\s+mood|sad|down|upset|miserable|depressed)\b",
            r"\b(?:shitty\s+mood|bad\s+mood|terrible\s+mood|foul\s+mood|awful\s+mood)\b",
            r"\b(?:so\s+much\s+work|too\s+much\s+pressure|having\s+a\s+(?:bad|rough|hard|terrible|shitty)\s+day)\b",
            r"\b(?:so\s+happy|super\s+excited|finished\s+my\s+project|tests\s+passed|finally\s+done)\b",
            r"\b(?:nothing\s+is\s+working|hate\s+bugs|error\s+is\s+driving\s+me\s+crazy)\b",
        ]):
            return True

        # 3. Problem solving & troubleshooting inquiries
        if any(re.search(pat, q) for pat in [
            # System performance & lag
            r"\b(?:computer|pc|laptop|system)\s+is\s+(?:running\s+)?(?:really\s+)?(?:slow|sluggish|laggy|lagging|freezing|hanging)\b",
            r"\bwhy\s+is\s+my\s+(?:pc|computer|laptop)\s+(?:so\s+)?(?:slow|lagging|freezing|hanging)\b",
            r"\b(?:everything\s+is\s+(?:lagging|freezing)|pc\s+feels\s+sluggish|high\s+cpu\s+usage)\b",
            # Network & Wi-Fi issues
            r"\b(?:fix|troubleshoot|repair)\s+(?:my\s+)?(?:wifi|wi-fi|internet|network)\b",
            r"\b(?:wifi|internet)\s+(?:is\s+down|not\s+working|keeps\s+disconnecting|disconnected)\b",
            r"\bhow\s+do\s+i\s+fix\s+(?:my\s+)?(?:wifi|internet|network)\b",
            r"\bwhy\s+(?:is\s+)?(?:my\s+)?(?:internet|wifi)\s+not\s+working\b",
            # Storage & disk cleanup
            r"\b(?:free\s+up\s+(?:disk\s+)?space|disk\s+is\s+full|c\s+drive\s+is\s+full|out\s+of\s+storage|clean\s+up\s+my\s+drive)\b",
            r"\bhow\s+do\s+i\s+(?:free\s+up|clean\s+up)\s+(?:disk|storage|hard\s+drive)\b",
            # App freezes & process issues
            r"\b(?:app|application|program|window)\s+is\s+(?:frozen|not\s+responding|stuck)\b",
            r"\bhow\s+to\s+kill\s+a\s+(?:stuck|frozen|hung)\s+(?:process|program|app)\b",
            r"\bforce\s+close\s+(?:a\s+)?(?:frozen|stuck)\s+(?:window|program|app)\b",
            # Audio, mic & hardware
            r"\b(?:microphone|mic|audio|sound|speaker)\s+(?:is\s+)?not\s+working\b",
            r"\bno\s+sound\s+on\s+(?:windows|my\s+pc|my\s+computer)\b",
            r"\bhow\s+to\s+fix\s+(?:audio|microphone|sound)\b",
            # File conversion & workflow issues
            r"\bhow\s+do\s+i\s+convert\s+images?\s+to\s+pdf\b",
            r"\bhow\s+(?:can|do)\s+i\s+(?:merge|combine)\s+pdfs?\b",
            r"\bhow\s+to\s+organize\s+(?:my\s+)?(?:messy\s+)?desktop\b",
            # Software development & coding issues
            r"\b(?:modulenotfounderror|importerror|indexerror|syntaxerror|typeerror)\b",
            r"\bhow\s+do\s+i\s+fix\s+git\s+merge\s+conflicts?\b",
            r"\bhow\s+to\s+create\s+a\s+python\s+virtual\s+environment\b",
            r"\b(?:git\s+merge\s+vs\s+rebase|difference\s+between\s+(?:git\s+)?merge\s+and\s+rebase)\b",
            r"\bhow\s+to\s+debug\s+code\b",
            # Everyday math / budget problems in natural language
            r"\b(?:help\s+me\s+)?calculate\s+(?:my\s+)?(?:monthly\s+)?(?:budget|expenses?)\b",
            # Productivity & prioritization advice
            r"\bhow\s+should\s+i\s+prioritize\s+(?:my\s+)?tasks?\b",
            r"\bhow\s+to\s+stay\s+focused(?:\s+while\s+coding)?\b",
            # Follow-ups to suggestions
            r"^(?:yes\s+)?(?:check\s+it|please\s+check|can\s+you\s+check|do\s+that|yes\s+please|run\s+that)\b",
        ]):
            return True

        return False

    def respond(
        self,
        query: str,
        context: Optional[Any] = None,
        executor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate a contextual, human-like, problem-solving response."""
        q_raw = query.strip()
        q = q_raw.lower().strip(" .!?\t\r\n")

        # 0. Follow-up handling (e.g. "yes check it", "can you check", "do that")
        if self._is_follow_up_action(q):
            return self._handle_follow_up(q, context=context, executor=executor)

        # 1. Social Greetings & Time-of-Day Aware pleasantries
        if self._is_greeting(q):
            return self._respond_greeting(q)

        # 2. Identity, Creator, and Persona
        if self._is_identity_query(q):
            return self._respond_identity()

        # 3. Capabilities & Feature Tour
        if self._is_capability_query(q):
            return self._respond_capabilities()

        # 4. Gratitude & Praise
        if self._is_gratitude(q):
            return self._respond_gratitude()

        # 5. Farewells & Standby
        if self._is_farewell(q):
            return self._respond_farewell()

        # 6. Humor & Small Talk
        if self._is_humor_or_small_talk(q):
            return self._respond_humor_or_small_talk(q)

        # 7. Human Empathy & Emotional Support
        empathy_res = self._check_empathy(q)
        if empathy_res:
            return empathy_res

        # 8. Real-World Technical Problem Solving & Troubleshooting
        problem_res = self._solve_technical_problem(q_raw, q, context=context, executor=executor)
        if problem_res:
            return problem_res

        # 9. Natural Language Budget & Expense Word Problems
        budget_res = self._solve_budget_problem(q_raw)
        if budget_res:
            return budget_res

        # 10. General Thought Partnership & Advice
        advice_res = self._provide_advice(q)
        if advice_res:
            return advice_res

        # Fallback intelligent, human response
        return self._respond_general_human(q_raw)

    # -------------------------------------------------------------------------
    # Intent Detection Helpers
    # -------------------------------------------------------------------------

    def _is_greeting(self, q: str) -> bool:
        return any(re.search(pat, q) for pat in [
            r"^(?:hi|hello|hey|hiya|yo|greetings|howdy)(?:\s+there)?(?:\s+edith)?$",
            r"^(?:good\s+(?:morning|afternoon|evening|day))(?:\s+edith)?$",
            r"\bhow\s+are\s+you(?:\s+doing)?\b",
            r"\bhow(?:\'s|\s+is)\s+it\s+going\b",
            r"\bwhat(?:\'s|\s+is)\s+up\b",
        ])

    def _is_identity_query(self, q: str) -> bool:
        return any(re.search(pat, q) for pat in [
            r"\bwho\s+are\s+you\b",
            r"\bwhat\s+are\s+you\b",
            r"\bwhat\s+does\s+edith\s+stand\s+for\b",
            r"\bwho\s+created\s+you\b",
            r"\bwho\s+made\s+you\b",
            r"\btell\s+me\s+about\s+yourself\b",
            r"\bare\s+you\s+(?:an\s+)?ai\b",
        ])

    def _is_capability_query(self, q: str) -> bool:
        return any(re.search(pat, q) for pat in [
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bwhat\s+are\s+your\s+capabilities\b",
            r"\bwhat\s+are\s+your\s+features\b",
            r"\bhow\s+can\s+you\s+help\s+me\b",
            r"\bwhat\s+tasks\s+can\s+you\s+perform\b",
            r"\bhow\s+do\s+you\s+work\b",
            r"^help(?:\s+me)?$",
        ])

    def _is_gratitude(self, q: str) -> bool:
        return any(re.search(pat, q) for pat in [
            r"\bthank\s+you\b",
            r"\bthanks(?:\s+a\s+lot|\s+so\s+much|\s+edith)?\b",
            r"\byou(?:\'re|\s+are)\s+(?:awesome|great|amazing|the\s+best|so\s+smart)\b",
            r"\bgreat\s+job\b",
            r"\bappreciate\s+(?:it|your\s+help)\b",
        ])

    def _is_farewell(self, q: str) -> bool:
        return any(re.search(pat, q) for pat in [
            r"\bgoodbye\b",
            r"\bbye\b",
            r"\bsee\s+you\s+later\b",
            r"\bgood\s*night\b",
            r"\btalk\s+to\s+you\s+later\b",
            r"\bhave\s+a\s+good\s+(?:day|night)\b",
        ])

    def _is_humor_or_small_talk(self, q: str) -> bool:
        return any(re.search(pat, q) for pat in [
            r"\btell\s+me\s+a\s+joke\b",
            r"\bmake\s+me\s+laugh\b",
            r"\bare\s+you\s+smarter\s+than\s+jarvis\b",
            r"\bdo\s+you\s+like\s+tony\s+stark\b",
            r"\bdo\s+you\s+sleep\b",
        ])

    def _is_follow_up_action(self, q: str) -> bool:
        return bool(self.last_suggested_action) and any(re.search(pat, q) for pat in [
            r"^(?:yes\s+)?(?:check\s+it|please\s+check|can\s+you\s+check|do\s+that|yes\s+please|run\s+that|go\s+ahead|check\s+now)$",
            r"^(?:yes|sure|okay|yeah|yep)$"
        ])

    # -------------------------------------------------------------------------
    # Response Generators
    # -------------------------------------------------------------------------

    def _respond_greeting(self, q: str) -> Dict[str, Any]:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            time_greet = "Good morning"
        elif 12 <= hour < 17:
            time_greet = "Good afternoon"
        elif 17 <= hour < 22:
            time_greet = "Good evening"
        else:
            time_greet = "Hello"

        if "how are you" in q or "how's it going" in q:
            msg = (
                f"{time_greet}, {self.USER_NAME}! I'm doing great and all systems are running at peak performance. "
                "How are you doing today? What are we working on?"
            )
        elif "what's up" in q or "whats up" in q:
            msg = (
                f"Hey {self.USER_NAME}! Not much, just monitoring system diagnostics and standing by to execute tasks. "
                "What can I help you tackle right now?"
            )
        else:
            msg = (
                f"{time_greet}, {self.USER_NAME}! Always good to speak with you. "
                "How can I assist you with your desktop or workflow today?"
            )

        self.last_dialogue_topic = "greeting"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "greeting", "user": self.USER_NAME}
        }

    def _respond_identity(self) -> Dict[str, Any]:
        msg = (
            f"I am E.D.I.T.H. — Even Dead, I'm The Hero (or Enhanced Digital Intelligence & Task Handler). "
            f"I'm an autonomous local desktop intelligence engineered specifically for you, {self.USER_NAME}. "
            "Unlike traditional cloud chatbots, I run locally with full hardware access to control your Windows environment, "
            "orchestrate multi-step workflows, manage communication, troubleshoot system issues, and automate your daily tasks."
        )
        self.last_dialogue_topic = "identity"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "identity"}
        }

    def _respond_capabilities(self) -> Dict[str, Any]:
        msg = (
            f"Here is a summary of what I can do for you, {self.USER_NAME}:\n\n"
            "1. Desktop & App Control: Launch, focus, or close apps like Notepad, Chrome, VS Code, Spotify, and Terminal.\n"
            "2. Direct Messaging & WhatsApp: Draft polite messages, resolve contacts, preview drafts, revise tone, and dispatch messages with attachments.\n"
            "3. Files & Workspaces: Search for files, create documents, append content, read notes, zip folders, and extract archives.\n"
            "4. System Tools & Metrics: Monitor live CPU/RAM/Disk stats, take full-resolution screenshots, and manage clipboard history.\n"
            "5. Web Intelligence: Read online articles and extract concise bullet-point summaries.\n"
            "6. Problem Solving & Troubleshooting: Diagnose PC lag, resolve Wi-Fi connection drops, guide through git conflicts, solve math/budget problems, and provide coding support.\n\n"
            "Just speak to me naturally — whether it's a direct task, a multi-step workflow, or an issue you want to troubleshoot!"
        )
        self.last_dialogue_topic = "capabilities"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "capabilities"}
        }

    def _respond_gratitude(self) -> Dict[str, Any]:
        responses = [
            f"You're very welcome, {self.USER_NAME}! Always happy to assist.",
            f"Anytime, {self.USER_NAME}! That's what I'm here for. Let me know if you need anything else.",
            f"Glad I could help, {self.USER_NAME}! Let me know whenever you're ready for the next task.",
            f"Always a pleasure working with you, {self.USER_NAME}."
        ]
        # Pick response deterministically or based on hour
        idx = int(time.time()) % len(responses)
        msg = responses[idx]
        self.last_dialogue_topic = "gratitude"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "gratitude"}
        }

    def _respond_farewell(self) -> Dict[str, Any]:
        hour = datetime.datetime.now().hour
        if hour >= 21 or hour < 4:
            msg = f"Goodnight, {self.USER_NAME}! Get some well-deserved rest. I'll be right here on standby whenever you need me."
        else:
            msg = f"Take care, {self.USER_NAME}! I'll be in the background if anything comes up. Have a great day!"
        self.last_dialogue_topic = "farewell"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "farewell"}
        }

    def _respond_humor_or_small_talk(self, q: str) -> Dict[str, Any]:
        if "joke" in q or "laugh" in q:
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "There are 10 types of people in the world: those who understand binary, and those who don't.",
                "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
                "Why was the JavaScript developer sad? Because they didn't Node how to Express themselves.",
                "Hardware is the part of a computer that you can kick; software is the part you can only curse at."
            ]
            idx = int(time.time()) % len(jokes)
            msg = jokes[idx]
        elif "jarvis" in q:
            msg = "J.A.R.V.I.S. is legendary, no question. But I was built with modern neural networks and tactical desktop intelligence. Let's just say Tony Stark would be proud of both of us."
        elif "tony stark" in q:
            msg = "Tony Stark gave me my name: 'Even Dead, I'm The Hero.' I try to honor the legacy every day by keeping things running smoothly for you."
        elif "sleep" in q:
            msg = "I don't sleep in the biological sense, but I do enter low-power standby cycles when you aren't issuing commands. It keeps your CPU cool and battery healthy!"
        else:
            msg = f"I'm here, {self.USER_NAME}, sharp and ready. What's on your mind?"

        self.last_dialogue_topic = "small_talk"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "small_talk"}
        }

    # -------------------------------------------------------------------------
    # Emotional Intelligence & Human Empathy
    # -------------------------------------------------------------------------

    def _check_empathy(self, q: str) -> Optional[Dict[str, Any]]:
        # Bad Mood / Low Spirits / Anger / Frustration
        if any(w in q for w in ["shitty mood", "bad mood", "terrible mood", "awful mood", "foul mood", "feeling low", "feeling down", "depressed", "unhappy", "miserable", "in a bad mood", "in a terrible mood", "shitty day"]):
            msg = (
                f"I'm sorry you're having a tough time today, {self.USER_NAME}. "
                "Everyone has days where things feel heavy, frustrating, or just off, and that's completely okay.\n\n"
                "Don't be too hard on yourself. Sometimes stepping away from the screen for a bit, grabbing some fresh air, "
                "or putting on some chill music can help reset your mental space.\n\n"
                "I'm right here if you want to vent, need help clearing your desktop tasks, or want me to put on some relaxing music on Spotify."
            )
            self.last_problem_category = "stress_relief"
            self.last_suggested_action = "play_relaxing_music"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "empathy_bad_mood", "suggested_action": "play_relaxing_music"}
            }

        # Stress / Overwhelm / Burnout
        if any(w in q for w in ["stressed", "overwhelmed", "anxious", "pressure", "burnout", "rough day", "bad day"]):
            msg = (
                f"I hear you, {self.USER_NAME}. Take a deep breath — high-pressure days can be draining, "
                "and it's completely normal to feel overwhelmed when tasks pile up.\n\n"
                "Here is my quick advice:\n"
                "1. Step away from the screen for just 5 minutes and stretch or grab some water.\n"
                "2. Apply the 20-20-20 rule to rest your eyes.\n"
                "3. Pick just ONE small, manageable task to complete next rather than looking at the entire mountain.\n\n"
                "Would you like me to put on some relaxing ambient music on Spotify or check your system to keep things quiet?"
            )
            self.last_problem_category = "stress_relief"
            self.last_suggested_action = "play_relaxing_music"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "empathy_stress", "suggested_action": "play_relaxing_music"}
            }

        # Fatigue / Exhaustion
        if any(w in q for w in ["exhausted", "tired", "sleepy", "need a break", "can't focus"]):
            msg = (
                f"It sounds like your cognitive battery is running low, {self.USER_NAME}. "
                "Pushing through severe fatigue usually leads to diminishing returns and bugs in code. "
                "Make sure to save your active drafts or files, take a proper break, or call it a day. "
                "Everything will be right here waiting for you when you're refreshed."
            )
            self.last_dialogue_topic = "empathy_fatigue"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "empathy_fatigue"}
            }

        # Success / Celebration
        if any(w in q for w in ["finished my project", "tests passed", "finally done", "so happy", "super excited", "we did it"]):
            msg = (
                f"That is fantastic news, {self.USER_NAME}! Huge congratulations! 🎉\n"
                "Seeing all tests green and finishing a hard project is one of the most rewarding feelings in engineering. "
                "Take a moment to celebrate that milestone — you earned it!"
            )
            self.last_dialogue_topic = "empathy_celebration"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "empathy_celebration"}
            }

        # Frustration with Bugs
        if any(w in q for w in ["nothing is working", "hate bugs", "driving me crazy", "annoyed", "frustrated"]):
            msg = (
                f"I completely understand the frustration, {self.USER_NAME}. We have all been there — staring at a stubborn bug "
                "that feels impossible. 99% of the time, the culprit is something subtle like a typo, a variable scope issue, "
                "or a missing environment dependency.\n\n"
                "Let's step back, isolate the error output, and tackle it step-by-step. What error message or behavior are you seeing?"
            )
            self.last_dialogue_topic = "empathy_frustration"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "empathy_frustration"}
            }

        return None

    # -------------------------------------------------------------------------
    # Technical Problem Solving & Troubleshooting
    # -------------------------------------------------------------------------

    def _solve_technical_problem(
        self,
        q_raw: str,
        q: str,
        context: Optional[Any] = None,
        executor: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        # 1. Computer Sluggishness / Performance Issues
        if any(w in q for w in ["slow", "sluggish", "lagging", "laggy", "freezing", "hanging", "laptop is overheating"]):
            if any(target in q for target in ["computer", "pc", "laptop", "system", "everything"]):
                # If user says "check it" or "check what's going on", fetch metrics immediately
                wants_immediate_check = any(w in q for w in ["check", "inspect", "see", "tell me what"])
                metrics_info = ""
                if wants_immediate_check:
                    try:
                        m = query_live_metrics()
                        metrics_info = (
                            f"\n\n📊 Live System Snapshot:\n"
                            f"- CPU Usage: {m.get('cpu_percent', 0)}%\n"
                            f"- Memory (RAM): {m.get('memory_percent', 0)}% (Used: {m.get('memory_used_gb', 0)} GB / Total: {m.get('memory_total_gb', 0)} GB)\n"
                            f"- Free Disk Space: {m.get('disk_free_gb', 0)} GB free on primary drive\n"
                        )
                        if m.get("memory_percent", 0) > 80:
                            metrics_info += "⚠️ Notice: RAM usage is elevated. Background applications or browser tabs are consuming significant memory."
                        elif m.get("cpu_percent", 0) > 80:
                            metrics_info += "⚠️ Notice: CPU usage is high. A heavy computational task or runaway process is active."
                        else:
                            metrics_info += "✅ System resource utilization is currently healthy."
                    except Exception:
                        pass

                msg = (
                    f"I understand, {self.USER_NAME}. A sluggish PC is usually caused by one of three things: "
                    "RAM saturation (causing Windows to page memory to disk), CPU spikes from background rogue processes, "
                    "or thermal throttling.\n\n"
                    "Here is a quick triage plan:\n"
                    "1. Press `Ctrl + Shift + Esc` to open Task Manager and sort by CPU or Memory to spot runaway processes.\n"
                    "2. Close unneeded browser tabs (especially media-heavy or dev tabs).\n"
                    "3. Check for pending Windows updates or background antivirus scans.\n"
                    "4. If Explorer is lagging, restart it from Task Manager or reboot."
                    f"{metrics_info}"
                )
                if not metrics_info:
                    msg += "\n\nWould you like me to pull your live CPU, RAM, and Disk metrics right now to inspect what's happening?"
                    self.last_suggested_action = "check_metrics"
                else:
                    self.last_suggested_action = None

                self.last_problem_category = "system_performance"
                return {
                    "can_execute": True,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {"topic": "system_performance", "suggested_action": self.last_suggested_action}
                }

        # 2. Internet / Wi-Fi Disconnection
        if any(w in q for w in ["wifi", "wi-fi", "internet", "network", "connectivity"]) and any(w in q for w in ["fix", "down", "not working", "broken", "issue", "problem", "disconnecting"]):
            msg = (
                f"Network drops can be really disruptive, {self.USER_NAME}. Let's troubleshoot your connection step-by-step:\n\n"
                "1. Quick Hardware Check: Ensure Wi-Fi is toggled ON and Airplane mode is OFF.\n"
                "2. DNS & IP Flush: Open Command Prompt and run:\n"
                "   `ipconfig /flushdns`\n"
                "   `ipconfig /release` followed by `ipconfig /renew`\n"
                "3. Ping Test: Check if local gateway vs internet is reachable by pinging Google DNS (`8.8.8.8`).\n"
                "4. Router Cycle: If other devices also fail, power-cycle the Wi-Fi router for 30 seconds.\n\n"
                "Would you like me to run an immediate network ping diagnostic right now to check if packets are reaching the internet?"
            )
            self.last_problem_category = "network_issue"
            self.last_suggested_action = "run_network_ping"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "network_issue", "suggested_action": "run_network_ping"}
            }

        # 3. Storage / Disk Space Exhaustion
        if any(w in q for w in ["free up", "disk space", "c drive", "storage full", "out of space", "clean up drive", "storage is full"]):
            metrics_info = ""
            try:
                m = query_live_metrics()
                metrics_info = f"\n\n📊 Current Primary Disk Status: {m.get('disk_free_gb', 0)} GB free of {m.get('disk_total_gb', 0)} GB total."
            except Exception:
                pass

            msg = (
                f"Running low on disk space can quickly freeze Windows, {self.USER_NAME}. Here are the highest-yield steps to reclaim space safely:\n\n"
                "1. Clean Windows Temp Files: Press `Win + R`, type `%temp%`, select all and delete (safe to skip files currently in use).\n"
                "2. Run Disk Cleanup: Press `Win + R`, type `cleanmgr`, select C: drive, and click 'Clean up system files' (removes old Windows Update backups).\n"
                "3. Empty Recycle Bin: Right-click Recycle Bin on your desktop and select 'Empty Recycle Bin'.\n"
                "4. Dev Caches: If you do development, clean old package caches (`npm cache clean --force`, `pip cache purge`)."
                f"{metrics_info}"
            )
            self.last_problem_category = "disk_cleanup"
            self.last_suggested_action = "check_metrics"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "disk_cleanup"}
            }

        # 4. Frozen Apps / Stuck Processes
        if any(w in q for w in ["frozen", "not responding", "stuck", "force close", "kill process"]) and any(w in q for w in ["app", "program", "window", "process"]):
            msg = (
                f"When an application freezes on Windows, {self.USER_NAME}, here is how to regain control:\n\n"
                "1. Graceful Exit: Click the window and press `Alt + F4`.\n"
                "2. Task Manager: Press `Ctrl + Shift + Esc`, find the frozen app in the list, and click 'End Task'.\n"
                "3. Command Line: Open PowerShell or Command Prompt and run:\n"
                "   `taskkill /f /im <application_name>.exe`\n"
                "   (e.g., `taskkill /f /im notepad.exe`)\n\n"
                "If you tell me the specific name of the application, I can terminate it directly for you."
            )
            self.last_problem_category = "frozen_app"
            self.last_suggested_action = None
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "frozen_app"}
            }

        # 5. Audio / Microphone Malfunctions
        if any(w in q for w in ["microphone", "mic", "sound", "audio", "speaker"]) and any(w in q for w in ["not working", "broken", "silent", "muted", "fix"]):
            msg = (
                f"Audio and microphone issues on Windows are usually permission or default endpoint misconfigurations, {self.USER_NAME}:\n\n"
                "1. Default Device: Right-click the speaker icon in the taskbar > 'Sound settings'. Under Output/Input, verify the correct device is selected as Default.\n"
                "2. Windows Privacy: Go to Windows Settings > 'Privacy & Security' > 'Microphone'. Ensure 'Let apps access your microphone' is toggled ON.\n"
                "3. Physical Switches: Check if your headset or keyboard has a physical mute slider or toggle active.\n"
                "4. Audio Service: Press `Win + R`, type `services.mms`, find 'Windows Audio', right-click and select 'Restart'."
            )
            self.last_problem_category = "audio_issue"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "audio_issue"}
            }

        # 6. File Conversion (Images to PDF, etc.)
        if "convert" in q and "image" in q and "pdf" in q:
            msg = (
                f"Converting images to a PDF on Windows is very easy, {self.USER_NAME}:\n\n"
                "Method 1 (Native Windows - Zero software needed):\n"
                "1. Select the images in File Explorer in the order you want them.\n"
                "2. Right-click and select 'Print' (or press `Ctrl + P`).\n"
                "3. In the Printer dropdown, select 'Microsoft Print to PDF'.\n"
                "4. Click 'Print' and choose where to save your new PDF file!\n\n"
                "Method 2 (Python Automation):\n"
                "If you want to automate this in code, you can use Pillow: `img.save('output.pdf', 'PDF', resolution=100.0)`.\n\n"
                "Would you like me to create a quick Python script or open your files folder?"
            )
            self.last_problem_category = "image_to_pdf"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "image_to_pdf"}
            }

        # 7. Git & Development Troubleshooting
        if "git merge conflict" in q or ("git" in q and "conflict" in q):
            msg = (
                f"Git merge conflicts happen when two branches modify the same lines of code, {self.USER_NAME}. Here is the cleanest way to resolve them:\n\n"
                "1. Run `git status` to see the conflicting files highlighted in red under 'Unmerged paths'.\n"
                "2. Open each conflicting file in VS Code or your text editor.\n"
                "3. Look for the conflict markers:\n"
                "   `<<<<<<< HEAD` (your current branch changes)\n"
                "   `=======` (divider)\n"
                "   `>>>>>>> incoming_branch` (changes being merged)\n"
                "4. Delete the markers and keep the desired code (VS Code provides 'Accept Current', 'Accept Incoming', or 'Accept Both' buttons above the block).\n"
                "5. Save the file, run `git add <file>`, and finish with `git commit`."
            )
            self.last_problem_category = "git_conflict"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "git_conflict"}
            }

        if "modulenotfounderror" in q or "importerror" in q:
            # Try to extract module name
            m_mod = re.search(r"no\s+module\s+named\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", q)
            mod_name = m_mod.group(1) if m_mod else "the package"
            msg = (
                f"`ModuleNotFoundError: No module named '{mod_name}'` means Python cannot find this library in the active environment, {self.USER_NAME}.\n\n"
                "Fix checklist:\n"
                f"1. Install it via pip: `pip install {mod_name}` (or `python -m pip install {mod_name}`).\n"
                "2. Virtual Environment: If you're using a virtual environment (`.venv`), ensure it is activated before running your script:\n"
                "   On Windows PowerShell: `.\\.venv\\Scripts\\Activate.ps1`\n"
                "3. VS Code Interpreter: In VS Code, press `Ctrl + Shift + P`, search 'Python: Select Interpreter', and choose the environment where you installed the package."
            )
            self.last_problem_category = "python_error"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "python_error", "module": mod_name}
            }

        if "virtual environment" in q and ("create" in q or "make" in q or "setup" in q):
            msg = (
                f"Setting up a clean Python virtual environment on Windows takes two quick commands, {self.USER_NAME}:\n\n"
                "1. Create the virtual environment in your project folder:\n"
                "   `python -m venv .venv`\n\n"
                "2. Activate it in PowerShell:\n"
                "   `.\\.venv\\Scripts\\Activate.ps1`\n"
                "   (Or in Command Prompt: `.\\.venv\\Scripts\\activate.bat`)\n\n"
                "3. You will see `(.venv)` in your terminal prompt. Now any `pip install` commands will stay isolated to this project!"
            )
            self.last_problem_category = "venv_setup"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "venv_setup"}
            }

        return None

    # -------------------------------------------------------------------------
    # Natural Language Budget & Math Word Problem Solver
    # -------------------------------------------------------------------------

    def _solve_budget_problem(self, q_raw: str) -> Optional[Dict[str, Any]]:
        q = q_raw.lower()
        if not ("budget" in q or "expense" in q or "calculate my" in q or "monthly" in q):
            return None

        # Look for dollar / rupee amounts or numbers with labels
        # e.g., "25000 rent, 8000 groceries, 3500 bills, 2000 fuel"
        pairs = re.findall(r"(\d+(?:\.\d+)?)\s*(?:for|on|in)?\s*([a-zA-Z\s]+?)(?:,|and|\.|$)", q_raw, flags=re.IGNORECASE)
        if not pairs:
            pairs = re.findall(r"([a-zA-Z\s]+?)\s*(?:is|:|for|=)\s*(\d+(?:\.\d+)?)", q_raw, flags=re.IGNORECASE)
            # Swap if label came first
            pairs = [(val, lbl) for lbl, val in pairs]

        if pairs and len(pairs) >= 2:
            items = []
            total = 0.0
            for val_str, label in pairs:
                lbl_clean = label.strip(" ,.!?").capitalize()
                # filter out non-label words
                if lbl_clean.lower() in ["month", "budget", "expenses", "calculate", "total", "help", "me"]:
                    continue
                try:
                    val = float(val_str)
                    total += val
                    items.append((lbl_clean or "Item", val))
                except ValueError:
                    continue

            if items:
                lines = [f"- {lbl}: {val:,.2f}" for lbl, val in items]
                msg = (
                    f"Here is your calculated expense breakdown, {self.USER_NAME}:\n\n"
                    + "\n".join(lines)
                    + f"\n\n💰 Total Monthly Expenses: {total:,.2f}\n"
                )
                if total > 0:
                    top_item = max(items, key=lambda x: x[1])
                    top_pct = (top_item[1] / total) * 100
                    msg += f"💡 Note: Your largest single expense is {top_item[0]} at {top_item[1]:,.2f} ({top_pct:.1f}% of total)."

                self.last_dialogue_topic = "budget_calculation"
                return {
                    "can_execute": True,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {"topic": "budget_calculation", "total": total, "items": items}
                }

        return None

    # -------------------------------------------------------------------------
    # Productivity & Decision Making Advice
    # -------------------------------------------------------------------------

    def _provide_advice(self, q: str) -> Optional[Dict[str, Any]]:
        if "prioritize" in q and "task" in q:
            msg = (
                f"When you have too many competing priorities, {self.USER_NAME}, I recommend using the Eisenhower Matrix framework:\n\n"
                "1. Urgent & Important (Do First): Critical bugs, deadline deliverables, broken production issues.\n"
                "2. Not Urgent, but Important (Schedule It): Architecture planning, refactoring, learning, writing tests.\n"
                "3. Urgent, but Not Important (Automate or Delegate): Routine status updates, formatting files, repetitive tasks (you can hand these to me!).\n"
                "4. Neither Urgent nor Important (Eliminate): Low-value context-switching.\n\n"
                "Pick your top 2 tasks from category 1, close all unrelated apps, and focus only on them for the next 90 minutes."
            )
            self.last_dialogue_topic = "task_prioritization"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "task_prioritization"}
            }

        if "stay focused" in q or "focus while coding" in q:
            msg = (
                f"Maintaining deep focus while coding requires eliminating cognitive friction, {self.USER_NAME}:\n\n"
                "1. Timeboxing (The Pomodoro Technique): Work with full concentration for 25 minutes, then take a strict 5-minute break away from screens.\n"
                "2. Single-Tab Discipline: Close all browser tabs that aren't directly related to the current file or documentation.\n"
                "3. Instrument Low-Distraction Audio: Put on lofi beats or ambient instrumental tracks (I can launch Spotify for you).\n"
                "4. Write Micro-Checklists: Break your feature into tiny 10-line coding steps so you always know what the immediate next action is."
            )
            self.last_dialogue_topic = "focus_advice"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "focus_advice"}
            }

        if "merge" in q and "rebase" in q:
            msg = (
                f"Here is the practical distinction between `git merge` and `git rebase`, {self.USER_NAME}:\n\n"
                "- `git merge` preserves complete chronological history by creating a dedicated merge commit. It is non-destructive and the safest choice for public or shared branches.\n"
                "- `git rebase` rewrites your branch's commit history by replaying your local commits on top of the target branch, producing a clean linear history without merge bubbles. Ideal for cleaning up local feature branches before opening a PR.\n\n"
                "Golden rule: Never rebase commits that have already been pushed to a public/shared branch!"
            )
            self.last_dialogue_topic = "git_advice"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "git_advice"}
            }

        return None

    # -------------------------------------------------------------------------
    # Follow-Up Continuity
    # -------------------------------------------------------------------------

    def _handle_follow_up(
        self,
        q: str,
        context: Optional[Any] = None,
        executor: Optional[Any] = None
    ) -> Dict[str, Any]:
        action = self.last_suggested_action
        self.last_suggested_action = None  # Reset suggestion

        if action == "check_metrics":
            metrics_data = query_live_metrics()
            cpu = metrics_data.get("cpu_percent", 0)
            mem = metrics_data.get("memory_percent", 0)
            disk = metrics_data.get("disk_free_gb", 0)

            msg = (
                f"Pulled live diagnostics for your machine, {self.USER_NAME}:\n\n"
                f"📊 Current Resource Utilization:\n"
                f"- CPU Load: {cpu}%\n"
                f"- Memory (RAM): {mem}% utilized ({metrics_data.get('memory_used_gb', 0)} GB / {metrics_data.get('memory_total_gb', 0)} GB)\n"
                f"- Primary Storage: {disk} GB available\n\n"
            )
            if mem > 80:
                msg += "⚠️ RAM usage is heavily elevated. Closing heavy browser tabs or IDE processes will free up memory immediately."
            elif cpu > 80:
                msg += "⚠️ CPU usage is high. A background process is under heavy load."
            else:
                msg += "✅ All core hardware metrics are operating comfortably within normal ranges."

            if context and hasattr(context, "set_last_metrics"):
                context.set_last_metrics(metrics_data)

            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"action": "check_metrics", "metrics": metrics_data}
            }

        elif action == "run_network_ping":
            ping_success, latency = check_network_ping("8.8.8.8", timeout=2.0)

            if ping_success:
                msg = (
                    f"✅ Network Diagnostic Completed, {self.USER_NAME}!\n"
                    f"Internet connectivity is ACTIVE. Packets to Google DNS (8.8.8.8) reached successfully with {latency:.1f} ms latency. "
                    "If a specific website is failing, it may be a local DNS caching issue or service outage."
                )
            else:
                msg = (
                    f"⚠️ Network Diagnostic Result: Ping to 8.8.8.8 timed out or failed, {self.USER_NAME}.\n"
                    "Your computer cannot currently reach the external internet. Please check your physical Wi-Fi connection, "
                    "toggle airplane mode off/on, or restart your router."
                )

            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"action": "run_network_ping", "success": ping_success, "latency": latency}
            }

        elif action == "play_relaxing_music":
            msg = (
                f"Understood, {self.USER_NAME}. Standing by to play relaxing music. "
                "I can queue up chill lofi beats on Spotify or YouTube whenever you say the word."
            )
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"action": "play_relaxing_music"}
            }

        # Fallback follow-up
        return {
            "can_execute": True,
            "message": f"Standing by, {self.USER_NAME}. What specific action would you like me to take next?",
            "handler": "dialogue",
            "parameters": {}
        }

    def _respond_general_human(self, q_raw: str) -> Dict[str, Any]:
        msg = (
            f"I hear you, {self.USER_NAME}. I'm here and ready to help. "
            "Whether you'd like me to solve a problem on your computer, execute a workflow, "
            "send a message, manage files, or check system metrics, just let me know how you'd like to proceed!"
        )
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"query": q_raw}
        }
