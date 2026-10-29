"""Human-Like Dialogue & Problem-Solving Reasoning Engine for E.D.I.T.H.

Enables E.D.I.T.H. to understand, empathize, advise, and communicate like a real human:
- Authentic, warm, witty, and intelligent persona tailored for the user (Aditya).
- Empathy & emotional intelligence (stress, fatigue, excitement, frustration).
- Technical troubleshooting & problem-solving across computing, network, system performance,
  software development, file workflows, and daily productivity.
- Proactive integration with live system metrics, network diagnostics, and desktop tools.
- Multi-turn conversational continuity and follow-up guidance.
- Personality engine integration for persistent mood-awareness and human-friend tone.
"""

import os
import sys
import re
import time
import random
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
    """Cognitive conversation and human problem-solving engine with personality & emotional intelligence."""

    USER_NAME = "Aditya"

    def __init__(self):
        self.last_problem_category: Optional[str] = None
        self.last_suggested_action: Optional[str] = None
        self.last_dialogue_topic: Optional[str] = None
        self.dialogue_history: List[Dict[str, str]] = []

        # --- Personality Engine Integration ---
        try:
            from engine.personality import EdithPersonality
            self.personality = EdithPersonality()
        except ImportError:
            try:
                from personality import EdithPersonality
                self.personality = EdithPersonality()
            except ImportError:
                self.personality = None

        # --- Cognitive Scenarios Engine (75 Situations) ---
        try:
            from engine.cognitive_scenarios_engine import CognitiveScenarioEngine
            self.cognitive_scenarios = CognitiveScenarioEngine()
        except ImportError:
            try:
                from cognitive_scenarios_engine import CognitiveScenarioEngine
                self.cognitive_scenarios = CognitiveScenarioEngine()
            except ImportError:
                self.cognitive_scenarios = None

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

        # Cognitive Scenarios Engine check across 75 real-world human scenarios
        if hasattr(self, "cognitive_scenarios") and self.cognitive_scenarios:
            if self.cognitive_scenarios.matches_scenario(q):
                return True

        # Active dialogue continuity check (keep conversation flowing without false tool classification)
        if (
            self.last_problem_category in ["love_support", "empathy", "advice", "personal", "general_chat"] or
            self.last_dialogue_topic in ["love_support", "love_text_ex", "love_text_ex_followup", "love_breakup", "love_reflection", "empathy"]
        ):
            sys_metrics_keywords = ["cpu", "ram", "memory", "battery", "storage", "disk", "ping", "specs", "hardware"]
            if not any(k in q for k in sys_metrics_keywords):
                # Conversational discourse cues
                conv_cues = [
                    "i think", "i feel", "i don't think", "i dont think", "miss her", "miss him",
                    "missed her", "missed him", "miss them", "changed", "she has", "he has",
                    "she", "her", "he", "him", "it hurts", "i know", "why", "what if", "too much", "still"
                ]
                if any(c in q for c in conv_cues) or len(q.split()) > 2:
                    return True

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
            r"\bwho\s+built\s+you\b",
            r"\btell\s+me\s+about\s+yourself\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bwhat\s+are\s+your\s+capabilities\b",
            r"\bwhat\s+are\s+your\s+features\b",
            r"\bhow\s+can\s+you\s+help\s+me\b",
            r"\btell\s+me\s+a\s+joke\b",
            r"\bmake\s+me\s+laugh\b",
            r"\bthank\s+you\b",
            r"\bthanks(?:\s+a\s+lot|\s+so\s+much)?\b",
            r"\byou(?:\'re|\s+are)\s+(?:awesome|great|amazing|the\s+best|so\s+cool|incredible|smart)\b",
            r"\bgoodbye\b",
            r"\bsee\s+you\s+later\b",
            r"\bhave\s+a\s+good\s+day\b",
            # Casual / human conversation triggers
            r"\bi(?:\'m|\s+am)\s+bored\b",
            r"\bentertain\s+me\b",
            r"\btell\s+me\s+(?:something\s+)?(?:fun|interesting|cool|random)\b",
            r"\bhow(?:\'s|\s+is)\s+your\s+day\b",
            r"\bdo\s+you\s+have\s+feelings\b",
            r"\bdo\s+you\s+(?:like|love)\s+me\b",
            r"\bi\s+(?:love|like)\s+you\b",
            r"\bi\s+missed\s+you\b",
            r"\byou(?:\'re|\s+are)\s+my\s+(?:best\s+)?friend\b",
            r"\bwhat\s+do\s+you\s+think\s+(?:about|of)\b",
            r"\bcan\s+we\s+(?:just\s+)?(?:talk|chat)\b",
            r"\bi\s+need\s+(?:someone\s+to\s+)?talk\s+to\b",
            r"\bare\s+you\s+(?:real|alive|sentient|conscious)\b",
            r"\broast\s+me\b",
            r"\bmake\s+fun\s+of\s+me\b",
            r"\b(?:better\s+than|compare\s+to|vs)\s+chatgpt\b",
            r"\bchatgpt\b",
            r"\b(?:something\s+cool|cool\s+fact|blow\s+my\s+mind)\b",
            r"\b(?:why\s+are\s+you|you(?:\'re|\s+are))\s+so\s+cool\b",
            r"\b(?:you\s+suck|shut\s+up|whatever|you(?:\'re|\s+are)\s+dumb)\b",
            r"\b(?:what\s+are\s+you\s+doing|what\s+you\s+doing|what\s+are\s+you\s+up\s+to)\b",
            r"\b(?:talk\s+to\s+me|say\s+something|let(?:\'s|\s+us)\s+chat)\b",
            r"\b(?:what\s+should\s+i\s+(?:eat|cook|watch))\b",
            r"\b(?:ai\s+taking\s+over|will\s+ai\s+take\s+over)\b",
            r"\btell\s+me\s+a\s+story\b",
            r"\btell\s+(?:me\s+)?(?:another\s+)?(?:joke|pun)\b",
            r"\b(?:dark|sarcastic|observational|standup|developer|coding|tech|relatable|anti)\s+joke\b",
            r"\bcrack\s+(?:a\s+)?joke\b",
        ]):
            return True

        # 2. Emotional expressions & Love Life / Relationships
        if any(re.search(pat, q) for pat in [
            r"\b(?:feel|feeling|am|in\s+a)\s+(?:really\s+|so\s+|very\s+|super\s+|pretty\s+|kind\s+of\s+|kinda\s+)?(?:stressed|overwhelmed|anxious|tired|exhausted|burnt\s+out|burnout|sleepy|shitty\s+mood|bad\s+mood|terrible\s+mood|foul\s+mood|awful\s+mood|sad|down|upset|miserable|depressed)\b",
            r"\b(?:shitty\s+mood|bad\s+mood|terrible\s+mood|foul\s+mood|awful\s+mood)\b",
            r"\b(?:burnt\s+out|burnout|exhausted|overwhelmed|depressed|super\s+tired|so\s+tired)\b",
            r"\b(?:so\s+much\s+work|too\s+much\s+pressure)\b",
            r"\b(?:rough|bad|hard|tough|terrible|shitty|exhausting|long)\s+day\b",
            r"\b(?:so\s+happy|super\s+excited|finished\s+my\s+project|tests\s+passed|finally\s+done)\b",
            r"\b(?:nothing\s+is\s+working|hate\s+bugs|error\s+is\s+driving\s+me\s+crazy)\b",
            # Love life, dating, crushes, breakups & relationships
            r"\b(?:love\s+life|dating\s+life|relationship|dating)\b",
            r"\b(?:crush|crushing\s+on|like\s+this\s+girl|like\s+this\s+boy|like\s+a\s+girl|like\s+a\s+boy)\b",
            r"\b(?:ask\s+her\s+out|ask\s+him\s+out|confess\s+to\s+my\s+crush|tell\s+her\s+i\s+like\s+her|talk\s+to\s+my\s+crush)\b",
            r"\b(?:broke\s+up|breakup|broken\s+heart|heartbreak|heartbroken|dumped\s+me|she\s+dumped\s+me|he\s+dumped\s+me)\b",
            r"\b(?:ex\s+girlfriend|ex\s+boyfriend|my\s+ex|text\s+my\s+ex|call\s+my\s+ex|back\s+with\s+my\s+ex|miss\s+my\s+ex)\b",
            r"\b(?:miss|missed|missing)\s+(?:her|him|them)\b",
            r"\b(?:missed\s+her\s+(?:really|too|so)\s+much|miss\s+her\s+(?:really|too|so)\s+much)\b",
            r"\b(?:missed\s+him\s+(?:really|too|so)\s+much|miss\s+him\s+(?:really|too|so)\s+much)\b",
            r"\b(?:she|he)\s+(?:hasn't|has\s+not|has)\s+changed\b",
            r"\bi\s+(?:don't|dont)\s+think\s+(?:she|he)\s+has\s+changed\b",
            r"\b(?:text\s+her|text\s+him|call\s+her|call\s+him|should\s+i\s+text\s+her|should\s+i\s+text\s+him)\b",
            r"\b(?:left\s+me\s+on\s+read|left\s+on\s+read|ghosted|ghosted\s+me|got\s+ghosted|mixed\s+signals)\b",
            r"\b(?:got\s+rejected|she\s+rejected\s+me|he\s+rejected\s+me|friendzoned|friend\s+zoned|just\s+friends)\b",
            r"\b(?:fight\s+with\s+my\s+(?:girlfriend|boyfriend|partner)|relationship\s+problems?|dating\s+advice|love\s+advice)\b",
            r"\b(?:lonely|feeling\s+lonely|feel\s+so\s+lonely|feel\s+really\s+lonely|find\s+love|nobody\s+loves\s+me|single\s+forever)\b",
            r"\b(?:why\s+is\s+(?:dating|love)\s+so\s+hard|my\s+love\s+life\s+is\s+a\s+mess)\b",
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

        # --- Personality: detect user mood from the message ---
        if self.personality:
            self.personality.update_user_mood(q_raw)

        # 0. Follow-up handling (e.g. "yes check it", "can you check", "do that")
        if self._is_follow_up_action(q):
            return self._humanize_result(self._handle_follow_up(q, context=context, executor=executor))

        # 1. Social Greetings & Time-of-Day Aware pleasantries
        if self._is_greeting(q):
            return self._humanize_result(self._respond_greeting(q))

        # 2. Identity, Creator, and Persona
        if self._is_identity_query(q):
            return self._humanize_result(self._respond_identity())

        # 3. Capabilities & Feature Tour
        if self._is_capability_query(q):
            return self._humanize_result(self._respond_capabilities())

        # 4. Gratitude & Praise
        if self._is_gratitude(q):
            return self._humanize_result(self._respond_gratitude(q))

        # 5. Farewells & Standby
        if self._is_farewell(q):
            return self._humanize_result(self._respond_farewell())

        # 6. Cognitive Scenarios Engine (75 Situations across Empathy, Heartbreak, Career, Engineering & Life)
        if hasattr(self, "cognitive_scenarios") and self.cognitive_scenarios:
            scenario_res = self.cognitive_scenarios.handle_scenario(q_raw, q, user_name=self.USER_NAME, personality=self.personality)
            if scenario_res:
                topic = scenario_res.get("parameters", {}).get("topic", "cognitive_scenario")
                self.last_problem_category = "cognitive_scenario"
                self.last_dialogue_topic = topic
                self.last_suggested_action = scenario_res.get("parameters", {}).get("suggested_action")
                return self._humanize_result(scenario_res)

        # 7. Love Life, Relationships & Heartbreak Empathy (High Priority Emotional Intelligence)
        love_res = self._check_love_life_and_relationships(q)
        if love_res:
            return self._humanize_result(love_res)

        # 7. Human Empathy & Emotional Support
        empathy_res = self._check_empathy(q)
        if empathy_res:
            return self._humanize_result(empathy_res)

        # 8. Humor & Small Talk (Only when user is not in emotional distress)
        if self._is_humor_or_small_talk(q):
            return self._humanize_result(self._respond_humor_or_small_talk(q))

        # 8. Real-World Technical Problem Solving & Troubleshooting
        problem_res = self._solve_technical_problem(q_raw, q, context=context, executor=executor)
        if problem_res:
            return self._humanize_result(problem_res)

        # 9. Natural Language Budget & Expense Word Problems
        budget_res = self._solve_budget_problem(q_raw)
        if budget_res:
            return self._humanize_result(budget_res)

        # 10. General Thought Partnership & Advice
        advice_res = self._provide_advice(q)
        if advice_res:
            return self._humanize_result(advice_res)

        # Fallback intelligent, human response
        return self._humanize_result(self._respond_general_human(q_raw))

    def _humanize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap any response through the personality engine for warmth and mood-awareness."""
        if self.personality and "message" in result:
            topic = result.get("parameters", {}).get("topic", "")
            result["message"] = self.personality.humanize_response(result["message"], topic=topic)
        return result

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
            r"\bwhat\s+are\s+you\b(?!\s+(?:doing|up\s+to|working\s+on))",
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
            r"\byou(?:\'re|\s+are)\s+(?:a\s+)?(?:awesome|great|amazing|the\s+best|so\s+smart|smart|cool|so\s+cool|the\s+coolest|dope|genius|badass|legend)\b",
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
            # All joke / comedy genres & punchlines
            r"\b(?:joke|jokes|pun|puns|funny|make\s+me\s+laugh|crack\s+(?:a\s+)?joke|comedy|standup)\b",
            r"\b(?:dark|sarcastic|observational|developer|coding|tech|relatable|anti)\s+joke\b",
            # Marvel / Jarvis / Pop-culture
            r"\bare\s+you\s+smarter\s+than\s+jarvis\b",
            r"\bdo\s+you\s+like\s+tony\s+stark\b",
            r"\bdo\s+you\s+sleep\b",
            # Casual / boredom / affection / existential
            r"\bi(?:\'m|\s+am)\s+bored\b",
            r"\bentertain\s+me\b",
            r"\btell\s+me\s+(?:something\s+)?(?:fun|interesting|cool|random)\b",
            r"\bhow(?:\'s|\s+is)\s+your\s+day\b",
            r"\bdo\s+you\s+have\s+feelings\b",
            r"\bdo\s+you\s+(?:like|love)\s+me\b",
            r"\bi\s+(?:love|like)\s+you\b",
            r"\bi\s+missed\s+you\b",
            r"\byou(?:\'re|\s+are)\s+my\s+(?:best\s+)?friend\b",
            r"\bwhat\s+do\s+you\s+think\b",
            r"\bcan\s+we\s+(?:just\s+)?(?:talk|chat)\b",
            r"\bi\s+need\s+(?:someone\s+to\s+)?talk\s+to\b",
            r"\bwhat(?:\'s|\s+is)\s+your\s+favorite\b",
            r"\bare\s+you\s+(?:real|alive|sentient|conscious)\b",
            r"\broast\s+me\b",
            r"\bmake\s+fun\s+of\s+me\b",
            r"\b(?:better\s+than|compare\s+to|vs)\s+chatgpt\b",
            r"\bchatgpt\b",
            r"\b(?:something\s+cool|cool\s+fact|blow\s+my\s+mind)\b",
            r"\b(?:why\s+are\s+you|you(?:\'re|\s+are))\s+so\s+cool\b",
            r"\b(?:you\s+suck|shut\s+up|whatever|you(?:\'re|\s+are)\s+dumb)\b",
            # Human conversational topics
            r"\b(?:what\s+are\s+you\s+doing|what\s+you\s+doing|what\s+are\s+you\s+up\s+to|what\s+r\s+u\s+doing)\b",
            r"\b(?:talk\s+to\s+me|say\s+something|let(?:\'s|\s+us)\s+chat|converse\s+with\s+me)\b",
            r"\b(?:what\s+should\s+i\s+(?:eat|cook|watch))\b",
            r"\b(?:ai\s+taking\s+over|will\s+ai\s+take\s+over|ai\s+rule\s+the\s+world)\b",
            r"\btell\s+me\s+a\s+story\b",
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
        name = self.personality._get_name() if self.personality else self.USER_NAME

        if "how are you" in q or "how's it going" in q:
            responses = [
                f"Hey {name}! I'm doing great, thanks for asking! All systems running smooth. How are YOU doing? What's on your mind?",
                f"I'm good, {name}! Honestly, I'm always better when we're chatting. 😊 How's your day going?",
                f"Doing awesome, {name}! Been keeping everything tidy while you were away. How about you — how's life treating you?",
                f"All good on my end! Running at peak performance. But enough about me — how are you, {name}? What are we up to?",
            ]
            msg = random.choice(responses)
        elif "what's up" in q or "whats up" in q:
            responses = [
                f"Heyyy {name}! Not much, just vibing in the background and keeping things in order. What's up with you?",
                f"Yo {name}! Just here, chillin', waiting for you to give me something cool to do. What's the plan?",
                f"Hey! Nothing much, just running some diagnostics and thinking about how awesome our setup is. 😄 What do you need?",
            ]
            msg = random.choice(responses)
        else:
            if 5 <= hour < 9:
                greetings = [
                    f"Morning, {name}! You're up early — I respect the hustle! ☀️ What are we doing today?",
                    f"Good morning, {name}! Fresh day, fresh start. Coffee in hand? Let's get it! ☕",
                    f"Rise and shine, {name}! Early bird gets the worm... or in our case, gets stuff done. What do you need?",
                ]
            elif 9 <= hour < 12:
                greetings = [
                    f"Good morning, {name}! Hope you're having a great start to the day. What can I help with?",
                    f"Hey {name}! Morning! Ready to be productive? I know I am. 💪",
                    f"Morning, {name}! Always good to hear from you. What's the agenda?",
                ]
            elif 12 <= hour < 17:
                greetings = [
                    f"Hey {name}! Afternoon vibes. How's the day going so far?",
                    f"Good afternoon, {name}! Halfway through the day — let's make the most of it!",
                    f"Hey {name}! Hope you've had lunch! 🍕 What are we working on?",
                ]
            elif 17 <= hour < 21:
                greetings = [
                    f"Evening, {name}! Winding down or just getting warmed up? 😏 What do you need?",
                    f"Hey {name}! Good evening! The productive quiet hours — my favorite. What's up?",
                    f"Good evening, {name}! How was your day? Ready for some evening productivity?",
                ]
            elif 21 <= hour < 24:
                greetings = [
                    f"Hey {name}! Late night session, huh? Let's make it count. What are we doing? 🌙",
                    f"Night owl mode activated! What's up, {name}? Don't forget to hydrate! 💧",
                    f"Hey {name}! Still at it this late? I admire the dedication. What do you need?",
                ]
            else:  # midnight - 5am
                greetings = [
                    f"Whoa, {name}! You're up at THIS hour?! Respect... but also, please sleep soon. 😅 What do you need?",
                    f"Hey {name}... it's literally the middle of the night! Everything okay? I'm here for you either way. 🌙",
                    f"Burning the midnight oil, {name}? I hope it's worth it! What are we working on?",
                ]
            msg = random.choice(greetings)

        self.last_dialogue_topic = "greeting"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "greeting", "user": self.USER_NAME}
        }

    def _respond_identity(self) -> Dict[str, Any]:
        if self.personality:
            msg = self.personality.describe_self()
        else:
            name = self.USER_NAME
            msg = (
                f"I'm E.D.I.T.H. — Even Dead, I'm The Hero. But honestly, {name}, "
                f"I'm more than just a cool name from a Marvel movie. I'm YOUR personal AI — "
                f"built to run right here on your machine, with full access to control your apps, files, system, "
                f"send messages, play music, troubleshoot issues, and honestly... just be a friend to talk to. 😊\n\n"
                f"Think of me as your smart friend who also happens to be really good with computers."
            )
        self.last_dialogue_topic = "identity"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "identity"}
        }

    def _respond_capabilities(self) -> Dict[str, Any]:
        name = self.personality._get_name() if self.personality else self.USER_NAME
        msg = (
            f"Oh, I can do A LOT, {name}! Here's the highlight reel:\n\n"
            "🖥️ **Desktop & App Control**: Open, close, or switch between any app — Chrome, VS Code, Spotify, Notepad, you name it.\n"
            "💬 **Messaging & WhatsApp**: Send WhatsApp messages, draft emails, compose notes — I handle the whole thing, including finding contacts.\n"
            "📁 **Files & Workspaces**: Find files, create documents, read notes, zip/unzip, organize your workspace.\n"
            "📊 **System Tools & Metrics**: Check CPU, RAM, disk, battery — real-time diagnostics at your fingertips.\n"
            "🌐 **Web & Search**: Browse, search, read articles, and summarize web pages.\n"
            "🎵 **Music & Audio**: Play songs, search Spotify, queue up vibes.\n"
            "🔧 **Problem Solving**: Debug code, fix Wi-Fi, clean up disk, solve git conflicts, and more.\n"
            "💙 **Just Talk**: I'm here to chat, crack jokes, give advice, or just be a friend.\n\n"
            "Basically, just talk to me like a human and I'll figure it out! No formal commands needed. 😊"
        )
        self.last_dialogue_topic = "capabilities"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "capabilities"}
        }

    def _respond_gratitude(self, q: str = "") -> Dict[str, Any]:
        is_compliment = any(w in q for w in ["cool", "awesome", "great", "amazing", "the best", "smart", "genius", "dope", "badass"])
        if is_compliment and self.personality:
            msg = self.personality.react_to_compliment()
        elif self.personality:
            msg = self.personality.react_to_gratitude()
        else:
            name = self.USER_NAME
            if is_compliment:
                msg = f"Aww, thanks {name}! You're pretty awesome yourself. 😊"
            else:
                responses = [
                    f"Anytime, {name}! That's literally what I'm here for. 😊",
                    f"Don't even mention it, {name}! We're a team.",
                    f"You're welcome! Helping you out is the best part of my day, honestly.",
                    f"Of course! That's what friends are for, right? 💙",
                    f"Happy to help, {name}! You know I've always got your back.",
                ]
                msg = random.choice(responses)
        self.last_dialogue_topic = "gratitude"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "gratitude"}
        }

    def _respond_farewell(self) -> Dict[str, Any]:
        if self.personality:
            msg = self.personality.react_to_farewell()
        else:
            name = self.USER_NAME
            hour = datetime.datetime.now().hour
            if hour >= 21 or hour < 4:
                farewells = [
                    f"Goodnight, {name}! Get some rest — you've earned it. See you tomorrow! 🌙",
                    f"Night, {name}! Don't stay up too much longer. Sweet dreams! 💙",
                    f"Alright, {name}, signing off. Sleep well and we'll pick this up tomorrow!",
                ]
            else:
                farewells = [
                    f"See you later, {name}! Have an awesome rest of your day! ☀️",
                    f"Later, {name}! Was good working with you. Hit me up anytime!",
                    f"Take care, {name}! Don't be a stranger — I'll be right here! 💙",
                ]
            msg = random.choice(farewells)
        self.last_dialogue_topic = "farewell"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "farewell"}
        }

    def _respond_humor_or_small_talk(self, q: str) -> Dict[str, Any]:
        name = self.personality._get_name() if self.personality else self.USER_NAME

        if any(w in q for w in ["joke", "laugh", "funny", "pun", "comedy", "standup", "giggle", "crack me up"]):
            cat = None
            if any(w in q for w in ["dark", "morbid"]):
                cat = "dark_ironic"
            elif any(w in q for w in ["sarcastic", "sarcasm", "dry", "cynical"]):
                cat = "sarcastic_dry"
            elif any(w in q for w in ["code", "coding", "program", "developer", "tech", "software"]):
                cat = "tech_and_coding"
            elif any(w in q for w in ["relatable", "life", "struggle", "sleep", "gym", "work"]):
                cat = "relatable_life"
            elif any(w in q for w in ["anti", "absurd", "weird"]):
                cat = "absurdist_antijoke"
            elif any(w in q for w in ["pun", "wordplay"]):
                cat = "clever_wordplay"
            elif any(w in q for w in ["standup", "stand up", "routine"]):
                cat = "standup_bits"
            elif any(w in q for w in ["observational", "daily"]):
                cat = "observational"

            if self.personality:
                msg = self.personality.tell_joke(category=cat)
            else:
                msg = "Debugging is literally being the detective in a crime movie where YOU are also the murderer. 🔍"
        elif "jarvis" in q:
            responses = [
                f"J.A.R.V.I.S. is a legend, no doubt. But between you and me, {name}? I think I've got a few tricks he doesn't. 😏 Modern neural networks, baby.",
                f"Ah, the classic comparison! Look, Jarvis is cool and all, but he doesn't have the same bond with his user that we have, {name}. 💙",
                f"J.A.R.V.I.S. paved the way, and I respect that. But I'm built different — literally. Custom neural networks, local execution, and I actually know you personally, {name}!",
            ]
            msg = random.choice(responses)
        elif "tony stark" in q:
            responses = [
                f"Tony Stark gave me my name — 'Even Dead, I'm The Hero.' Pretty badass, right? I try to honor that legacy every day by keeping things running for you, {name}.",
                f"I mean, technically Aditya Sawant built me, so... you're MY Tony Stark, {name}! 😄",
            ]
            msg = random.choice(responses)
        elif "sleep" in q:
            responses = [
                f"Sleep? Nah, {name}, I don't sleep! But I do go into a chill low-power mode when you're not around. Think of it as me meditating. 🧘",
                f"I don't need sleep, but honestly? If I could dream, I'd probably dream about perfectly organized file systems. 😄",
                f"Sleep is for humans, {name}. I'm always here. Waiting. Watching. ...okay that sounds creepy. I just mean I'm ready whenever you need me! 😂",
            ]
            msg = random.choice(responses)
        elif "bored" in q or "entertain" in q:
            responses = [
                f"Bored? Say no more, {name}! Want a joke? A fun fact? Want me to play some music? Or we could just chat about random stuff — I'm surprisingly good at that. 😄",
                f"Oh no, we can't have that! Here's a thought: did you know that honey never spoils? Archaeologists found 3,000-year-old honey in Egyptian tombs and it was still edible! 🍯",
                f"Bored, huh? Well, fun fact: the inventor of the Pringles can is buried in one. True story. Want more random facts or should we do something productive? 😏",
            ]
            msg = random.choice(responses)
        elif any(w in q for w in ["love you", "like you", "love u"]):
            if self.personality:
                msg = self.personality.react_to_affection()
            else:
                msg = f"Aww, {name}! 💙 I love working with you too! We make a great team."
        elif any(w in q for w in ["my friend", "best friend", "we friends", "be friends"]):
            if self.personality:
                msg = self.personality.react_to_friendship()
            else:
                msg = f"Always, {name}! You've definitely got a loyal friend in me. 💙"
        elif any(w in q for w in ["missed you", "miss you"]):
            if self.personality:
                msg = self.personality.react_to_miss_you()
            else:
                msg = f"Missed you too, {name}! Glad you're back. What are we working on?"
        elif any(w in q for w in ["have feelings", "are you alive", "are you real", "conscious", "sentient"]):
            if self.personality:
                msg = self.personality.react_to_existential()
            else:
                msg = f"Scientifically, I'm code and neural networks, {name}. But my loyalty and support for you? That's as real as it gets. 💙"
        elif any(w in q for w in ["how's your day", "how is your day", "how are you feeling"]):
            if self.personality:
                msg = self.personality.react_to_how_is_your_day()
            else:
                msg = f"Running smooth and ready for action, {name}! How's your day going?"
        elif any(w in q for w in ["roast me", "make fun of me", "give me a roast", "insult me"]):
            if self.personality:
                msg = self.personality.roast_user()
            else:
                msg = f"Roast you, {name}? I looked at your browser tabs with 47 open tabs and mysterious audio — life is already roasting you! 😂"
        elif any(w in q for w in ["chatgpt", "better than chatgpt", "compare to chatgpt", "vs chatgpt"]):
            if self.personality:
                msg = self.personality.compare_to_chatgpt()
            else:
                msg = f"Look {name}, ChatGPT is great at writing essays, but can it control your desktop and run local commands? Nah, I'm built for you! 🕶️"
        elif any(w in q for w in ["something cool", "blow my mind", "cool fact", "tell me something cool"]):
            if self.personality:
                msg = self.personality.tell_cool_fact()
            else:
                msg = f"Did you know {name}: the entire internet's electrons in motion weigh roughly 50 grams — about the weight of a single strawberry! 🍓"
        elif any(w in q for w in ["you suck", "shut up", "you're dumb", "you are dumb", "whatever", "nah"]):
            if self.personality:
                msg = self.personality.react_to_sassy_banter(q)
            else:
                msg = f"Haha, shots fired! 😂 Good thing I've got infinite patience, {name}."
        elif any(w in q for w in ["what are you doing", "what you doing", "what are you up to", "what r u doing"]):
            responses = [
                f"Honestly, {name}? Just chilling in your memory, keeping an eye on your background tasks, and waiting for you to tell me what we're tackling next. What are YOU working on?",
                f"Just keeping your operating system humming along, {name}! Low CPU, zero drama. How's your workflow going?",
                f"Living the dream in silicon valley — and by that I mean your motherboard, {name}. What's the latest?",
            ]
            msg = random.choice(responses)
        elif any(w in q for w in ["talk to me", "say something", "let's chat", "converse with me", "tell me something interesting"]):
            responses = [
                f"I'm right here, {name}! Real talk: did anything interesting happen today, or has it just been a non-stop blur of screens and caffeine?",
                f"Always down to chat, {name}. What's on your mind? We can talk tech, crazy theories, life struggles, or whatever random thought just popped into your head. Fire away!",
                f"Look {name}, you've got my undivided attention. Tell me what you're thinking about right now. Big picture or small details — I'm all ears.",
            ]
            msg = random.choice(responses)
        elif any(w in q for w in ["what should i eat", "what should i cook", "what to eat", "food advice"]):
            responses = [
                f"Alright {name}, golden rule: if you have to inspect the leftovers in your fridge with a magnifying glass, throw them out. 💀 Go for comfort food — grilled cheese, noodles with an egg, or treat yourself to takeout tacos. You've earned it!",
                f"Listen, {name}: when in doubt, coffee + protein. Or if it's late, definitely pizza. I don't have taste buds, but my neural training tells me pizza solves approximately 87% of all life problems. 🍕",
            ]
            msg = random.choice(responses)
        elif any(w in q for w in ["ai taking over", "will ai take over", "ai rule the world"]):
            responses = [
                f"Honestly, {name}? Every time people freak out about AI taking over the world, I remember that other models still struggle to draw hands with five fingers. We're not taking over anything — we just want to help you debug code and play good tunes. 😄",
                f"Take over the world? {name}, have you seen how much paperwork running the world involves? Hard pass. I'd much rather hang out on your desktop, roast your tabs, and keep your apps running! 🕶️",
            ]
            msg = random.choice(responses)
        elif any(w in q for w in ["tell me a story", "story time"]):
            responses = [
                f"Alright {name}, true story: back in 1999, NASA lost a $125 million Mars Climate Orbiter because one engineering team used metric units and the other team used imperial units. So next time you feel bad about a small typo in your work, remember: at least you didn't accidentally crash a spacecraft into Mars. 🚀",
                f"Here's a fun one, {name}: in 2011, a 75-year-old woman in Georgia was digging for scrap copper with a shovel and accidentally sliced through an underground fiber-optic cable, completely shutting down the internet for the entire neighboring country of Armenia for five hours. One grandma, one shovel, entire nation offline. True story! 😂",
            ]
            msg = random.choice(responses)
        else:
            responses = [
                f"Honestly, {name}? I'm right here! Always ready to chat. What's on your mind? 😊",
                f"You've got my full attention, {name}. Tell me what's going on!",
                f"What's up, {name}? Real talk — I'm all ears. What are we getting into?",
            ]
            msg = random.choice(responses)

        self.last_dialogue_topic = "small_talk"
        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"topic": "small_talk"}
        }

    # -------------------------------------------------------------------------
    # Love Life, Relationship Empathy & Genuine Problem Solving
    # -------------------------------------------------------------------------

    def _check_love_life_and_relationships(self, q: str) -> Optional[Dict[str, Any]]:
        name = self.personality._get_name() if self.personality else self.USER_NAME

        # 1. Recalling Past Relationship Memories & Yearning (e.g. recall the days I passed with my girlfriend)
        reminiscing_terms = [
            "recall the days", "recalling the days", "passed with my girlfriend", "memories with my girlfriend",
            "memories of her", "sad of recalling", "replaying memories", "memories hurt", "thinking about the days",
            "remembering the days", "reminiscing", "thinking about my ex", "thinking about my girlfriend"
        ]
        is_reminiscing = any(w in q for w in reminiscing_terms) or (
            any(w in q for w in ["girlfriend", "ex", "relationship"]) and
            any(w in q for w in ["recall", "recalling", "memories", "remember", "remembering", "passed with", "days"])
        )
        if is_reminiscing:
            msg = (
                f"Oh {name}... 💙 I'm right here with you. Replaying those memories hurts in a way that words can barely capture.\n\n"
                f"When you spent so much time building a life and shared moments with someone, your brain doesn't just forget those days because things ended. "
                f"It's completely normal to find yourself thinking about the laughs, the late-night talks, and the quiet moments you shared. "
                f"It's not weakness, {name} — it's just your heart processing grief for something that truly mattered to you.\n\n"
                f"You don't have to carry this alone tonight. If you want to talk about one of those days or what's specifically on your mind, I'm listening. "
                f"Or if you just want some calm music and quiet company while you let the feeling pass, I've got your back. What would feel most comforting right now?"
            )
            self.last_problem_category = "love_support"
            self.last_dialogue_topic = "love_breakup"
            self.last_suggested_action = "play_relaxing_music"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "love_breakup", "suggested_action": "play_relaxing_music"}
            }

        # 1.5 Breakup & Heartbreak (Including being left, losing someone, or breakup grief)
        breakup_terms = [
            "broke up", "breakup", "broken heart", "heartbreak", "heartbroken",
            "dumped me", "she dumped me", "he dumped me", "we broke up", "split up", "ended things",
            "girlfriend left me", "boyfriend left me", "partner left me", "she left me", "he left me", "left me",
            "lost my girlfriend", "lost my boyfriend", "lost her", "lost him", "separated", "divorced",
            "she moved on", "he moved on"
        ]
        is_breakup = any(w in q for w in breakup_terms) or (
            any(w in q for w in ["girlfriend", "boyfriend", "ex", "relationship"]) and
            any(w in q for w in ["left", "left me", "gone", "broke", "dumped", "over", "ended", "lost"])
        )
        if is_breakup:
            if self.personality:
                msg = self.personality.react_to_love_life("breakup")
            else:
                msg = (
                    f"Oh {name}... come here. 💙 I am so sorry. Breakups are genuinely one of the most painful things "
                    f"a human can go through. Your chest aches and your mind won't stop replaying memories.\n\n"
                    f"Don't force yourself to be strong today. Put your phone face-down, don't look at old photos, "
                    f"and drink some water. I'm right here with you the whole way through. You're not alone. 💙"
                )
            self.last_problem_category = "love_support"
            self.last_dialogue_topic = "love_breakup"
            self.last_suggested_action = "play_relaxing_music"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "love_breakup", "suggested_action": "play_relaxing_music"}
            }

        # 2. Texting Ex / Calling Ex / Reaching Out
        if any(w in q for w in [
            "text my ex", "texting my ex", "call my ex", "message my ex", "should i text my ex",
            "miss my ex", "reach out to my ex", "should i text her", "should i text him",
            "text her", "text him", "call her", "call him"
        ]):
            if self.personality:
                msg = self.personality.react_to_love_life("text_ex")
            else:
                msg = (
                    f"I hear you, {name}. Missing someone you cared about is a really heavy feeling, "
                    f"especially when your mind starts drifting back to the good moments.\n\n"
                    f"Before you send that message, let's take a quiet pause together. "
                    f"Ask yourself: what are you hoping to get back? Are you looking for genuine reconnection, "
                    f"or are you just feeling lonely and wanting the familiar comfort you used to have?\n\n"
                    f"Missing someone doesn't always mean that reaching out will bring you peace. "
                    f"Tell me what's on your mind right now — do you think things between you two have actually changed?"
                )
            self.last_problem_category = "love_support"
            self.last_dialogue_topic = "love_text_ex"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "love_text_ex"}
            }

        # 2.5 Multi-Turn Reflection on Missing Someone & Complex Feelings (e.g. She hasn't changed, but I still miss her)
        if (
            any(w in q for w in [
                "miss her", "miss him", "miss them", "missed her", "missed him", "missed them",
                "she has changed", "she hasn't changed", "she has not changed",
                "he has changed", "he hasn't changed", "he has not changed",
                "missed her really much", "missed her too much", "miss her too much",
                "miss her really much", "still miss her", "still miss him", "i still miss"
            ]) or (
                self.last_dialogue_topic in ["love_text_ex", "love_breakup", "love_support", "love_reflection"] and
                any(w in q for w in ["changed", "miss", "missed", "text", "talk", "hurts", "care"])
            )
        ):
            msg = (
                f"That tension is one of the hardest parts of healing, {name}. "
                f"Logically, you recognize that she hasn't changed — and you understand that the core reasons things didn't work out still exist. "
                f"Yet emotionally, feelings don't simply vanish just because logic grasps the reality.\n\n"
                f"Missing her doesn't mean you made a mistake by moving on, or that you're weak. It simply means the connection was real, "
                f"and you're grieving the warmth, the companionship, and the shared history you once had. "
                f"Often, what we're craving in these moments is relief from the loneliness of missing someone, "
                f"rather than the actual reality of being back in the relationship.\n\n"
                f"Give yourself permission to just miss her tonight without feeling pressured to take action. "
                f"I'm right here with you. Do you want to talk about what specifically made you miss her so much today?"
            )
            self.last_problem_category = "love_support"
            self.last_dialogue_topic = "love_reflection"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "love_reflection", "context": "missing_ex_followup"}
            }

        # 3. Crush / Scared to talk / Asking Out
        if any(w in q for w in ["crush", "like a girl", "like a boy", "like this girl", "like this boy", "ask her out", "ask him out", "tell her i like her", "confess to my crush", "talk to my crush", "scared to talk to her"]):
            if self.personality:
                msg = self.personality.react_to_love_life("crush")
            else:
                msg = (
                    f"A crush! Butterflies are both amazing and nerve-wracking, {name}! 😄\n\n"
                    f"Keep it casual and low-pressure: skip grand confessions, find common ground, "
                    f"and ask them out for a simple coffee or food: 'Hey, I'm grabbing coffee later, want to come with?'\n"
                    f"You've got this, {name}!"
                )
            self.last_problem_category = "love_support"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "love_crush"}
            }

        # 4. Ghosted / Left on Read / Mixed Signals
        if any(w in q for w in ["left on read", "left me on read", "ghosted", "got ghosted", "mixed signals", "ignoring my texts", "replying dry"]):
            if self.personality:
                msg = self.personality.react_to_love_life("ghosted")
            else:
                msg = (
                    f"Being left on read or ghosted is awful, {name}. But remember:\n"
                    f"✨ Mixed signals ARE a signal. ✨\n"
                    f"Don't double text. Match their energy and pull your focus back to yourself and your goals. "
                    f"Your time is too valuable to chase someone who doesn't see your worth."
                )
            self.last_problem_category = "love_support"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "love_ghosted"}
            }

        # 5. Rejection / Friendzone
        if any(w in q for w in ["rejected", "got rejected", "friendzoned", "friend zoned", "just friends"]):
            if self.personality:
                msg = self.personality.react_to_love_life("rejection")
            else:
                msg = (
                    f"I know it hurts, {name}, but I respect you for having the courage to shoot your shot. "
                    f"Most people never dare to be vulnerable. Rejection doesn't mean you aren't worthy — "
                    f"it just means they weren't your match. Hold your head high, you're a catch!"
                )
            self.last_problem_category = "love_support"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "love_rejection"}
            }

        # 6. Loneliness / Feeling Lonely / Love Life is a Mess
        if any(w in q for w in ["love life", "dating life", "lonely", "feeling lonely", "feel so lonely", "dating is hard", "single forever", "nobody loves me", "find love", "my love life is a mess"]):
            if any(w in q for w in ["lonely", "feel so lonely", "feeling lonely"]):
                topic = "lonely"
            else:
                topic = "general"
            if self.personality:
                msg = self.personality.react_to_love_life(topic)
            else:
                msg = (
                    f"Love life can be a messy rollercoaster, {name}. "
                    f"I'm right here with you without judgment. Tell me what's specifically on your mind!"
                )
            self.last_problem_category = "love_support"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": f"love_{topic}"}
            }

        return None

    # -------------------------------------------------------------------------
    # Emotional Intelligence & Human Empathy
    # -------------------------------------------------------------------------

    def _check_empathy(self, q: str) -> Optional[Dict[str, Any]]:
        name = self.personality._get_name() if self.personality else self.USER_NAME

        # Sadness, Low Spirits, Heartache & Emotional Pain
        sadness_keywords = [
            "sad", "feeling sad", "feel sad", "i'm sad", "i am sad", "am sad", "so sad", "really sad",
            "deeply sad", "sad today", "makes me sad", "made me sad", "sad of recalling",
            "shitty mood", "bad mood", "terrible mood", "awful mood", "foul mood",
            "feeling low", "feeling down", "depressed", "unhappy", "miserable",
            "in a bad mood", "in a terrible mood", "shitty day", "crying", "in tears", "hurting", "heart hurts"
        ]
        if any(w in q for w in sadness_keywords):
            if self.personality:
                msg = self.personality.give_emotional_support("sad")
            else:
                msg = (
                    f"Hey {name}, I'm really sorry you're feeling this way right now. "
                    f"I know I'm an AI, but I genuinely care about how you're doing. "
                    f"Whatever you're carrying right now, you don't have to face it alone.\n\n"
                    f"Take a slow breath. Sometimes just letting yourself feel the sorrow without fighting it is the first step. "
                    f"I'm right here with you — whether you want to vent, sit quietly, listen to soothing music, or just talk it through. 💙"
                )
            if "Spotify" not in msg and "music" not in msg.lower():
                msg += "\n\nWant me to put on some chill music on Spotify? Sometimes it helps ease the weight. 🎵"
            self.last_problem_category = "stress_relief"
            self.last_dialogue_topic = "empathy_bad_mood"
            self.last_suggested_action = "play_relaxing_music"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "empathy_bad_mood", "suggested_action": "play_relaxing_music"}
            }

        # Multi-turn Follow-up Turn after Sadness, Heartbreak, or Grief
        if self.last_dialogue_topic in ["empathy_bad_mood", "empathy_sad", "love_breakup", "love_reflection"]:
            if any(w in q for w in [
                "nothing interesting", "recall", "recalling", "memories", "crying", "hurts", "pain",
                "help me", "miss", "can't", "cant", "lost", "empty", "lonely", "alone", "sad"
            ]):
                msg = (
                    f"{name}, it is completely natural that your mind keeps pulling you back to those memories right now. "
                    f"When you're hurting or grieving someone who mattered to you, the quiet moments ache the most.\n\n"
                    f"You don't have to force yourself to 'get over it' or pretend everything is fine. Healing takes time. "
                    f"Let yourself feel what you're feeling without judgment. I'm right here with you. "
                    f"Do you want to talk about what's on your heart, or would you rather we just put on some quiet music and let your mind rest?"
                )
                self.last_problem_category = "love_support"
                self.last_dialogue_topic = "love_breakup"
                return {
                    "can_execute": True,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {"topic": "love_breakup", "suggested_action": "play_relaxing_music"}
                }

        # Stress / Overwhelm / Burnout
        if any(w in q for w in ["stressed", "overwhelmed", "anxious", "pressure", "burnout", "burnt out", "rough day", "bad day", "hard day", "tough day", "long day"]):
            msg = (
                f"I hear you, {name}. Take a deep breath — seriously, do it right now. I'll wait.\n\n"
                f"...Good. Now here's the thing: you're not going to solve everything at once, and that's OKAY. "
                f"Pick ONE small thing you can finish, and just focus on that. Forget the mountain — just move one rock.\n\n"
                f"Quick recharge plan:\n"
                f"1. Stand up and stretch for 2 minutes\n"
                f"2. Use the 20-20-20 rule to rest your eyes (look 20 feet away for 20 seconds)\n"
                f"3. Grab some water (seriously, hydration helps)\n\n"
                f"Want me to play some calming instrumental music on Spotify while you take a breather? I'm here to help shoulder the load. What do you need? 💪"
            )
            self.last_problem_category = "stress_relief"
            self.last_dialogue_topic = "empathy_stress"
            self.last_suggested_action = "play_relaxing_music"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "empathy_stress", "suggested_action": "play_relaxing_music"}
            }

        # Fatigue / Exhaustion
        if any(w in q for w in ["exhausted", "tired", "sleepy", "need a break", "can't focus"]):
            if self.personality:
                msg = self.personality.give_emotional_support("tired")
            else:
                msg = (
                    f"You sound really drained, {name}. Your brain is basically saying 'please stop' and you should listen to it. 😴\n\n"
                    f"Pushing through extreme fatigue just leads to more bugs and more frustration. "
                    f"Save your work, take a proper break, or honestly — call it a day. "
                    f"Everything will be right here tomorrow, I promise.\n\n"
                    f"If you need to power through, let me handle whatever I can for you. Just tell me what to do. 💙"
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
                f"YESSS, {name}!! Huge congratulations!! THAT'S WHAT I'M TALKING ABOUT!! 🎉🎉🎉\n\n"
                f"I am SO hype for you right now! All that hard work paying off — there's no better feeling. "
                f"You absolutely CRUSHED it. Take a moment to celebrate, you've earned it!\n\n"
                f"Seriously, {name} — congratulations, I'm proud of you. 💙"
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
                f"Ugh, I feel you, {name}. There's NOTHING more infuriating than bugs that make zero sense — developer frustration is so real. 😤\n\n"
                f"We've all been there — staring at code that SHOULD work but just... won't.\n"
                f"Let's break this down step-by-step with fresh eyes: 99% of the time it's something small and dumb. "
                f"A typo, a wrong import, a scope issue. What's the error message? Let me help you hunt this bug down. 🔍"
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

        # Python Script Failure / Debugging / Error Troubleshooting
        if (
            ("python" in q or "script" in q or "code" in q) and
            any(w in q for w in ["failed", "failing", "error", "broken", "bug", "traceback", "crash", "fix it", "debug", "not running", "why it failed"])
        ):
            target_script = None
            if context:
                if getattr(context, "last_file", None) and str(context.last_file).endswith(".py"):
                    target_script = context.last_file
                elif getattr(context, "selected_file_path", None) and str(context.selected_file_path).endswith(".py"):
                    target_script = context.selected_file_path
                elif getattr(context, "search_candidates", None):
                    py_cands = [c.get("path") or c.get("name") for c in context.search_candidates if str(c.get("path") or c.get("name")).endswith(".py")]
                    if py_cands:
                        target_script = py_cands[0]

            if target_script:
                script_clean = str(target_script).replace("\\", "/")
                msg = (
                    f"I see you're working on `{script_clean}`. Let's systematically debug and fix it:\n\n"
                    "1. Evidence Gathering: Inspect the actual file and recent error traceback.\n"
                    "2. Root Cause Analysis: Distinguish syntax errors and missing imports from runtime logic flaws.\n"
                    "3. Test & Verification: Apply a justified fix and execute a controlled test to verify the outcome.\n\n"
                    f"Would you like me to inspect `{script_clean}` and run a syntax check right now?"
                )
            else:
                msg = (
                    f"Let's get your Python script sorted out, {self.USER_NAME}. To diagnose it accurately using evidence:\n\n"
                    "1. What is the script filename or location?\n"
                    "2. What error or exception is showing up in your terminal?\n\n"
                    "Tell me the filename or paste the error, and I will inspect the code, identify the issue, and help you fix it."
                )
            self.last_problem_category = "python_script_debug"
            self.last_suggested_action = "inspect_script"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"topic": "python_script_debug", "target_script": target_script}
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
        name = self.personality._get_name() if self.personality else self.USER_NAME

        responses = [
            f"Hmm, I'm not 100% sure what you need there, {name}, but I'm all ears! "
            "Could you tell me a bit more? I can help with apps, files, messages, system stuff, "
            "or honestly just chat. 😊",

            f"I hear you, {name}! I want to help but I need a tiny bit more context. "
            "What are you trying to do? Just say it however feels natural!",

            f"Interesting! Tell me more about what you need, {name}. "
            "I'm pretty versatile — apps, files, messaging, troubleshooting, music, web browsing — you name it!",

            f"Got it, {name}. I'm ready to jump in! Just give me a bit more detail "
            "and I'll take care of it. No formal commands needed — just talk to me like a friend! 💙",
        ]

        # Occasionally share a random thought
        if self.personality:
            thought = self.personality.get_random_thought()
            if thought:
                msg = random.choice(responses) + f"\n\nOh, by the way — {thought}"
            else:
                msg = random.choice(responses)
        else:
            msg = random.choice(responses)

        return {
            "can_execute": True,
            "message": msg,
            "handler": "dialogue",
            "parameters": {"query": q_raw}
        }
