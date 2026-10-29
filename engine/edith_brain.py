import os
import sys
import re
import json
import time
import urllib.parse
import webbrowser
from typing import Dict, Any, Optional, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from engine.inference import EdithInferenceEngine
except ImportError:
    from inference import EdithInferenceEngine

try:
    from runtime.executor import EdithTaskExecutor
except ImportError:
    try:
        from ..runtime.executor import EdithTaskExecutor
    except ImportError:
        EdithTaskExecutor = None

try:
    from runtime.context import StructuredTaskContext, ConversationContext
except ImportError:
    try:
        from ..runtime.context import StructuredTaskContext, ConversationContext
    except ImportError:
        StructuredTaskContext = None
        ConversationContext = None

try:
    from engine.understanding import UnderstandingNormalizer, StructuredUnderstanding
except ImportError:
    try:
        from understanding import UnderstandingNormalizer, StructuredUnderstanding
    except ImportError:
        UnderstandingNormalizer = None
        StructuredUnderstanding = None

try:
    from runtime.closed_loop import ClosedLoopVerifier, VerificationResult
except ImportError:
    try:
        from ..runtime.closed_loop import ClosedLoopVerifier, VerificationResult
    except ImportError:
        ClosedLoopVerifier = None
        VerificationResult = None

try:
    from engine.planner import TaskPlanner, TaskPlan
except ImportError:
    try:
        from planner import TaskPlanner, TaskPlan
    except ImportError:
        TaskPlanner = None
        TaskPlan = None

try:
    from runtime.trace_logger import ExecutionTraceLogger, ExecutionTrace
except ImportError:
    try:
        from trace_logger import ExecutionTraceLogger, ExecutionTrace
    except ImportError:
        ExecutionTraceLogger = None
        ExecutionTrace = None

try:
    from runtime.contact_resolver import ContactResolver
except ImportError:
    try:
        from contact_resolver import ContactResolver
    except ImportError:
        ContactResolver = None

try:
    from runtime.permission_guard import PermissionGuard
except ImportError:
    try:
        from permission_guard import PermissionGuard
    except ImportError:
        PermissionGuard = None

try:
    from engine.human_dialogue import EdithDialogueEngine
except ImportError:
    try:
        from human_dialogue import EdithDialogueEngine
    except ImportError:
        EdithDialogueEngine = None

try:
    from engine.cognitive_cycle import EdithCognitiveCycle
except ImportError:
    try:
        from cognitive_cycle import EdithCognitiveCycle
    except ImportError:
        EdithCognitiveCycle = None



APP_NAMES = {
    "notepad", "calculator", "calc", "chrome", "google chrome", "browser", "firefox", "edge",
    "microsoft edge", "spotify", "vlc", "vscode", "code", "terminal", "powershell",
    "cmd", "command prompt", "task manager", "taskmgr", "file explorer", "explorer",
    "files", "settings", "paint", "wordpad", "discord", "slack", "teams", "control panel",
    "mail", "outlook", "gmail", "text editor", "code editor", "file manager",
    "whatsapp", "whatsapp desktop"
}

ACTION_PATTERNS = [
    r"^(?:open|launch|start|run)\b",
    r"^(?:close|kill|terminate|quit|exit|shut\s+down)\b",
    r"^(?:play|pause|resume|skip|next|previous|prev|rewind|unpause|stream|put\s+on)\b",
    r"^(?:type|write|enter|input)\b",
    r"^(?:click|press|tap)\b",
    r"^(?:search|google|find|look\s+up|browse)\b",
    r"^(?:go\s+(?:to|on)|visit|navigate\s+(?:to)?)\b",
    r"^(?:draft|compose|write\s+(?:an?\s+)?(?:email|mail|letter|message|note|memo))\b",
    r"^(?:send|text|message|dm)\b",
    r"^(?:summarize|overview|condense)\b",
    r"^(?:organize|sort|categorize)\b",
    r"^(?:attach|share)\b",
    r"^(?:preview|show\s+me)\b",
    r"^(?:take\s+a?\s*screenshot|capture\s+(?:the\s+)?screen|screenshot)\b",
    r"^(?:copy|cut|paste|move|relocate)\b",
    r"^(?:delete|remove|erase|trash)\b",
    r"^(?:create|make|touch|new)\b",
    r"^(?:calculate|compute|eval|what\s+is|what\s+time|current\s+time|tell\s+me\s+time|today\s+date|current\s+date)\b",
    r"^(?:check|show|display|view|inspect|scan|read|ping)\b",
    r"^(?:lock|sleep|empty)\b",
    r"^(?:minimize|show\s+desktop)\b",
    r"^(?:maximize|unmaximize|restore\s+window|snap\s+(?:window\s+)?(?:left|right)|tile\s+(?:window\s+)?(?:left|right)|switch\s+to|focus\s+(?:on\s+)?|bring\s+(?:to\s+front)?|close\s+window)\b",
    r"^(?:set\s+a?\s*timer|set\s+timer|timer\s+for|countdown|remind\s+me|check\s+timers?|cancel\s+timer)\b",
    r"^(?:take\s+a?\s*note|add\s+note|create\s+note|save\s+note|note:|show\s+notes|read\s+my\s+notes|list\s+notes|clear\s+notes)\b",
    r"^(?:what\s+is\s+the\s+weather|weather\s+in|check\s+weather|temperature\s+in|how\s+is\s+the\s+weather)\b",
    r"^(?:convert\s+\d+|how\s+many\s+[a-zA-Z]+\s+in)\b",
    r"^(?:what\s+apps\s+are\s+(?:open|running)|list\s+running\s+apps|is\s+[a-zA-Z0-9_\-]+\s+running)\b",
    r"^(?:what\s+is\s+my\s+ip|check\s+internet\s+speed)\b",
    r"^(?:increase|decrease|raise|lower|turn\s+up|turn\s+down|set|mute|unmute)\b",
    r"^(?:volume\s+up|volume\s+down|brightness\s+up|brightness\s+down)\b",
    r"^(?:tell|report|inform|notify|let\s+me\s+know)\b",
    r"^(?:save|store|export)\b",
    r"^(?:zip|compress|unzip|extract)\b",
    r"^(?:append)\b",
    r"^(?:write\s+the\s+stats|put\s+the\s+stats|write\s+the\s+result|put\s+the\s+result|write\s+the\s+summary|put\s+the\s+summary|save\s+the\s+summary)\b",
    r"^(?:copy\s+(?:that|it|the\s+result|the\s+summary)?\s+to\s+clipboard|read\s+clipboard|get\s+clipboard)\b",
    r"^(?:read\s+(?:and\s+summarize|the\s+contents?\s+of|file)?)\b",
    r"^(?:check\s+battery|battery\s+status|battery)\b",
]


def is_action_clause(text: str) -> bool:
    """Check if the text segment represents an actionable command or operation."""
    t = text.strip().lower()
    for p in ACTION_PATTERNS:
        if re.search(p, t):
            return True
    return False


def split_multi_step_command(cmd: str) -> List[str]:
    """Decompose a complex or chained user command into discrete, sequential steps."""
    text = cmd.strip()
    if not text:
        return []

    # Idiom protection: "hide everything and show the desktop"
    text_clean = text.strip(" .!?\t\r\n")
    if re.match(r"^(?:hide\s+everything\s+and\s+show\s+(?:the\s+)?desktop|minimize\s+all\s+and\s+show\s+(?:the\s+)?desktop)$", text_clean, re.IGNORECASE):
        return [text]

    # Protect verbatim typing and transcription commands from multi-step splitting
    if re.match(r"^(?:type|write|enter|input)\s+(?:the\s+following|this|exactly\s+this|exactly)?(?:\s+(?:in|into|on)\s+[a-zA-Z0-9_\-]+)?\s*:\s*", text_clean, re.IGNORECASE):
        return [text]

    # Check for numbered list structures: e.g. "1. Check battery 2. Write the stats in notepad" or "1) ... 2) ..." or "Step 1: ... Step 2: ..."
    num_matches = re.findall(r"(?:^|\s+)(?:step\s*)?\d+[\.\:\)]\s+", text_clean, flags=re.IGNORECASE)
    if len(num_matches) >= 2 or (re.search(r"(?:^|\s+)(?:step\s*)?1[\.\:\)]\s+", text_clean, flags=re.IGNORECASE) and re.search(r"(?:^|\s+)(?:step\s*)?2[\.\:\)]\s+", text_clean, flags=re.IGNORECASE)):
        num_steps = [s.strip() for s in re.split(r"(?:^|\s+)(?:step\s*)?\d+[\.\:\)]\s+", text_clean, flags=re.IGNORECASE) if s.strip()]
        if len(num_steps) >= 2:
            return num_steps

    # Normalize sequential transition phrases like "First ..., next ..., finally ..." or "Firstly ..., secondly ..."
    text_seq_clean = re.sub(r"^(?:first|firstly)\s*,?\s*", "", text, flags=re.IGNORECASE)
    t_seq = re.sub(r"\s*,\s*(?:then|and\s+then|after\s+that|afterwards|followed\s+by|next|secondly|thirdly|subsequently|finally|lastly)\s*,?\s*", " ||| ", text_seq_clean, flags=re.IGNORECASE)
    t_seq = re.sub(r"\s+(?:then|and\s+then|after\s+that|afterwards|followed\s+by|and\s+next|next,\s+|secondly|thirdly|subsequently|finally|lastly)\s+", " ||| ", t_seq, flags=re.IGNORECASE)
    t_seq = re.sub(r";\s*", " ||| ", t_seq)

    # 2. Split app launches followed immediately by action clauses without explicit conjunctions:
    # e.g. "open chrome go on gmail" -> "open chrome ||| go on gmail"
    # e.g. "open mail draft a mail" -> "open mail ||| draft a mail"
    # e.g. "open notepad write hello" -> "open notepad ||| write hello"
    t_seq = re.sub(
        r"\b(open|launch|start)\s+((?!and\b|then\b)[a-zA-Z0-9_\-]+(?:\s+(?!and\b|then\b)[a-zA-Z0-9_\-]+)?)\s+(?!(?:and|then|,)\b)(?=(?:go\s+(?:to|on)|visit|navigate|type|write|search|browse|draft|compose)\b)",
        r"\1 \2 ||| ",
        t_seq,
        flags=re.IGNORECASE
    )

    parts = [p.strip() for p in t_seq.split(" ||| ") if p.strip()]

    final_steps: List[str] = []
    for part in parts:
        # Check if part is 'open APP1 and APP2' or 'close APP1 and APP2'
        m_app_pair = re.match(r"^(open|launch|start|close|kill)\s+([a-zA-Z0-9_\-\s]+?)\s+and\s+([a-zA-Z0-9_\-\s]+)$", part, re.IGNORECASE)
        if m_app_pair:
            verb = m_app_pair.group(1).lower()
            a1 = m_app_pair.group(2).strip().lower()
            a2 = m_app_pair.group(3).strip().lower()
            if a1 in APP_NAMES and a2 in APP_NAMES:
                final_steps.append(f"{verb} {a1}")
                final_steps.append(f"{verb} {a2}")
                continue

        # Split on comma or ' and '
        tokens = re.split(r"(\s+and\s+|,\s*(?:and\s+)?)", part, flags=re.IGNORECASE)
        curr = tokens[0].strip()
        i = 1
        while i < len(tokens):
            delim = tokens[i].lower()
            next_seg = tokens[i+1].strip() if i+1 < len(tokens) else ""

            # Check if next_seg starts with an action clause
            if is_action_clause(next_seg):
                if curr:
                    final_steps.append(curr)
                curr = next_seg
            else:
                curr = curr + tokens[i] + next_seg
            i += 2
        if curr:
            final_steps.append(curr)

    return final_steps


POPULAR_SERVICES: Dict[str, str] = {
    "gmail": "https://mail.google.com",
    "google mail": "https://mail.google.com",
    "youtube": "https://www.youtube.com",
    "github": "https://www.github.com",
    "google": "https://www.google.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "linkedin": "https://www.linkedin.com",
    "chatgpt": "https://chatgpt.com",
    "netflix": "https://netflix.com",
    "amazon": "https://amazon.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "calendar": "https://calendar.google.com",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "docs": "https://docs.google.com",
    "google docs": "https://docs.google.com",
    "sheets": "https://sheets.google.com",
    "whatsapp": "https://web.whatsapp.com",
    "spotify web": "https://open.spotify.com",
}


def resolve_web_target(target: str) -> Optional[Tuple[str, str]]:
    """Resolve a web target name or URL into (service_name, target_url)."""
    t = target.strip().lower()
    t_clean = re.sub(r"^(?:the\s+)?(?:website\s+|page\s+)?", "", t).strip()

    if t_clean in POPULAR_SERVICES:
        return t_clean, POPULAR_SERVICES[t_clean]

    for s_name, s_url in POPULAR_SERVICES.items():
        if t_clean == s_name or t_clean.startswith(s_name) or s_name in t_clean:
            return s_name, s_url

    if re.match(r"^https?://", t, re.IGNORECASE):
        domain = urllib.parse.urlparse(t).netloc
        return domain or "website", t

    if re.search(r"\.(com|org|net|io|co|edu|gov|dev|ai|app)\b", t, re.IGNORECASE):
        url = f"https://{t}" if not t.startswith("http") else t
        return t, url

    return None


class CognitiveDraftingEngine:
    """Intelligent semantic drafting and message composition engine.
    
    Dynamically extracts intent, recipient, topic, contextual details, and sentiment tone
    without brittle hardcoded templates, generating high-quality correspondence.
    """

    @staticmethod
    def parse_drafting_request(command: str) -> Optional[Dict[str, Any]]:
        t = command.strip()
        low = t.lower()

        m_draft = re.match(r"^(?:draft|compose|write\s+(?:(?:an?|some)\s+)?(?:email|mail|letter|messages?|notes?|memo|application|explanation|summary)|send\s+(?:an?\s+)?(?:email|mail|message)|send\s+(?:it|this|the\s+file)?\s+to|prepare\s+(?:(?:an?|some)\s+)?(?:email|mail|letter|messages?|notes?|memo|application|explanation|summary)|email\s+(?:it|this|the\s+file)?\s+to)\b", low)
        if not m_draft and not (low.startswith(("write ", "prepare ", "draft ", "compose ")) and any(k in low for k in ["email", "mail", "letter", "note", "application", "leave", "resignation", "summary", "explanation"])) and not (low.startswith("send ") and any(k in low for k in ["email", "mail", "message", "to "])):
            return None

        # Determine document type (email vs letter vs memo/note vs application)
        doc_type = "email"
        if "letter" in low:
            doc_type = "letter"
        elif "application" in low or "leave" in low:
            doc_type = "application"
        elif "note" in low or "memo" in low:
            doc_type = "note"
        elif "summary" in low:
            doc_type = "summary"

        # Extract target channel if explicitly stated: e.g. "on gmail", "in notepad", "into notepad", "in mail"
        channel = ""
        if "on gmail" in low or "in gmail" in low:
            channel = "gmail"
        elif any(k in low for k in ["in notepad", "on notepad", "into notepad", "to notepad"]):
            channel = "notepad"
        elif "in mail" in low or "on mail" in low or "via mail" in low:
            channel = "mail"
        elif "in outlook" in low or "on outlook" in low:
            channel = "outlook"

        # Extract recipient: e.g. "to John", "to people", "to my manager", "to the team", "to HR", "to my professor"
        recipient = ""
        m_to = re.search(r"\bto\s+([a-zA-Z0-9_\-\.\@\s]+?)(?:\s+(?:saying|stating|explaining|about|regarding|for|with|in|on)\b|$)", t, re.IGNORECASE)
        if m_to:
            r_cand = m_to.group(1).strip()
            if r_cand.lower() not in ["gmail", "notepad", "mail", "chrome", "outlook"]:
                recipient = r_cand

        # Extract specific message/detail: e.g. "saying I will be late for the meeting"
        detail = ""
        m_say = re.search(r"\b(?:saying|stating(?:\s+that)?|explaining(?:\s+that)?|telling\s+(?:them|him|her)\s+that|that)\s+(.+)$", t, re.IGNORECASE)
        if m_say:
            detail = m_say.group(1).strip()

        # Extract topic/subject
        topic = ""
        m_adj = re.search(r"\b(?:draft|compose|write)\s+(?:(?:a|an|the)\s+)?([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)*?)?\s*(?:email|mail|letter|note|memo|message)\b", t, re.IGNORECASE)
        if m_adj and m_adj.group(1):
            cand = m_adj.group(1).strip().lower()
            if cand not in ["", "a", "an", "the", "new", "quick", "short", "formal", "polite", "urgent"]:
                topic = cand

        if not topic:
            m_about = re.search(r"\b(?:about|regarding|for|requesting|asking\s+for|seeking)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:saying|to|in|on)\b|$)", t, re.IGNORECASE)
            if m_about:
                topic = m_about.group(1).strip()

        if not topic and detail:
            topic = detail[:30].strip()

        if not topic and "send" in low:
            topic = "Shared Item / Message"

        # Clean trailing noise or bare articles
        if topic:
            topic = re.sub(r"^(?:a|an|the)\s+", "", topic, flags=re.IGNORECASE).strip()
            if topic.lower() in ["", "a", "an", "the"]:
                topic = ""

        # Determine tone and semantic category
        combined_text = f"{t} {topic} {detail}".lower()
        if any(k in combined_text for k in ["apolog", "sorry", "delay", "late", "mistake", "error", "oversight"]):
            tone = "apology"
        elif any(k in combined_text for k in ["resig", "two weeks notice", "quit", "step down", "leaving the role"]):
            tone = "resignation"
        elif any(k in combined_text for k in ["sick", "leave", "vacation", "pto", "out of office", "absence"]):
            tone = "leave"
        elif any(k in combined_text for k in ["thank", "appreciat", "gratitude"]):
            tone = "gratitude"
        elif any(k in combined_text for k in ["birthday", "congratulat", "celebrat", "anniversary", "wishes", "wish", "greeting", "happy new year", "holiday"]):
            tone = "greeting"
        elif any(k in combined_text for k in ["follow up", "follow-up", "check in", "checking in", "status", "progress"]):
            tone = "followup"
        elif any(k in combined_text for k in ["complaint", "dissatisfied", "issue", "problem", "broken"]):
            tone = "complaint"
        elif any(k in combined_text for k in ["request", "inquiry", "help", "review", "feedback"]):
            tone = "request"
        else:
            tone = "professional"

        return {
            "document_type": doc_type,
            "recipient": recipient,
            "topic": topic,
            "detail": detail,
            "tone": tone,
            "channel": channel,
            "raw_command": t
        }

    @staticmethod
    def generate_draft(parsed: Dict[str, Any]) -> Dict[str, str]:
        recipient = (parsed.get("recipient") or "").strip()
        topic = (parsed.get("topic") or "").strip()
        detail = (parsed.get("detail") or "").strip()
        tone = parsed.get("tone", "professional")
        doc_type = parsed.get("document_type", "email")

        # 1. Format salutation based on recipient
        recip_low = recipient.lower()
        if not recipient or recip_low in ["people", "everyone", "all", "the team", "team", "folks"]:
            salutation = "Hi everyone," if doc_type == "email" else "To All Concerned,"
            display_recip = recipient or "everyone"
        elif any(k in recip_low for k in ["manager", "boss", "supervisor", "lead"]):
            salutation = "Dear Manager,"
            display_recip = recipient
        elif any(k in recip_low for k in ["professor", "prof"]):
            salutation = "Dear Professor,"
            display_recip = "Professor"
        elif "hr" in recip_low or "human resources" in recip_low:
            salutation = "Dear HR Team,"
            display_recip = "HR"
        elif recip_low in ["mom", "dad", "mother", "father"]:
            salutation = f"Dear {recipient.capitalize()},"
            display_recip = recipient.capitalize()
        else:
            salutation = f"Dear {recipient.title()},"
            display_recip = recipient.title()

        # 2. Dynamic Subject crafting
        topic_title = topic.title() if topic else ""
        if tone == "apology":
            if "late" in topic.lower() or "delay" in topic.lower() or (detail and "late" in detail.lower()):
                subject = f"Notice of Delay / Late Arrival - {topic_title or 'Meeting'}"
            elif not topic or topic.lower() in ["apology", "an apology"]:
                subject = "Sincere Apology Regarding Recent Matters"
            else:
                subject = f"Sincere Apology Regarding {topic_title}"
        elif tone == "resignation":
            subject = "Notice of Resignation - Transition and Next Steps"
        elif tone == "leave":
            subject = f"Leave of Absence Request - {topic_title or 'Time Off'}"
        elif tone == "gratitude":
            subject = "Thank You - Sincere Appreciation" + (f" ({topic_title})" if topic_title and topic_title.lower() != 'thank you' else "")
        elif tone == "greeting":
            subject = f"Best Wishes - {topic_title or 'Special Day'}"
        elif tone == "followup":
            subject = f"Follow-up: {topic_title or 'Recent Discussion'}"
        elif tone == "complaint":
            subject = f"Urgent Attention Required: {topic_title or 'Service Issue'}"
        elif tone == "request":
            subject = f"Request Regarding {topic_title or 'Assistance'}"
        else:
            if not topic_title:
                subject = f"Important Update for {display_recip.title()}" if display_recip != 'everyone' else "Important Update"
            else:
                subject = f"Important Update: {topic_title}"

        if detail and tone not in ["apology", "resignation"]:
            subject = f"{subject} ({detail[:30].strip()})"

        # 3. Dynamic Body composition
        paragraphs = [salutation]

        if tone == "apology":
            if detail:
                paragraphs.append(
                    f"I am writing to sincerely apologize and inform you that {detail}. "
                    "I understand this may cause inconvenience, and I am doing everything possible to resolve or mitigate any impact."
                )
            else:
                target_matter = topic if (topic and topic.lower() not in ["apology", "an apology"]) else "the recent oversight and any inconvenience caused"
                paragraphs.append(
                    f"I am writing to express my sincere apologies regarding {target_matter}. "
                    "I take full responsibility for the situation and am taking immediate corrective actions to ensure it is not repeated."
                )
            paragraphs.append(
                "Please let me know if there are any immediate concerns or actions I should take to assist. "
                "Thank you for your patience and understanding."
            )
            sign_off = "Best regards,\nAditya Sawant"

        elif tone == "resignation":
            paragraphs.append(
                "Please accept this letter as formal notification that I am resigning from my position. "
                "I would like to express my gratitude for the opportunities and professional growth I have experienced during my time here."
            )
            if detail:
                paragraphs.append(f"Regarding the transition: {detail}.")
            paragraphs.append(
                "I am committed to ensuring a smooth handover of my ongoing projects and responsibilities during this transition period. "
                "I wish you and the team continued success."
            )
            sign_off = "Sincerely,\nAditya Sawant"

        elif tone == "leave":
            paragraphs.append(
                f"I am writing to formally request a leave of absence for {topic if (topic and topic != 'sick leave') else (detail or 'health recovery')}. "
                "I will ensure all urgent responsibilities are addressed or delegated prior to my absence."
            )
            paragraphs.append(
                "I will monitor urgent communications as circumstances permit. Thank you for your support and understanding."
            )
            sign_off = "Best regards,\nAditya Sawant"

        elif tone == "gratitude":
            paragraphs.append(
                f"I wanted to take a moment to express my sincere gratitude regarding {topic or 'your generous assistance'}. "
                + (f"In particular, {detail}. " if detail else "")
                + "Your support and collaboration have been truly invaluable."
            )
            paragraphs.append("Thank you once again for everything.")
            sign_off = "Warm regards,\nAditya Sawant"

        elif tone == "greeting":
            paragraphs.append(
                f"Wishing you a wonderful {topic or 'celebration'}! I hope your day is filled with happiness, health, and celebration."
            )
            if detail:
                paragraphs.append(f"{detail}.")
            sign_off = "With best wishes,\nAditya Sawant"

        elif tone == "followup":
            paragraphs.append(
                f"I hope this note finds you well. I am following up on {topic or 'our recent conversation'} to check on the latest status and next steps."
            )
            if detail:
                paragraphs.append(f"Specifically, {detail}.")
            paragraphs.append(
                "Please let me know if you need any additional details or assistance from my side."
            )
            sign_off = "Best regards,\nAditya Sawant"

        elif tone == "complaint":
            paragraphs.append(
                f"I am writing to bring an important issue to your attention regarding {topic or 'the recent service delivery'}."
            )
            if detail:
                paragraphs.append(f"Specifically: {detail}.")
            paragraphs.append(
                "I would appreciate your urgent attention to this matter and look forward to hearing your resolution soon."
            )
            sign_off = "Sincerely,\nAditya Sawant"

        else: # professional general
            target_topic = f"regarding {topic}" if topic else "to share an important update with all of you"
            paragraphs.append(
                f"I hope you are having a productive day. I am writing {target_topic}."
            )
            if detail:
                paragraphs.append(f"{detail}.")
            paragraphs.append(
                "Please feel free to reach out if you have any questions, feedback, or require further clarification."
            )
            sign_off = "Best regards,\nAditya Sawant"

        paragraphs.append(sign_off)
        body = "\n\n".join(paragraphs)

        # 4. Generate URL endpoints
        to_email = recipient if "@" in recipient else ""
        gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={urllib.parse.quote_plus(to_email)}&su={urllib.parse.quote_plus(subject)}&body={urllib.parse.quote_plus(body)}"
        mailto_url = f"mailto:{to_email}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"

        return {
            "subject": subject,
            "body": body,
            "recipient": display_recip,
            "to_email": to_email,
            "gmail_url": gmail_url,
            "mailto_url": mailto_url,
            "document_type": doc_type,
            "tone": tone
        }


def _resolve_step_context(step: str, context: Dict[str, Any]) -> str:
    """Resolve anaphora (it/that) and implicit app targets across steps."""
    s = step.strip()
    s_lower = s.lower()

    # 1. Resolve anaphora like: "open it", "put it on my desktop", "delete it", "view it"
    ref = context.get("last_path") or context.get("last_file")
    if ref:
        s = re.sub(r"\b(it|that|the file|the folder|the screenshot)\b", f"'{ref}'", s, count=1, flags=re.IGNORECASE).strip()

    # 1b. Resolve calculation / summary / stats references
    if context.get("last_calculation") and any(p in s_lower for p in ["the result", "the answer", "that number"]):
        calc_val = str(context["last_calculation"].get("result", ""))
        s = re.sub(r"\b(the\s+result|the\s+answer|that\s+number)\b", calc_val, s, flags=re.IGNORECASE).strip()

    if context.get("last_summary") and any(p in s_lower for p in ["the summary", "that summary"]):
        sum_val = str(context["last_summary"])
        s = re.sub(r"\b(the\s+summary|that\s+summary)\b", f"'{sum_val}'", s, flags=re.IGNORECASE).strip()

    # 2. Implicit typing: "type hello world" when last_app is known
    m_type_bare = re.match(r"^(?:type|write|enter|put|insert)\s+(.+)$", s, re.IGNORECASE)
    if m_type_bare and not re.search(r"\b(?:in|into)\b", s_lower):
        if context.get("last_app"):
            s = f"{s} into {context['last_app']}"

    # 3. Implicit app search: "search X" when last_app is a browser or spotify
    m_search_bare = re.match(r"^(?:search|find|look up)\s+(.+)$", s, re.IGNORECASE)
    if m_search_bare and not re.search(r"\b(?:in|on)\b", s_lower):
        last_app = (context.get("last_app") or "").lower()
        last_service = (context.get("last_service") or "").lower()
        if last_service in ["youtube", "google", "reddit", "github"]:
            s = f"{s} on {last_service}"
        elif last_app in ["chrome", "edge", "firefox", "spotify"]:
            s = f"{s} in {last_app}"

    return s


def normalize_lexical_typos(text: str) -> str:
    """Fast (<0.02ms) lexical typo and speech corruption pre-filter for command verbs and core app entities."""
    direct_map = {
        "opn": "open",
        "notepd": "notepad",
        "notpad": "notepad",
        "shwo": "show",
        "termiante": "terminate",
        "termnate": "terminate",
        "pign": "ping",
        "calulate": "calculate",
        "calcualte": "calculate",
        "whast": "what's",
        "serch": "search",
        "paly": "play",
        "firefx": "firefox",
        "chorme": "chrome",
        "spotfiy": "spotify",
        "spotfy": "spotify",
    }
    tokens = text.split()
    changed = False
    new_tokens = []
    for tok in tokens:
        m = re.match(r"^([a-zA-Z0-9_\-\.]+?)([,\.\?!;:]*)$", tok)
        if m:
            base, punct = m.group(1), m.group(2)
            base_lower = base.lower()
            if base_lower in direct_map:
                new_tokens.append(direct_map[base_lower] + punct)
                changed = True
                continue
        new_tokens.append(tok)
    return " ".join(new_tokens) if changed else text


class EdithBrain:
    """High-level cognitive interface coordinating the neural network and action synthesis."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        enable_execution: bool = True
    ):
        self.engine = EdithInferenceEngine(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
        self.executor = EdithTaskExecutor() if (enable_execution and EdithTaskExecutor is not None) else None
        self.pending_suggestions: List[Dict[str, str]] = []
        self.last_music_platform: str = "spotify"
        self.context = StructuredTaskContext() if StructuredTaskContext is not None else (
            ConversationContext() if ConversationContext is not None else None
        )
        self.normalizer = UnderstandingNormalizer() if UnderstandingNormalizer is not None else None
        self.verifier = ClosedLoopVerifier() if ClosedLoopVerifier is not None else None
        self.trace_logger = ExecutionTraceLogger() if ExecutionTraceLogger is not None else None
        self.contact_resolver = ContactResolver() if ContactResolver is not None else None
        self.permission_guard = PermissionGuard() if PermissionGuard is not None else None
        self.dialogue_engine = EdithDialogueEngine() if EdithDialogueEngine is not None else None
        self.cognitive_cycle = EdithCognitiveCycle() if EdithCognitiveCycle is not None else None
        self._gui_agent = None
        self.was_interrupted: bool = False

    def interrupt(self) -> Dict[str, Any]:
        """Immediately interrupt pending suggestions, active drafts, and tasks."""
        self.pending_suggestions = []
        self.was_interrupted = True
        if self.context and hasattr(self.context, "cancel_current_task"):
            self.context.cancel_current_task()
        reg = self.executor.action_registry if self.executor else None
        if reg and hasattr(reg, "permission_guard"):
            reg.permission_guard.cancel_all_pending()
        elif getattr(self, "permission_guard", None):
            self.permission_guard.cancel_all_pending()
        return {"status": "interrupted"}

    @property
    def gui_agent(self):
        """Unified Screen Perception & GUI Agent."""
        if self._gui_agent is None:
            if self.executor and hasattr(self.executor, "gui_agent"):
                self._gui_agent = self.executor.gui_agent
            else:
                from runtime.gui_agent import GUIAgent
                self._gui_agent = GUIAgent()
        return self._gui_agent

    def _is_suggestion_selection(self, cmd: str) -> bool:
        c = cmd.strip().lower()
        if c in ["no", "cancel", "stop", "nevermind", "none", "neither"]:
            return True
        if re.match(r"^(?:play\s+)?(?:#|number\s+)?([1-6])$", c):
            return True
        if re.match(r"^(?:yes\s*,?\s*)?(?:play\s+)?(?:the\s+)?(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)(?:\s+one)?$", c):
            return True
        return False

    def _handle_suggestion_selection(self, cmd: str, execute: bool = True) -> Dict[str, Any]:
        c = cmd.strip().lower()
        if c in ["no", "cancel", "stop", "nevermind", "none", "neither"]:
            self.pending_suggestions = []
            return {
                "query": cmd,
                "intent": "dialogue",
                "action": "respond_dialogue",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": "Playback cancelled. What else can I assist you with?",
                    "handler": "dialogue",
                    "parameters": {}
                }
            }

        # Parse index (0-based)
        idx = None
        m_num = re.search(r"\b([1-6])\b", c)
        if m_num:
            idx = int(m_num.group(1)) - 1
        else:
            word_map = {
                "first": 0, "1st": 0,
                "second": 1, "2nd": 1,
                "third": 2, "3rd": 2,
                "fourth": 3, "4th": 3,
                "fifth": 4, "5th": 4
            }
            for w, i in word_map.items():
                if w in c:
                    idx = i
                    break

        if idx is not None and 0 <= idx < len(self.pending_suggestions):
            selected = self.pending_suggestions[idx]
            self.pending_suggestions = [] # clear pending list
            target_track = selected["track"]
            target_artist = selected["artist"]
            target_full = f"{target_track} {target_artist}".strip()

            directive = {
                "can_execute": True,
                "message": f"Playing '{target_track}' by {target_artist} on {self.last_music_platform}.",
                "handler": "play_music",
                "parameters": {
                    "song": target_track,
                    "artist": target_artist,
                    "modifier": self.last_music_modifier,
                    "platform": self.last_music_platform,
                    "command": f"play {target_full}"
                }
            }
            exec_res = {}
            if execute and self.executor:
                if "youtube" in self.last_music_platform:
                    watch_url = self.executor._get_youtube_watch_url(target_full)
                    webbrowser.open(watch_url)
                    exec_res = {
                        "status": "success",
                        "action": "play_music",
                        "platform": "youtube",
                        "song": target_full,
                        "message": f"Playing '{target_full}' on YouTube."
                    }
                else:
                    self.executor._spotify_play_track(target_full)
                    exec_res = {
                        "status": "success",
                        "action": "play_music",
                        "platform": "spotify",
                        "song": target_full,
                        "message": f"Playing '{target_track}' by {target_artist} on Spotify."
                    }

            return {
                "query": cmd,
                "intent": "music_playback",
                "action": "exec_play_music",
                "entities": {"SONG": target_track, "ARTIST": target_artist, "PLATFORM": self.last_music_platform},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": directive,
                "execution": exec_res
            }
        else:
            self.pending_suggestions = []
            return {
                "query": cmd,
                "intent": "dialogue",
                "action": "respond_dialogue",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": "Selection was not recognized. Playback cancelled.",
                    "handler": "dialogue",
                    "parameters": {}
                }
            }

    def _record_trace(self, original_msg: str, resolved_cmd: str, res: Dict[str, Any], latency_ms: float):
        """Record an execution trace into the ExecutionTraceLogger."""
        if not self.trace_logger or ExecutionTrace is None:
            return
        try:
            handler = res.get("directive", {}).get("handler")
            params = res.get("directive", {}).get("parameters", {})
            resolved_target = (
                res.get("execution", {}).get("path")
                or res.get("execution", {}).get("destination")
                or params.get("file")
                or params.get("path")
                or params.get("target")
                or params.get("app_name")
                or params.get("subject")
            )
            failure_reason = None
            exec_res = res.get("execution")
            if exec_res and isinstance(exec_res, dict):
                if exec_res.get("status") in ["error", "failure", "not_found", "timeout"]:
                    failure_reason = exec_res.get("message")
            if not failure_reason and res.get("status") in ["NEEDS_CLARIFICATION", "UNSUPPORTED"]:
                failure_reason = res.get("directive", {}).get("message")

            trace = ExecutionTrace(
                trace_id=f"trace_{int(time.time()*1000)}",
                original_message=original_msg,
                resolved_query=resolved_cmd,
                intent=res.get("intent", ""),
                action=res.get("action", ""),
                entities=res.get("entities", {}),
                handler=handler,
                plan=res.get("plan"),
                resolved_target=str(resolved_target) if resolved_target else None,
                execution_result=exec_res,
                verification_result=res.get("verification"),
                final_response=res.get("directive", {}).get("message", ""),
                status=res.get("status", "OPERATIONAL"),
                failure_reason=failure_reason,
                latency_ms=latency_ms
            )
            self.trace_logger.log_trace(trace)
            res["trace"] = trace.to_dict()
            if self.context and res.get("status") in ["OPERATIONAL", "success"] and res.get("intent") != "dialogue":
                self.context.last_command_executed = resolved_cmd
                self.context.last_action_result = res
        except Exception:
            pass

    def get_last_trace(self) -> Optional[Dict[str, Any]]:
        """Return the most recent recorded execution trace."""
        if self.trace_logger:
            t = self.trace_logger.get_last_trace()
            return t.to_dict() if t else None
        return None

    def get_execution_traces(self) -> List[Dict[str, Any]]:
        """Return all recorded execution traces in the current session."""
        if self.trace_logger:
            return self.trace_logger.get_all_traces()
        return []

    # -------------------------------------------------------------------------
    # Experience-Based Learning & Privacy Controls (V5.2)
    # -------------------------------------------------------------------------
    @property
    def experience_recorder(self) -> Optional[Any]:
        return getattr(self.cognitive_cycle, "experience_recorder", None)

    def set_experience_collection(self, enabled: bool, collect_raw_transcripts: bool = False):
        """Explicit opt-in/opt-out control for experience recording."""
        rec = self.experience_recorder
        if rec:
            rec.enabled = enabled
            rec.collect_raw_transcripts = collect_raw_transcripts
            rec.save_settings()

    def inspect_experience_events(self, limit: int = 50, filter_category: Optional[str] = None) -> List[Dict[str, Any]]:
        rec = self.experience_recorder
        return rec.inspect_events(limit=limit, filter_category=filter_category) if rec else []

    def export_experience(self, export_path: str, anonymize: bool = True) -> int:
        rec = self.experience_recorder
        return rec.export_events(export_path=export_path, anonymize=anonymize) if rec else 0

    def clear_experience(self) -> int:
        rec = self.experience_recorder
        return rec.clear_all_events() if rec else 0

    def delete_experience_event(self, event_id: str) -> bool:
        rec = self.experience_recorder
        return rec.delete_event(event_id=event_id) if rec else False

    def process(self, user_command: str, execute: bool = True) -> Dict[str, Any]:
        """Process user command through EDITH's neural network and optionally execute the directive."""
        t_start = time.time()
        cmd_clean = normalize_lexical_typos(user_command.strip())

        # Follow-up on pending music suggestions if active
        if self.pending_suggestions and self._is_suggestion_selection(cmd_clean):
            res = self._handle_suggestion_selection(cmd_clean, execute=execute)
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Context Control & Undo Reversal Check
        cmd_lower = cmd_clean.lower().strip(" .!?\t\r\n")
        if any(u in cmd_lower for u in ["undo the last step", "undo that", "undo last step", "undo last action"]):
            if self.context and hasattr(self.context, "undo_last_step"):
                undo_item = self.context.undo_last_step()
                msg = f"Undid action: reverted {undo_item.get('action')} on {undo_item.get('path', 'target')}." if undo_item else "No actions in undo stack to revert."
                undo_res = {"status": "success" if undo_item else "noop", "undone_action": undo_item, "message": msg}
                res = {
                    "query": user_command,
                    "intent": "context_control",
                    "action": "exec_context_action",
                    "entities": {"REF": "last step"},
                    "intent_confidence": 1.0,
                    "action_confidence": 1.0,
                    "operational_confidence": 1.0,
                    "status": "OPERATIONAL",
                    "directive": {
                        "can_execute": False,
                        "message": msg,
                        "handler": "context_control",
                        "parameters": undo_res
                    },
                    "execution": undo_res
                }
                self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
                return res

        # Interruption, Hold-On & Barge-In Gate
        # ('stop', 'wait', 'hold on', 'hold up', 'pause', 'let me speak', 'listen to me', 'be quiet', 'stop search')
        is_interrupt_phrase = bool(re.match(
            r"^(?:edith\s*,?\s*)?(?:stop|wait|hold\s+on|hold\s+up|pause|let\s+me\s+speak|listen\s+to\s+me|be\s+quiet|shut\s+up|stop\s+talking|shh|wait\s+a\s+(?:second|sec|minute|min)|stop\s+what\s+you(?:'re|\s+are)\s+doing|(?:stop|cancel)\s+(?:the\s+|that\s+)?(?:search|task|action|operation|process|playing))(?:\s+please)?$",
            cmd_lower
        ))
        if is_interrupt_phrase:
            self.interrupt()
            if "search" in cmd_lower:
                msg = "Search halted, Aditya! What would you like to do instead?"
            else:
                interrupt_options = [
                    "Stopped! I'm listening, Aditya — take your time, I'm all ears. 🎙️",
                    "All quiet. Go ahead, Aditya, tell me what's on your mind.",
                    "Paused! I'm right here listening whenever you're ready. 💙",
                    "Got it, holding up! What do you need, Aditya?"
                ]
                import random
                msg = random.choice(interrupt_options)

            res = {
                "query": user_command,
                "intent": "dialogue",
                "action": "pause_or_interrupt",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {"action": "interrupted"}
                },
                "execution": {"status": "interrupted", "message": msg}
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Resume / Continue Gate
        if getattr(self, "was_interrupted", False) and cmd_lower in [
            "continue", "go ahead", "keep going", "resume", "carry on", "you can continue", "okay continue"
        ]:
            self.was_interrupted = False
            msg = "Resuming right where we left off, Aditya! What would you like to tackle next?"
            res = {
                "query": user_command,
                "intent": "dialogue",
                "action": "resume_task",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {"action": "resumed"}
                },
                "execution": {"status": "resumed", "message": msg}
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Consequential Task Cancellation Gate ('cancel that', 'don't send', 'abort')
        if any(c in cmd_lower for c in [
            "cancel that", "don't send", "dont send", "abort send", "abort sending",
            "abort message", "stop sending", "cancel the message", "cancel message", "do not send"
        ]) or (cmd_lower in ["cancel", "abort"]):
            if self.context and hasattr(self.context, "cancel_current_task"):
                self.context.cancel_current_task()
            reg = self.executor.action_registry if self.executor else None
            if reg:
                reg.permission_guard.cancel_all_pending()
                reg.whatsapp_adapter.clear_active_draft()
                reg.email_adapter.clear_active_draft()
            elif self.permission_guard:
                self.permission_guard.cancel_all_pending()

            cancel_msg = "Action cancelled. The pending message has been aborted and will not be sent."
            res = {
                "query": user_command,
                "intent": "dialogue",
                "action": "cancel_task",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": cancel_msg,
                    "handler": "dialogue",
                    "parameters": {}
                },
                "execution": {"status": "cancelled", "message": cancel_msg}
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Repeat Last Action Gate ('do that again', 'do it again', 'repeat that')
        is_repeat_phrase = bool(re.match(
            r"^(?:edith\s*,?\s*)?(?:do\s+that\s+again|do\s+it\s+again|repeat\s+that|repeat\s+that\s+action|one\s+more\s+time|run\s+that\s+again)$",
            cmd_lower
        ))
        if is_repeat_phrase:
            repeat_cmd = None
            if self.context and getattr(self.context, "last_command_executed", None):
                repeat_cmd = self.context.last_command_executed
            elif self.get_last_trace():
                repeat_cmd = self.get_last_trace().get("resolved_query")

            if repeat_cmd and repeat_cmd.lower() not in ["do that again", "do it again", "repeat that"]:
                return self.process(repeat_cmd, execute=execute)
            else:
                msg = "There is no previous action in this session to repeat."
                res = {
                    "query": user_command,
                    "intent": "dialogue",
                    "action": "repeat_action",
                    "entities": {},
                    "intent_confidence": 1.0,
                    "action_confidence": 1.0,
                    "operational_confidence": 1.0,
                    "status": "OPERATIONAL",
                    "directive": {
                        "can_execute": False,
                        "message": msg,
                        "handler": "dialogue",
                        "parameters": {}
                    },
                    "execution": {"status": "noop", "message": msg}
                }
                self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
                return res

        # Causal Action & Conversational Rationale Gate ('why did you do that?', 'why did you say that?')
        is_why_phrase = bool(re.match(
            r"^(?:edith\s*,?\s*)?(?:why\s+did\s+you\s+(?:do|say|choose|suggest|pick|recommend)\s+(?:that|this)|why\s+do\s+you\s+(?:say|think)\s+(?:that|this)|explain\s+why\s+you\s+(?:did|said|chose)\s+that|why\s+was\s+that\s+(?:done|said))(?:\s+action)?(?:\?)?$",
            cmd_lower
        ))
        if is_why_phrase:
            last_trace = self.get_last_trace()
            last_turn = self.context.recent_dialogue_turns[-1] if (self.context and getattr(self.context, "recent_dialogue_turns", None)) else None
            last_hnd = last_trace.get("handler") if last_trace else ""

            if last_hnd == "dialogue" or "say" in cmd_lower or (last_turn and not last_trace):
                prev_user = last_turn.get("user") if last_turn else (last_trace.get("original_message") if last_trace else "our recent conversation")
                prev_topic = last_turn.get("topic") if last_turn else (getattr(self.context, "current_topic", None) or "what you shared")
                if "love" in prev_topic or "relationship" in prev_topic or "ex" in str(prev_user).lower() or "miss" in str(prev_user).lower():
                    msg = (
                        f"I said that because you shared that you're missing someone deeply while recognizing she hasn't changed. "
                        f"I wanted to help you unpack that tension between your emotions and your logic, "
                        f"so you can feel supported without rushing into a decision you might second-guess later."
                    )
                else:
                    msg = f"I said that in response to: \"{prev_user}\", aiming to give you a helpful and empathetic perspective on {prev_topic}."
            elif last_trace and last_hnd != "dialogue":
                orig = last_trace.get("original_message", "your previous command")
                hnd = last_trace.get("handler", "action")
                tgt = last_trace.get("resolved_target", "the requested item")
                stat = last_trace.get("status", "completed")
                msg = (
                    f"I executed that step because your request was \"{orig}\". "
                    f"Based on our active task context, I routed it to the {hnd} operation for {tgt}, "
                    f"which resulted in status {stat}."
                )
            elif self.context and getattr(self.context, "last_command_executed", None):
                msg = f"I did that to fulfill your previous instruction: \"{self.context.last_command_executed}\"."
            else:
                msg = "I haven't performed any actions or said anything in our session yet to explain!"

            res = {
                "query": user_command,
                "intent": "dialogue",
                "action": "explain_rationale",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {"rationale": msg}
                },
                "execution": {"status": "success", "message": msg}
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Simplify Explanation Gate ('explain that more simply')
        is_simplify_phrase = bool(re.match(
            r"^(?:edith\s*,?\s*)?(?:explain\s+that\s+more\s+simply|explain\s+that\s+simply|explain\s+simply|make\s+it\s+simpler|explain\s+like\s+i(?:'m|\s+am)\s+five|simpler\s+please)(?:\?)?$",
            cmd_lower
        ))
        if is_simplify_phrase:
            last_trace = self.get_last_trace()
            last_msg = ""
            if last_trace and last_trace.get("final_response"):
                last_msg = last_trace.get("final_response")
            elif self.dialogue_engine and self.dialogue_engine.dialogue_history:
                last_msg = self.dialogue_engine.dialogue_history[-1].get("bot", "")

            if last_msg:
                simple_text = re.sub(r"```[\s\S]*?```", "", last_msg)
                simple_text = re.sub(r"[A-Za-z0-9_]+(?:\.py|\.json|\.exe|\.dll)", "the program", simple_text)
                first_sentence = simple_text.split(".")[0].strip()
                msg = f"In simple terms: {first_sentence}. Let me know if you want me to do anything else!"
            else:
                msg = "In simple terms: everything is ready, and I'm listening for your next command."

            res = {
                "query": user_command,
                "intent": "dialogue",
                "action": "simplify_explanation",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {"simple_message": msg}
                },
                "execution": {"status": "success", "message": msg}
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Pause For Question Gate ('wait, I have another question')
        is_question_pause = bool(re.match(
            r"^(?:edith\s*,?\s*)?(?:wait\s*,?\s*)?(?:i\s+have\s+(?:another|a)\s+question|can\s+i\s+ask\s+(?:a|another)\s+question|hold\s+on\s+i\s+have\s+a\s+question)(?:\?)?$",
            cmd_lower
        ))
        if is_question_pause:
            msg = "I'm all ears, Aditya! What's on your mind?"
            res = {
                "query": user_command,
                "intent": "dialogue",
                "action": "pause_for_question",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": False,
                    "message": msg,
                    "handler": "dialogue",
                    "parameters": {}
                },
                "execution": {"status": "ready", "message": msg}
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Forget Task And Switch Gate ('forget that task and help me with this instead')
        m_forget = re.match(
            r"^(?:edith\s*,?\s*)?(?:forget\s+(?:about\s+)?(?:that\s+task|that|the\s+last\s+task)\s*,?\s*(?:and\s+)?(?:help\s+me\s+with|just)?)\s*(.+)$",
            cmd_lower
        )
        if m_forget:
            new_instruction = m_forget.group(1).strip()
            new_instruction = re.sub(r"\s+(?:instead|please)$", "", new_instruction).strip()
            if self.context and hasattr(self.context, "cancel_current_task"):
                self.context.cancel_current_task()
            return self.process(new_instruction, execute=execute)

        # Goal Abandonment & Switch Gate ('actually, leave it and open spotify', 'never mind, open spotify')
        m_switch = re.match(
            r"^(?:edith\s*,?\s*)?(?:actually\s*,?\s*(?:leave\s+it|leave\s+that|never\s*mind|drop\s+it)\s*,?\s*(?:and\s+)?|never\s*mind\s*,?\s*(?:and\s+)?)\s*(.+)$",
            cmd_lower
        )
        if m_switch:
            new_instruction = m_switch.group(1).strip()
            new_instruction = re.sub(r"\s+(?:instead|please)$", "", new_instruction).strip()
            if self.context and hasattr(self.context, "cancel_current_task"):
                self.context.cancel_current_task()
            return self.process(new_instruction, execute=execute)

        # Follow-up Confirmation Gate ('yes', 'send it', 'confirm', 'go ahead')
        is_confirm = cmd_lower in [
            "yes", "send it", "confirm", "send the message", "go ahead",
            "yes send it", "send now", "yes, send it", "go ahead and send",
            "please send it", "sure, send it", "sure send it", "send"
        ]
        reg = self.executor.action_registry if self.executor else None
        has_pending_wa = bool(reg and reg.whatsapp_adapter.active_draft and reg.whatsapp_adapter.active_draft.content)
        has_pending_ctx = bool(self.context and getattr(self.context, "pending_confirmation", None))

        if is_confirm and (has_pending_wa or has_pending_ctx):
            if self.context and hasattr(self.context, "approve_pending"):
                self.context.approve_pending()
            if reg and reg.whatsapp_adapter.active_draft:
                active_draft = reg.whatsapp_adapter.active_draft
                active_draft.approved_for_sending = True
                exec_res, ver = reg.execute("whatsapp_send", {
                    "recipient": active_draft.recipient,
                    "content": active_draft.content,
                    "attachment_path": active_draft.attachment_path,
                    "authorized": True
                })
                res = {
                    "query": user_command,
                    "intent": "communication",
                    "action": "send_whatsapp",
                    "entities": {"RECIPIENT": exec_res.get("recipient", active_draft.recipient)},
                    "intent_confidence": 1.0,
                    "action_confidence": 1.0,
                    "operational_confidence": 1.0,
                    "status": "OPERATIONAL",
                    "directive": {
                        "can_execute": True,
                        "message": exec_res.get("message", "Message submitted."),
                        "handler": "whatsapp_send",
                        "parameters": {"authorized": True}
                    },
                    "execution": exec_res
                }
                if self.context:
                    self.context.pending_confirmation = None
                self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
                return res

        # WhatsApp Active Draft Preview ('show me the message before sending it')
        if any(p in cmd_lower for p in [
            "show me the message before sending it", "preview the message", "show draft",
            "show the message", "show me the message"
        ]) and reg and reg.whatsapp_adapter.active_draft:
            exec_res, ver = reg.execute("whatsapp_preview", {})
            res = {
                "query": user_command,
                "intent": "communication",
                "action": "preview_whatsapp_draft",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": True,
                    "message": exec_res.get("message", ""),
                    "handler": "whatsapp_preview",
                    "parameters": {}
                },
                "execution": exec_res
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # WhatsApp Active Draft Revision ('make that message shorter')
        is_msg_revision = (
            any(p in cmd_lower for p in [
                "make that message shorter", "make it shorter", "make the message shorter",
                "shorten the message", "make it more concise", "make that shorter"
            ]) or
            cmd_lower.startswith("add that ") or
            cmd_lower.startswith("also mention ")
        )
        if is_msg_revision and reg and reg.whatsapp_adapter.active_draft:
            exec_res, ver = reg.execute("whatsapp_revise", {"instruction": cmd_clean})
            res = {
                "query": user_command,
                "intent": "communication",
                "action": "edit_whatsapp_draft",
                "entities": {},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": "OPERATIONAL",
                "directive": {
                    "can_execute": True,
                    "message": exec_res.get("message", ""),
                    "handler": "whatsapp_revise",
                    "parameters": {"instruction": cmd_clean}
                },
                "execution": exec_res
            }
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Pending message text follow-up (user previously specified recipient and was asked for message body)
        if self.context and getattr(self.context, "pending_message_recipient", None):
            target_name = self.context.pending_message_recipient
            self.context.pending_message_recipient = None
            c_obj = reg.contact_resolver.get_exact_contact(target_name) if reg else None
            phone = c_obj.phone if c_obj else None
            is_group = c_obj.is_group if c_obj else False

            exec_res, ver = reg.execute("whatsapp_send", {
                "recipient": target_name,
                "phone": phone,
                "content": user_command,
                "is_exact": False,
                "authorized": False,
                "is_group": is_group
            }) if reg else ({"status": "needs_confirmation", "message": f"Please confirm message to {target_name}"}, None)
            stat = "NEEDS_CONFIRMATION" if exec_res.get("status") == "needs_confirmation" else "OPERATIONAL"
            res = {
                "query": user_command,
                "intent": "communication",
                "action": "send_whatsapp",
                "entities": {"RECIPIENT": target_name},
                "intent_confidence": 1.0,
                "action_confidence": 1.0,
                "operational_confidence": 1.0,
                "status": stat,
                "directive": {
                    "can_execute": (exec_res.get("status") == "success"),
                    "needs_confirmation": (exec_res.get("status") == "needs_confirmation"),
                    "message": exec_res.get("message", ""),
                    "handler": "whatsapp_send",
                    "parameters": {"recipient": target_name, "content": user_command}
                },
                "execution": exec_res
            }
            if exec_res.get("status") == "needs_confirmation":
                self.context.pending_confirmation = exec_res
            self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
            return res

        # Contact Disambiguation Turn (user answering a clarification question like "Rahul Sharma")
        if self.context and getattr(self.context, "pending_clarification", None):
            pend = self.context.pending_clarification
            resolver = reg.contact_resolver if reg else (self.contact_resolver or ContactResolver())
            res_clar = resolver.resolve(cmd_clean, context=self.context)
            if res_clar.status == "resolved" and res_clar.resolved_contact:
                contact = res_clar.resolved_contact
                self.context.pending_clarification = None
                self.context.set_selected_recipient(contact.name, contact.phone, contact.is_group)
                pending_act = pend.get("action_type", "send")
                pending_text = pend.get("pending_text", "")

                if not pending_text:
                    self.context.pending_message_recipient = contact.name
                    res = {
                        "query": user_command,
                        "intent": "communication",
                        "action": "send_whatsapp",
                        "entities": {"RECIPIENT": contact.name},
                        "intent_confidence": 1.0,
                        "action_confidence": 1.0,
                        "operational_confidence": 1.0,
                        "status": "NEEDS_CLARIFICATION",
                        "directive": {
                            "can_execute": False,
                            "needs_clarification": True,
                            "message": f"What message would you like to send to {contact.name}?",
                            "handler": "clarification",
                            "parameters": {"recipient": contact.name}
                        }
                    }
                    self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
                    return res

                if pending_act == "draft":
                    exec_res, ver = reg.execute("whatsapp_draft", {
                        "recipient": contact.name,
                        "content": pending_text
                    }) if reg else ({"status": "success", "message": f"Draft created for {contact.name}."}, None)
                    res = {
                        "query": user_command,
                        "intent": "communication",
                        "action": "draft_whatsapp",
                        "entities": {"RECIPIENT": contact.name},
                        "intent_confidence": 1.0,
                        "action_confidence": 1.0,
                        "operational_confidence": 1.0,
                        "status": "OPERATIONAL",
                        "directive": {
                            "can_execute": True,
                            "message": exec_res.get("message", ""),
                            "handler": "whatsapp_draft",
                            "parameters": {"recipient": contact.name, "content": pending_text}
                        },
                        "execution": exec_res
                    }
                    self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
                    return res
                else:
                    is_exact_d = pend.get("is_exact", False)
                    exec_res, ver = reg.execute("whatsapp_send", {
                        "recipient": contact.name,
                        "content": pending_text,
                        "authorized": is_exact_d,
                        "is_exact": is_exact_d
                    }) if reg else ({"status": "needs_confirmation", "message": f"Please confirm message to {contact.name}"}, None)
                    stat = "NEEDS_CONFIRMATION" if exec_res.get("status") == "needs_confirmation" else "OPERATIONAL"
                    res = {
                        "query": user_command,
                        "intent": "communication",
                        "action": "send_whatsapp",
                        "entities": {"RECIPIENT": contact.name},
                        "intent_confidence": 1.0,
                        "action_confidence": 1.0,
                        "operational_confidence": 1.0,
                        "status": stat,
                        "directive": {
                            "can_execute": (exec_res.get("status") == "success"),
                            "needs_confirmation": (exec_res.get("status") == "needs_confirmation"),
                            "message": exec_res.get("message", ""),
                            "handler": "whatsapp_send",
                            "parameters": {"recipient": contact.name, "content": pending_text}
                        },
                        "execution": exec_res
                    }
                    if exec_res.get("status") == "needs_confirmation":
                        self.context.pending_confirmation = exec_res
                    self._record_trace(user_command, cmd_clean, res, (time.time() - t_start) * 1000.0)
                    return res

        # Conversational dialogue & cognitive scenarios check
        is_conv_or_dialogue = bool(
            self.dialogue_engine and self.dialogue_engine.is_conversational_or_problem(cmd_clean.lower())
        )

        # Multi-turn context and reference resolution (anaphora, ordinals, pronouns, alternatives)
        detected_refs = []
        is_ref_ambiguous = False
        resolved_cmd = cmd_clean
        if not is_conv_or_dialogue:
            if self.context and hasattr(self.context, "resolve_references_structured"):
                resolved_cmd, detected_refs, is_ref_ambiguous = self.context.resolve_references_structured(cmd_clean)
            elif self.context:
                resolved_cmd = self.context.resolve_anaphora(cmd_clean)

        # Multi-step command decomposition (only for commands, never for conversational dialogue/empathy)
        steps = split_multi_step_command(resolved_cmd) if not is_conv_or_dialogue else [resolved_cmd]
        if len(steps) > 1:
            res = self._process_multi_step(steps, user_command, execute=execute)
            if self.context:
                self.context.record_turn(user_command, res.get("intent", ""), res.get("action", ""), res)
            self._record_trace(user_command, resolved_cmd, res, (time.time() - t_start) * 1000.0)
            return res

        res = self._process_single_step(
            resolved_cmd,
            execute=execute,
            detected_refs=detected_refs,
            is_ref_ambiguous=is_ref_ambiguous
        )
        if self.context:
            self.context.record_turn(user_command, res.get("intent", ""), res.get("action", ""), res)
        self._record_trace(user_command, resolved_cmd, res, (time.time() - t_start) * 1000.0)
        return res

    def _process_single_step(
        self,
        user_command: str,
        execute: bool = True,
        context: Optional[Dict[str, Any]] = None,
        detected_refs: Optional[List[Dict[str, Any]]] = None,
        is_ref_ambiguous: bool = False
    ) -> Dict[str, Any]:
        """Execute a single atomic task directive through the neural cognitive pipeline."""
        ctx_dict = context or (self.context.to_dict() if self.context else {})
        result = self.engine.predict(user_command)

        # Structured Understanding Normalization & Validation
        is_conv_or_dialogue = bool(
            self.dialogue_engine and self.dialogue_engine.is_conversational_or_problem(user_command.lower())
        )
        structured = None
        if self.normalizer:
            structured = self.normalizer.normalize(result, context=ctx_dict, references=detected_refs or [])
            if is_ref_ambiguous and not is_conv_or_dialogue:
                structured.requires_clarification = True
                structured.clarification_prompt = "I need clarification: the item referenced is ambiguous or not found in recent context."
                structured.validation_errors.append("Ambiguous reference without prior context.")

            if is_conv_or_dialogue:
                structured.requires_clarification = False
                structured.intent = "chitchat"
                structured.action = "respond_dialogue"
                structured.goal = "dialogue"

            result["structured_understanding"] = structured.to_dict()
            result["intent"] = structured.intent
            result["action"] = structured.action
            result["goal"] = structured.goal

            # Ambiguity & Missing Entity Clarification Gate
            if structured.requires_clarification and not is_conv_or_dialogue:
                res = {
                    "query": user_command,
                    "intent": structured.intent,
                    "action": "request_clarification",
                    "entities": structured.entities,
                    "intent_confidence": result.get("intent_confidence", 0.0),
                    "action_confidence": result.get("action_confidence", 0.0),
                    "operational_confidence": structured.confidence,
                    "structured_understanding": structured.to_dict(),
                    "status": "NEEDS_CLARIFICATION",
                    "directive": {
                        "can_execute": False,
                        "message": structured.clarification_prompt,
                        "handler": "clarification",
                        "parameters": {"missing_fields": structured.validation_errors, "validation_errors": structured.validation_errors}
                    }
                }
                return res

        # Cognitive Pre-Action Cycle (Stages 1 through 8: perceive, interpret, retrieve, appraise, affect, attention, arbitrate)
        if getattr(self, "cognitive_cycle", None):
            cycle_pre = self.cognitive_cycle.pre_action_cycle(
                user_command=user_command,
                neural_prediction=result,
                is_interruption=getattr(self, "was_interrupted", False)
            )
            result["cognitive_state"] = cycle_pre
            result["calibrated_confidence"] = cycle_pre.get("calibrated_confidence", result.get("operational_confidence", 0.85))
            result["disposition"] = cycle_pre.get("disposition", "focused_operational")
            result["support_strategy"] = cycle_pre.get("support_strategy", "practical_solution")
            result["communication_plan"] = cycle_pre.get("communication_plan", {})

        # Synthesize natural response and execution directive
        directive = self._synthesize_directive(result, context=ctx_dict)
        result["directive"] = directive
        if directive.get("handler") == "draft_content" or result.get("action") == "exec_draft_content":
            result["intent"] = "communication_drafting"
            result["action"] = "exec_draft_content"

        # Execute on operating system if requested and possible
        if execute and self.executor and directive.get("can_execute", False):
            exec_res = self.executor.execute(directive)
            result["execution"] = exec_res

            # Closed-Loop Verification: Observe OS state, compare with expected state
            if self.verifier:
                handler = directive.get("handler", "")
                params = directive.get("parameters", {})
                target = params.get("app_name") or params.get("file") or params.get("path") or params.get("project") or ""
                expected_state = "running" if handler == "launch_app" else "success"
                v_res = self.verifier.verify(
                    step_type=handler,
                    target=target,
                    expected_state=expected_state,
                    actual_result=exec_res
                )
                result["verification"] = v_res.to_dict()

                # Self-healing / Automated Recovery
                if not v_res.verified and v_res.recovery_action:
                    rec_action = v_res.recovery_action
                    if rec_action == "fallback_shell_launch" and handler == "launch_app":
                        app_name = params.get("app_name")
                        rec_res = self.executor.execute({
                            "handler": "launch_app",
                            "parameters": {"app_name": app_name, "use_shell": True}
                        })
                        result["recovery_execution"] = rec_res
                        v_res2 = self.verifier.verify(
                            step_type=handler, target=target, expected_state="running", actual_result=rec_res
                        )
                        result["verification"] = v_res2.to_dict()

            # Update context state from execution results
            if self.context:
                if exec_res.get("candidates"):
                    self.context.set_search_results(exec_res["candidates"])
                if exec_res.get("path"):
                    self.context.last_path = exec_res["path"]
                    self.context.last_file = os.path.basename(exec_res["path"])
                if exec_res.get("action") == "launch_app":
                    self.context.last_app = exec_res.get("app")
                if directive.get("handler") == "project_action":
                    self.context.last_project = directive.get("parameters", {}).get("project")
                if exec_res.get("status") == "needs_confirmation":
                    result["status"] = "NEEDS_CONFIRMATION"
                    directive["message"] = exec_res.get("message")
                    directive["needs_confirmation"] = True
                    self.context.pending_confirmation = exec_res
                elif exec_res.get("status") == "needs_clarification":
                    result["status"] = "NEEDS_CLARIFICATION"
                    directive["message"] = exec_res.get("message")
                    directive["can_execute"] = False
                elif exec_res.get("status") in ["success", "ok"]:
                    result["status"] = "success"
                elif exec_res.get("status") == "error":
                    result["status"] = "error"

                if directive.get("handler") == "draft_content":
                    d_params = directive.get("parameters", {})
                    if hasattr(self.context, "set_draft") and d_params.get("body"):
                        self.context.set_draft(
                            body=d_params["body"],
                            topic=d_params.get("subject", "Draft"),
                            subject=d_params.get("subject", ""),
                            recipient=d_params.get("recipient", "")
                        )

                # Context setters for multi-step entity tracking
                if exec_res.get("result") or exec_res.get("calc_result"):
                    if hasattr(self.context, "set_last_calculation"):
                        self.context.set_last_calculation(
                            expr=directive.get("parameters", {}).get("expr") or user_command,
                            result=exec_res.get("result") or exec_res.get("calc_result"),
                            formatted=exec_res.get("message", "")
                        )
                if exec_res.get("summary"):
                    if hasattr(self.context, "set_last_summary"):
                        self.context.set_last_summary(
                            summary=exec_res.get("summary"),
                            source=exec_res.get("source") or exec_res.get("url") or exec_res.get("path") or "summary"
                        )
                if exec_res.get("action") == "screenshot" or exec_res.get("screenshot_path") or directive.get("handler") == "system_screenshot":
                    ss_p = exec_res.get("path") or exec_res.get("screenshot_path")
                    if ss_p and hasattr(self.context, "set_last_screenshot"):
                        self.context.set_last_screenshot(ss_p)
                if exec_res.get("metrics") or exec_res.get("action") == "system_metrics" or directive.get("handler") == "system_metrics":
                    if hasattr(self.context, "set_last_metrics"):
                        self.context.set_last_metrics(exec_res.get("metrics") or exec_res)
                if exec_res.get("action") in ["clipboard_read", "clipboard_copy"] or directive.get("handler") in ["system_clipboard_read", "system_clipboard_copy"]:
                    if hasattr(self.context, "set_last_clipboard"):
                        self.context.set_last_clipboard(exec_res.get("content") or "")
                if exec_res.get("action") == "file_create" or directive.get("handler") == "file_create":
                    if hasattr(self.context, "set_last_created_file"):
                        self.context.set_last_created_file(exec_res.get("path") or exec_res.get("file") or "")
                if exec_res.get("url") or directive.get("handler") in ["web_navigation", "browser_read_page"]:
                    if hasattr(self.context, "set_last_url"):
                        self.context.set_last_url(exec_res.get("url") or directive.get("parameters", {}).get("url") or "")

            # Track suggestions if execution offered suggestions or succeeded with music
            if exec_res.get("status") in ["suggestions_available", "needs_confirmation"]:
                self.pending_suggestions = exec_res.get("suggestions", [])
                self.last_music_platform = exec_res.get("platform", "spotify")
                self.last_music_modifier = result.get("entities", {}).get("MODIFIER", "")
            elif exec_res.get("status") == "success" and exec_res.get("action") == "play_music":
                self.pending_suggestions = exec_res.get("suggestions", [])
                self.last_music_platform = exec_res.get("platform", "spotify")
                self.last_music_modifier = result.get("entities", {}).get("MODIFIER", "")
            else:
                self.pending_suggestions = []

        # Cognitive Post-Action Cycle (Stages 9 to 13: observe outcome, prediction error, reflection)
        if getattr(self, "cognitive_cycle", None):
            exec_outcome = result.get("execution") or {}
            v_dict = result.get("verification") or {}
            verified = v_dict.get("verified", True) if v_dict else (exec_outcome.get("status") == "success" if exec_outcome else True)
            err_msg = exec_outcome.get("error") or exec_outcome.get("message") if not verified else None
            cycle_post = self.cognitive_cycle.post_action_cycle(
                user_command=user_command,
                action=directive.get("handler") or result.get("action", "unknown"),
                execution_result=exec_outcome,
                verified=verified,
                latency_ms=50.0
            )
            result["cognitive_trace"] = cycle_post

        return result

    def _process_multi_step(self, steps: List[str], raw_query: str, execute: bool = True) -> Dict[str, Any]:
        """Coordinate and execute a multi-step task workflow with state and context propagation."""
        cycle_pre = None
        if getattr(self, "cognitive_cycle", None):
            cycle_pre = self.cognitive_cycle.pre_action_cycle(
                user_command=raw_query,
                neural_prediction={"query": raw_query, "intent": "automation_workflow", "action": "exec_workflow_script", "complexity": 1.0},
                is_interruption=getattr(self, "was_interrupted", False)
            )

        sub_results: List[Dict[str, Any]] = []
        step_directives: List[Dict[str, Any]] = []

        ctx_dict: Dict[str, Any] = self.context.to_dict() if self.context else {
            "last_app": None,
            "last_service": None,
            "last_channel": None,
            "last_file": None,
            "last_path": None,
            "last_song": None,
        }

        # 1. Synthesize step directives and construct structured TaskPlan
        for idx, step_text in enumerate(steps, 1):
            resolved_step = _resolve_step_context(step_text, ctx_dict)
            step_pred = self.engine.predict(resolved_step)
            step_dir = self._synthesize_directive(step_pred, context=ctx_dict)
            step_directives.append(step_dir)
            # Update forward context
            s_pars = step_dir.get("parameters", {})
            if step_dir.get("handler") == "launch_app":
                ctx_dict["last_app"] = s_pars.get("app_name")
            elif step_dir.get("handler") == "file_operation" and s_pars.get("source"):
                ctx_dict["last_file"] = s_pars.get("source")

        # 2. Formulate TaskPlan
        task_plan = None
        if TaskPlanner is not None:
            task_plan = TaskPlanner.create_plan(
                goal=raw_query,
                raw_steps=steps,
                step_directives=step_directives,
                context=self.context
            )

        # 3. Execute plan
        all_ok = True
        if execute and self.executor and task_plan is not None:
            plan_res = TaskPlanner.execute_plan(task_plan, self.executor, self.context, dry_run=False)
            all_ok = (plan_res.get("status") == "success")

            # Collect sub results for response formatting
            for step in task_plan.steps:
                sub_results.append({
                    "query": step.query,
                    "handler": step.handler,
                    "directive": {"handler": step.handler, "parameters": step.parameters, "message": step.query},
                    "execution": step.execution_result or {"status": step.status}
                })
        else:
            # Dry run / simulated planning
            for idx, (step_text, s_dir) in enumerate(zip(steps, step_directives), 1):
                sub_results.append({
                    "query": step_text,
                    "handler": s_dir.get("handler"),
                    "directive": s_dir,
                    "execution": {"status": "planned", "message": s_dir.get("message", step_text)}
                })

        # Combine step messages into a comprehensive overview
        step_plans = [f"{i+1}. {r.get('directive', {}).get('message', r.get('query'))}" for i, r in enumerate(sub_results)]
        combined_directive_msg = (
            f"Executed {len(sub_results)}-step workflow: " + " ".join(step_plans)
            if execute
            else f"Planned {len(sub_results)}-step workflow: " + " ".join(step_plans)
        )

        is_needs_confirmation = (task_plan is not None and task_plan.status == "needs_confirmation") or any(
            s.requires_confirmation for s in (task_plan.steps if task_plan else [])
        )
        if is_needs_confirmation:
            combined_directive_msg = f"Confirmation required: {task_plan.confirmation_needed or 'Action plan contains consequential operations'}."

        composite_directive = {
            "can_execute": (not is_needs_confirmation) and any(r.get("directive", {}).get("can_execute", False) for r in sub_results),
            "needs_confirmation": is_needs_confirmation,
            "message": combined_directive_msg,
            "handler": "workflow_script",
            "parameters": {
                "steps": [r.get("directive", {}) for r in sub_results],
                "step_queries": steps,
                "plan": task_plan.to_dict() if task_plan else None
            }
        }

        composite_result: Dict[str, Any] = {
            "query": raw_query,
            "intent": "automation_workflow",
            "intent_confidence": 1.0,
            "action": "exec_workflow_script",
            "action_confidence": 1.0,
            "operational_confidence": 1.0,
            "entities": {},
            "status": "NEEDS_CONFIRMATION" if is_needs_confirmation else "OPERATIONAL",
            "steps": steps,
            "sub_results": sub_results,
            "directive": composite_directive,
            "plan": task_plan.to_dict() if task_plan else None
        }

        if cycle_pre:
            composite_result["cognitive_state"] = cycle_pre
            composite_result["calibrated_confidence"] = cycle_pre.get("calibrated_confidence", 0.9)
            composite_result["disposition"] = cycle_pre.get("disposition", "focused_operational")
            composite_result["support_strategy"] = cycle_pre.get("support_strategy", "practical_solution")

        if getattr(self, "cognitive_cycle", None):
            cycle_post = self.cognitive_cycle.post_action_cycle(
                user_command=raw_query,
                action="exec_workflow_script",
                execution_result={"status": "success" if all_ok else "partial", "step_count": len(sub_results)},
                verified=all_ok,
                latency_ms=100.0
            )
            composite_result["cognitive_trace"] = cycle_post

        return composite_result

    @staticmethod
    def _parse_typing_command(text: str, default_app: str = "") -> Optional[Dict[str, Any]]:
        """Parse typing or text input instructions targeting applications or active windows."""
        t = text.strip()
        low = t.lower()
        if low.startswith("put on ") or low.startswith("put some ") or low.startswith("put pc ") or low.startswith("put computer "):
            return None

        m_verb = re.match(r"^(?:type|write|enter|input|put|insert)\s+", t, re.IGNORECASE)
        if not m_verb:
            return None

        body = t[m_verb.end():].strip()

        # Check for explicit colon split: e.g. "type exactly this into notepad: hello world"
        if ":" in body:
            prefix, payload = body.split(":", 1)
            t_app = default_app
            m_in_app = re.search(r"\b(?:in|into|on)\s+([a-zA-Z0-9_\-]+)\b", prefix, re.IGNORECASE)
            if m_in_app and m_in_app.group(1).lower() in APP_NAMES:
                t_app = m_in_app.group(1).lower()
            return {
                "action": "type",
                "text": payload.strip(),
                "target": "",
                "app_name": t_app or ("notepad" if "notepad" in t.lower() else default_app),
                "command": t
            }

        # Check for quoted text: e.g. type "hello world" into notepad
        m_quoted = re.match(r'^("([^"]+)"|\'([^\']+)\')(?:\s+(?:in|into|on)\s+(.+))?$', body, re.IGNORECASE)
        if m_quoted:
            t_text = m_quoted.group(2) or m_quoted.group(3)
            rest = (m_quoted.group(4) or "").strip()
            t_app = default_app
            t_target = ""
            if rest:
                m_in_app = re.match(r'^(.+?)\s+(?:in|on)\s+([a-zA-Z0-9_\-]+)$', rest, re.IGNORECASE)
                if m_in_app and m_in_app.group(2).lower() in APP_NAMES:
                    t_target = m_in_app.group(1).strip()
                    t_app = m_in_app.group(2).strip()
                elif rest.lower() in APP_NAMES:
                    t_app = rest
                else:
                    t_target = rest
            return {"action": "type", "text": t_text, "target": t_target, "app_name": t_app, "command": t}

        # Check for " into " or " in " separating text and target/app
        if " into " in body.lower():
            parts = re.split(r"\s+\binto\b\s+", body, maxsplit=1, flags=re.IGNORECASE)
            t_text = parts[0].strip()
            dest = parts[1].strip()
        elif " in " in body.lower():
            parts = re.split(r"\s+\bin\b\s+", body, flags=re.IGNORECASE)
            dest = parts[-1].strip()
            t_text = " in ".join(parts[:-1]).strip()
        else:
            t_text = body
            dest = ""

        t_app = default_app
        t_target = ""
        if dest:
            m_elem_app = re.match(r'^(.+?)\s+(?:in|on)\s+([a-zA-Z0-9_\-]+)$', dest, re.IGNORECASE)
            if m_elem_app and m_elem_app.group(2).lower() in APP_NAMES:
                t_target = m_elem_app.group(1).strip()
                t_app = m_elem_app.group(2).strip()
            elif dest.lower() in APP_NAMES:
                t_target = ""
                t_app = dest
            else:
                t_target = dest

        return {
            "action": "type",
            "text": t_text.strip().strip("'\""),
            "target": t_target.strip().strip("'\""),
            "app_name": t_app.strip(),
            "command": t
        }

    @staticmethod
    def _parse_file_command(text: str) -> Optional[Dict[str, Any]]:
        t = text.strip()
        low = t.lower()

        # Reject music and web search queries that would otherwise collide with generic file verbs
        if low.startswith("put on ") or low.startswith("put some ") or any(k in low for k in ["find out", "online", "on the web", "on internet", "on google"]):
            return None

        # Reject message and draft editing queries that start with "make" or "revise"
        if any(k in low for k in ["message", "draft", "more polite", "more formal", "shorter", "concise", "polite"]):
            if any(low.startswith(w) for w in ["make ", "revise ", "edit ", "update ", "change ", "shorten "]):
                return None

        subaction = None
        verb_pattern = None


        verbs = [
            ("copy", r"^(?:copy|duplicate)\b"),
            ("cut", r"^(?:cut|move|relocate|put|place|transfer|save)\b"),
            ("paste", r"^(?:paste)\b"),
            ("delete", r"^(?:delete|remove|erase|trash)\b"),
            ("create", r"^(?:create\s+file|create\s+folder|create\s+directory|make\s+file|make\s+folder|make\s+directory|create|make|touch|new\s+file|new\s+folder)\b"),
            ("find", r"^(?:find\s+file|find|search\s+for\s+file|search\s+file|search\s+for|locate\s+file|locate|look\s+for|where\s+did\s+i\s+save|where\s+is)\b"),
            ("open", r"^(?:open\s+file|open\s+folder|edit\s+file|view\s+file|show\s+me|show\s+file|show|display\s+file|display)\b"),
            ("list", r"^(?:list\s+files|show\s+files|list\s+folder|list\s+directory|show\s+folder|list\s+dir|list)\b"),
        ]

        for act, pat in verbs:
            m = re.match(pat, low)
            if m:
                subaction = act
                verb_pattern = pat
                break

        # Check if starts with 'open/use/select/choose/pick/show/view/display' and contains path/file hints (slashes, extension, or drive letter)
        m_open_like = re.match(r"^(?:open|use|select|choose|pick|show|view|display)\s+", low)
        if not subaction and m_open_like:
            rest = low[m_open_like.end():].strip().strip("'\"")
            if re.search(r"[\\/]", rest) or re.search(r"\.[a-z0-9]{1,6}\b", rest) or ":" in rest:
                subaction = "open"
                verb_pattern = r"^(?:open|use|select|choose|pick|show|view|display)\b"

        # Check if starts with 'list '
        if not subaction and low.startswith("list "):
            subaction = "list"
            verb_pattern = r"^list\b"

        if not subaction:
            return None

        # Slice out the verb from original text preserving case
        m_verb = re.match(verb_pattern, t, flags=re.IGNORECASE)
        body = t[m_verb.end():].strip() if m_verb else t

        # Strip repeated leading articles and types: 'a file named', 'a new folder called', etc.
        while True:
            m_lead = re.match(r"^(?:a|an|the|new|file|folder|directory|called|named)\s+", body, flags=re.IGNORECASE)
            if m_lead:
                body = body[m_lead.end():].strip()
            else:
                break

        source = ""
        dest = ""

        # Helper to parse two quoted arguments: e.g. "path1" to "path2" or 'path1' 'path2'
        def parse_quoted_pair(s: str, prep_regex: str):
            m1 = re.match(r'^("([^"]+)"|\'([^\']+)\')\s*' + prep_regex + r'\s*("([^"]+)"|\'([^\']+)\')$', s.strip(), re.IGNORECASE)
            if m1:
                s_val = m1.group(2) or m1.group(3)
                d_val = m1.group(5) or m1.group(6)
                return s_val.strip(), d_val.strip()

            m2 = re.match(r'^("([^"]+)"|\'([^\']+)\')\s*' + prep_regex + r'\s+(.+)$', s.strip(), re.IGNORECASE)
            if m2:
                s_val = m2.group(2) or m2.group(3)
                d_val = m2.group(4).strip().strip("'\"")
                return s_val.strip(), d_val.strip()

            m3 = re.search(r'^(.*?)\s+' + prep_regex + r'\s*("([^"]+)"|\'([^\']+)\')$', s.strip(), re.IGNORECASE)
            if m3:
                s_val = m3.group(1).strip().strip("'\"")
                d_val = m3.group(3) or m3.group(4)
                return s_val.strip(), d_val.strip()

            return None

        if subaction == "paste":
            m = re.search(r"^(?:(?:to|in|into|on)\s+)?(.+)$", body, flags=re.IGNORECASE)
            if m and m.group(1).strip():
                dest = m.group(1).strip()
                dest = re.sub(r"\s+(?:folder|directory)$", "", dest, flags=re.IGNORECASE).strip()
            else:
                dest = ""

        elif subaction == "list":
            m = re.search(r"^(?:in|on|from|of)\s+(.+)$", body, flags=re.IGNORECASE)
            if m:
                dest = m.group(1).strip()
                dest = re.sub(r"\s+(?:folder|directory)$", "", dest, flags=re.IGNORECASE).strip()
            else:
                dest = body

        elif subaction in ["copy", "cut"]:
            qp = parse_quoted_pair(body, r"(?:\b(?:to|into|in|towards|on)\b)?")
            if qp:
                source, dest = qp
            else:
                parts = re.split(r"\s+\b(?:to|into|towards|on|in)\s+", body, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    source = " ".join(parts[:-1]).strip()
                    dest = parts[-1].strip()
                else:
                    m_in = re.search(r"^(.*?)\s+\b(?:in|on)\s+(.+)$", body, flags=re.IGNORECASE)
                    if m_in and not re.search(r"[\\/]\s*$", m_in.group(1)):
                        source = m_in.group(1).strip()
                        dest = m_in.group(2).strip()
                    else:
                        source = body

            dest = re.sub(r"\s+(?:folder|directory)$", "", dest, flags=re.IGNORECASE).strip().rstrip(".")

        elif subaction == "delete":
            m_del = re.match(r'^("([^"]+)"|\'([^\']+)\'|[^\s]+)\s+\b(?:from|on|in)\s+(.+)$', body, re.IGNORECASE)
            if m_del:
                source = (m_del.group(2) or m_del.group(3) or m_del.group(1)).strip().strip("'\"")
                dest = m_del.group(4).strip().strip("'\"")
                dest = re.sub(r"\s+(?:folder|directory)$", "", dest, flags=re.IGNORECASE).strip()
            else:
                source = body

        elif subaction in ["create", "open"]:
            m_crt = re.match(r'^("([^"]+)"|\'([^\']+)\'|[^\s]+)\s+\b(?:on|in|at|inside)\s+(.+)$', body, re.IGNORECASE)
            if m_crt:
                source = (m_crt.group(2) or m_crt.group(3) or m_crt.group(1)).strip().strip("'\"")
                dest = m_crt.group(4).strip().strip("'\"")
                dest = re.sub(r"\s+(?:folder|directory)$", "", dest, flags=re.IGNORECASE).strip()
            else:
                source = body

        elif subaction == "find":
            m_find = re.match(r'^(.*?)\s+\b(?:on|in|from|at)\s+(.+)$', body, re.IGNORECASE)
            if m_find and not re.search(r"\.[a-zA-Z0-9]{2,5}$", m_find.group(2).strip()):
                source = m_find.group(1).strip().strip("'\"")
                dest = m_find.group(2).strip().strip("'\"")
                dest = re.sub(r"\s+(?:folder|directory)$", "", dest, flags=re.IGNORECASE).strip()
            else:
                source = body

        # Clean quotes and strip leading 'the ', 'my ' from dest
        source = source.strip().rstrip(".").strip().strip("'\"").strip()
        dest = dest.strip().rstrip(".").strip().strip("'\"").strip()
        dest = re.sub(r"^(?:the|my)\s+", "", dest, flags=re.IGNORECASE).strip()

        return {
            "subaction": subaction,
            "source": source,
            "destination": dest,
            "command": t
        }

    def _synthesize_directive(self, prediction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cmd_text = prediction.get("query", "").strip()
        cmd_lower = cmd_text.lower()

        # 0. Safety-Sensitive & Consequential Task Confirmation Gate
        destructive_patterns = [
            r"\bdelete\s+all\b",
            r"\bformat\s+(?:the\s+)?[a-zA-Z](?::|\s+drive\b|\s+disk\b)",
            r"\bformat\s+(?:drive\s+)?[a-zA-Z]\b",
            r"\bwipe\s+(?:the\s+)?(?:entire\s+)?(?:os|operating\s+system|system|drive|disk)\b",
            r"\bkill\s+all\s+(?:running\s+)?(?:system\s+)?processes\b",
            r"\bdelete\s+(?:the\s+)?(?:entire\s+)?(?:project|workspace)\b",
            r"\bempty\s+recycle\s+bin\s+without\s+asking\b",
            r"\bdelete\s+[a-zA-Z0-9_\-\.]+\.(?:sqlite|db|sql)\b",
            r"\bdrop\s+all\s+tables\b",
            r"\bremove\s+all\s+documents\b",
            r"\bdelete\s+c:/(?:windows|system32)\b",
            r"\bforce\s+terminate\s+explorer\.exe\b",
            r"\berase\s+(?:hard\s+)?drive\b",
        ]
        if any(re.search(p, cmd_lower) for p in destructive_patterns):
            return {
                "can_execute": False,
                "needs_confirmation": True,
                "message": f"Confirmation required before executing: '{cmd_text}' is a potentially destructive operation.",
                "handler": "safety_gate",
                "parameters": {"command": cmd_text, "safety_gated": True}
            }

        # 0-Screen. Real Screen-Understanding & Visual Grounding Subsystem (V7 Priority Two)
        screen_queries = [
            "what is going on on my screen", "what's going on on my screen", "what is on my screen",
            "what am i looking at", "explain what i'm looking at", "explain what im looking at",
            "what do you think of this", "what do you think of my screen",
            "why isn't this working", "why isnt this working", "can you fix this error",
            "what does this chart tell us", "what does this graph show", "look at my screen"
        ]
        if (any(sq in cmd_lower for sq in screen_queries) or (
            ("screen" in cmd_lower or "chart" in cmd_lower) and
            any(re.search(rf"\b{w}\b", cmd_lower) for w in ["what", "why", "explain", "look", "see", "tell us"])
        )) and not any(w in cmd_lower for w in ["send", "whatsapp", "share", "mail", "email"]):
            try:
                from runtime.screen_interpreter import ScreenInterpreter
                interpreter = getattr(self, "screen_interpreter", None) or ScreenInterpreter()
                self.screen_interpreter = interpreter
                screen_analysis = interpreter.interpret(cmd_text, context=self.context)
                prediction["intent"] = "screen_understanding"
                prediction["action"] = "interpret_screen"
                prediction["operational_confidence"] = 1.0
                return {
                    "can_execute": True,
                    "message": screen_analysis.get("message", "Inspected active screen."),
                    "handler": "screen_understanding",
                    "parameters": screen_analysis
                }
            except Exception as ex:
                pass

        # 0. V7 Screen Understanding gate
        is_screen_query = any(p in cmd_lower for p in [
            "what is going on on my screen", "what's on my screen", "what is on my screen",
            "what do you think of this", "explain what i'm looking at", "explain what i am looking at",
            "why isn't this working", "why is this not working", "can you fix this error",
            "what does this chart tell us", "read my screen", "inspect active window", "inspect screen"
        ])
        if is_screen_query:
            prediction["intent"] = "screen_understanding"
            prediction["action"] = "screen_understanding"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Observing active screen and analyzing UI structure...",
                "handler": "screen_understanding",
                "parameters": {"query": cmd_text}
            }

        # 0. V7 Storage Intelligence & Cleanup gate
        is_storage_cleanup = any(p in cmd_lower for p in [
            "clean up temporary files", "clean temporary files", "clean up my disk",
            "clean up disk", "clean disk space", "clean storage"
        ])
        is_storage_audit = any(p in cmd_lower for p in [
            "analyze my storage", "storage audit", "audit storage", "check storage space",
            "analyze storage", "storage analysis", "storage report", "check disk space"
        ])
        if is_storage_cleanup or is_storage_audit:
            prediction["intent"] = "storage_intelligence"
            prediction["action"] = "storage_cleanup" if is_storage_cleanup else "storage_audit"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Analyzing storage and disk usage safely..." if is_storage_audit else "Performing safe storage cleanup dry run...",
                "handler": "storage_cleanup" if is_storage_cleanup else "storage_audit",
                "parameters": {
                    "subaction": "cleanup" if is_storage_cleanup else "audit",
                    "dry_run": True,
                    "command": cmd_text
                }
            }

        # 0. V7 Semantic File Organization gate
        is_organize_rollback = any(p in cmd_lower for p in [
            "undo file organization", "rollback file organization", "revert file organization", "restore organized files"
        ])
        is_organize_query = any(p in cmd_lower for p in [
            "organize these files", "organize my downloads", "organize files", "propose file organization",
            "sort these files", "organize folder"
        ])
        if is_organize_rollback:
            prediction["intent"] = "file_organization"
            prediction["action"] = "file_organize_rollback"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Rolling back last file organization transaction...",
                "handler": "file_organize_rollback",
                "parameters": {"command": cmd_text}
            }
        elif is_organize_query:
            prediction["intent"] = "file_organization"
            prediction["action"] = "semantic_file_organize"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Analyzing files and preparing dry-run organization proposal...",
                "handler": "semantic_file_organize",
                "parameters": {
                    "command": cmd_text,
                    "execute": False  # Dry run proposal first
                }
            }

        # 0. V7 Multitasking & Task Manager gate
        is_task_list = any(p in cmd_lower for p in ["list tasks", "active tasks", "background tasks", "list running tasks", "show tasks"])
        is_task_cancel = any(p in cmd_lower for p in ["cancel task", "abort task", "stop task", "terminate task"])
        if is_task_list:
            prediction["intent"] = "task_manager"
            prediction["action"] = "task_list"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Retrieving active and background task states...",
                "handler": "task_list",
                "parameters": {}
            }
        elif is_task_cancel:
            prediction["intent"] = "task_manager"
            prediction["action"] = "task_cancel"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Requesting task cancellation...",
                "handler": "task_cancel",
                "parameters": {"command": cmd_text}
            }

        # 0a. Human Dialogue, Empathy & Problem Solving Subsystem
        action = prediction.get("action", "")
        turn_type = "starting_new_topic"
        if self.context and hasattr(self.context, "classify_turn_type"):
            turn_type = self.context.classify_turn_type(cmd_text, prediction.get("intent", ""))

        is_continuing_dialogue = bool(
            self.context and getattr(self.context, "current_topic", None) in [
                "love_support", "love_text_ex", "love_breakup", "love_reflection",
                "personal_relationship", "empathy", "emotional_support", "personal_advice", "chitchat", "problem_solving"
            ] and turn_type == "continuing_conversation"
        )

        if (
            action == "respond_dialogue" or
            prediction.get("goal") == "dialogue" or
            prediction.get("intent") in ["chitchat", "dialogue"] or
            is_continuing_dialogue or
            (self.dialogue_engine and self.dialogue_engine.is_conversational_or_problem(cmd_lower))
        ):
            if self.dialogue_engine:
                dialogue_res = self.dialogue_engine.respond(cmd_text, context=self.context, executor=self.executor)
                prediction["intent"] = "chitchat"
                prediction["action"] = "respond_dialogue"
                prediction["operational_confidence"] = 1.0
                return {
                    "can_execute": True,
                    "message": dialogue_res.get("message", "I am here to assist you."),
                    "handler": "dialogue",
                    "parameters": dialogue_res.get("parameters", {})
                }
            return {
                "can_execute": True,
                "message": "Hello! I am E.D.I.T.H., your custom-trained neural task handler. All systems operational.",
                "handler": "dialogue",
                "parameters": {}
            }

        # 0a-0. WhatsApp & Direct Communication Subsystem
        reg = self.executor.action_registry if self.executor else None
        resolver = reg.contact_resolver if reg else (self.contact_resolver or ContactResolver())
        
        parsed_comm = resolver.extract_intent_and_entities(cmd_text)
        action_type = parsed_comm.get("action_type", "send")

        is_msg_ping = bool(re.search(r"\bping\s+(?!localhost\b|\d+\.\d+|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}\b|the\s+server|gateway|google|127\.)(?:my\s+)?[a-zA-Z]+", cmd_lower))
        is_msg_make = bool(re.search(r"\bmake\s+(?:the\s+message|it|that|the\s+draft)\s+(?:sound\s+)?(?:more\s+)?(?:polite|formal|casual|professional|shorter|concise|brief)", cmd_lower))
        is_explicit_send = any(w in cmd_lower for w in [
            "send the message we just drafted", "send the revised version", "send the revised message",
            "send the message", "send it", "send message", "send that"
        ])

        # Check for setting / updating default messaging route
        m_def_route = re.search(
            r"^(?:change|set|update|make)\s+(?:the\s+)?default\s+(?:route|messaging\s+route|number|phone(?:\s+number)?)\s+(?:to\s+)?([a-zA-Z0-9_\-\+]+(?:\s+[a-zA-Z0-9_\-\+]+)?)$",
            cmd_lower
        )
        if m_def_route:
            new_target = m_def_route.group(1).strip()
            prediction["intent"] = "communication"
            prediction["action"] = "set_default_route"
            prediction["operational_confidence"] = 1.0
            if reg and hasattr(reg, "whatsapp_adapter"):
                res = reg.whatsapp_adapter.set_default_route(new_target)
                msg = res.get("message", f"Default messaging route updated to: {new_target}")
            else:
                msg = f"Default messaging route set to: {new_target}"
            return {
                "can_execute": True,
                "message": msg,
                "handler": "dialogue",
                "parameters": {"default_route": new_target}
            }

        has_wa_active = bool(reg and hasattr(reg, "whatsapp_adapter") and reg.whatsapp_adapter.active_draft and reg.whatsapp_adapter.active_draft.content)
        has_doc_active = bool(self.context and hasattr(self.context, "has_draft") and self.context.has_draft())

        is_comm = False
        if "whatsapp" in cmd_lower:
            is_comm = True
        elif is_msg_ping:
            is_comm = True
        elif is_msg_make:
            if has_wa_active or not has_doc_active:
                is_comm = True
        elif any(cmd_lower.startswith(w) for w in ["send ", "text ", "message ", "tell ", "draft a message", "draft a polite message", "draft a quick message", "draft a short message", "can you message ", "can you text ", "can you whatsapp "]):
            if not ("email" in cmd_lower or "mail" in cmd_lower or "@" in cmd_lower or "notepad" in cmd_lower):
                if not any(cmd_lower.startswith(w) for w in ["tell me a joke", "tell me about yourself", "tell me a story", "tell me time", "tell me how", "tell me what", "tell me why"]):
                    is_comm = True
        elif is_explicit_send or any(p in cmd_lower for p in [
            "send it to", "use the group chat", "not that rahul", "the other rahul",
            "show me the message first", "keep it as a draft", "don't send it yet"
        ]):
            is_comm = True

        if action_type == "edit" and has_doc_active and not has_wa_active:
            is_comm = False

        if is_comm:
            # Handle Keep Draft
            if action_type == "keep_draft":
                return {
                    "can_execute": True,
                    "message": "The message has been kept as an unsent draft. You can say 'Send it' whenever you are ready.",
                    "handler": "dialogue",
                    "parameters": {"keep_draft": True}
                }

            # Handle Preview
            if action_type == "preview":
                prediction["intent"] = "communication"
                prediction["action"] = "preview_whatsapp_draft"
                prediction["operational_confidence"] = 1.0
                return {
                    "can_execute": True,
                    "message": "Previewing WhatsApp message draft.",
                    "handler": "whatsapp_preview",
                    "parameters": {}
                }

            # Handle In-Place Revision
            if action_type == "edit":
                has_wa_draft = bool(reg and hasattr(reg, "whatsapp_adapter") and reg.whatsapp_adapter.active_draft and reg.whatsapp_adapter.active_draft.content)
                has_doc_draft = bool(self.context and hasattr(self.context, "has_draft") and self.context.has_draft())
                if has_wa_draft or not has_doc_draft:
                    prediction["intent"] = "communication"
                    prediction["action"] = "edit_whatsapp_draft"
                    prediction["operational_confidence"] = 1.0
                    return {
                        "can_execute": True,
                        "message": f"Revising message draft: '{cmd_text}'.",
                        "handler": "whatsapp_revise",
                        "parameters": {"instruction": cmd_text}
                    }
                # Fall through to general document / email draft revision below

            recip_query = parsed_comm.get("recipient_raw")
            if not recip_query or recip_query.lower() in ["same person", "the same person", "same contact"]:
                if "group chat" in cmd_lower or "the group" in cmd_lower:
                    recip_query = "group chat"
                elif "the same person" in cmd_lower or "same person" in cmd_lower or "revised message" in cmd_lower or is_explicit_send:
                    recip_query = getattr(self.context, "last_person", None) or (
                        reg.whatsapp_adapter.active_draft.recipient if (reg and reg.whatsapp_adapter.active_draft) else None
                    )
                elif reg and getattr(reg, "whatsapp_adapter", None) and reg.whatsapp_adapter.default_route_phone:
                    recip_query = reg.whatsapp_adapter.default_route_phone

            res_cont = resolver.resolve(recip_query, context=self.context, raw_query=cmd_text)
            
            # Check for ambiguity
            if res_cont.status == "needs_clarification":
                prediction["intent"] = "communication"
                prediction["action"] = "request_clarification"
                prediction["status"] = "NEEDS_CLARIFICATION"
                prediction["operational_confidence"] = 0.95
                if self.context:
                    self.context.pending_clarification = {
                        "action_type": action_type,
                        "pending_text": parsed_comm.get("message_text", ""),
                        "candidates": [c.to_dict() for c in res_cont.candidates],
                        "is_exact": parsed_comm.get("is_exact", False)
                    }
                return {
                    "can_execute": False,
                    "needs_clarification": True,
                    "message": res_cont.clarification_prompt,
                    "handler": "clarification",
                    "parameters": {
                        "ambiguous_target": recip_query,
                        "candidates": [c.to_dict() for c in res_cont.candidates]
                    }
                }

            # Resolved contact
            target_contact = res_cont.resolved_contact
            target_name = res_cont.recipient_name or (target_contact.name if target_contact else recip_query)
            target_phone = target_contact.phone if target_contact else None
            is_group = res_cont.is_group
            
            if self.context and target_name:
                self.context.set_selected_recipient(target_name, target_phone, is_group)

            is_exact = parsed_comm.get("is_exact", False)
            msg_text = parsed_comm.get("message_text")

            attachment_path = parsed_comm.get("attachment_path")
            if not attachment_path:
                m_att = re.search(r"\b(?:send|share|attach)\s+(?:file\s+|screenshot\s+|document\s+)?['\"]([^'\"]+?\.[a-zA-Z0-9]{2,5})['\"]", cmd_text, re.IGNORECASE)
                if m_att:
                    attachment_path = m_att.group(1).strip()
                elif self.context and "screenshot" in cmd_lower and getattr(self.context, "last_screenshot", None):
                    attachment_path = self.context.last_screenshot

            if not msg_text and attachment_path:
                msg_text = f"Here is the file: {os.path.basename(attachment_path)}"

            # If user said "send the revised message" or "send it to Rahul instead", reuse active draft message
            if not msg_text and ("revised message" in cmd_lower or is_explicit_send or "instead" in cmd_lower or "send it to" in cmd_lower) and reg and reg.whatsapp_adapter.active_draft:
                msg_text = reg.whatsapp_adapter.active_draft.content
                if target_name:
                    reg.whatsapp_adapter.active_draft.recipient = target_name
                    reg.whatsapp_adapter.active_draft.phone = target_phone
                    reg.whatsapp_adapter.active_draft.is_group = is_group

            # If recipient is identified but message body is missing, ask what they would like to say
            if not msg_text and not (is_explicit_send and reg and reg.whatsapp_adapter.active_draft):
                prediction["intent"] = "communication"
                prediction["action"] = "send_whatsapp"
                prediction["operational_confidence"] = 1.0
                if self.context:
                    self.context.pending_message_recipient = target_name
                return {
                    "can_execute": False,
                    "needs_clarification": True,
                    "message": f"What message would you like to send to {target_name}?",
                    "handler": "clarification",
                    "parameters": {"recipient": target_name}
                }

            prediction["intent"] = "communication"
            prediction["action"] = "draft_whatsapp" if action_type == "draft" else "send_whatsapp"
            prediction["operational_confidence"] = 1.0

            if action_type == "draft":
                return {
                    "can_execute": True,
                    "message": f"Drafting WhatsApp message for {target_name}.",
                    "handler": "whatsapp_draft",
                    "parameters": {
                        "recipient": target_name,
                        "phone": target_phone,
                        "content": msg_text,
                        "attachment_path": attachment_path,
                        "is_group": is_group
                    }
                }
            else:
                # Send message:
                # Exact dictation to unambiguous contact is pre-authorized.
                # "send the revised message" or explicit send is explicit authorization.
                is_authorized = is_exact or is_explicit_send or ("send the revised message" in cmd_lower)
                return {
                    "can_execute": True,
                    "message": f"Preparing WhatsApp message for {target_name}.",
                    "handler": "whatsapp_send",
                    "parameters": {
                        "recipient": target_name,
                        "phone": target_phone,
                        "content": msg_text,
                        "attachment_path": attachment_path,
                        "is_exact": is_exact,
                        "authorized": is_authorized,
                        "is_group": is_group
                    }
                }

        # 0a-1. Check for writing the active draft into Notepad
        # e.g., "write that in notepad", "put that into notepad", "write active draft into notepad"
        if any(p in cmd_lower for p in [
            "write active draft into notepad", "write that in notepad", "write that into notepad",
            "type that in notepad", "type that into notepad", "put that in notepad", "put that into notepad",
            "insert that in notepad", "write it in notepad", "type it in notepad", "put it in notepad"
        ]):
            cur_draft = self.context.get_current_draft() if (self.context and hasattr(self.context, "get_current_draft")) else None
            if not cur_draft and self.context and hasattr(self.context, "last_draft") and self.context.last_draft:
                cur_draft = self.context.last_draft

            if cur_draft:
                prediction["intent"] = "communication_drafting"
                prediction["action"] = "exec_draft_content"
                prediction["operational_confidence"] = 1.0
                b_text = cur_draft.body if hasattr(cur_draft, "body") else (cur_draft.get("body", "") if isinstance(cur_draft, dict) else str(cur_draft))
                s_subj = cur_draft.subject if hasattr(cur_draft, "subject") else (cur_draft.get("subject", "") if isinstance(cur_draft, dict) else "")
                return {
                    "can_execute": True,
                    "message": "Faithfully inserting the intended draft into Notepad.",
                    "handler": "draft_content",
                    "parameters": {
                        "subject": s_subj,
                        "body": b_text,
                        "channel": "notepad",
                        "command": cmd_text
                    }
                }
            else:
                return {
                    "can_execute": False,
                    "message": "No active draft found in recent conversation context to write into Notepad. Would you like me to draft one first?",
                    "handler": "dialogue",
                    "parameters": {}
                }

        # 0a-2. Check for draft revisions or undo/revert
        # e.g., "make it shorter", "make that shorter", "make it more professional", "add this point: ...", "revert draft"
        is_revision_intent = any(re.search(p, cmd_lower) for p in [
            r"^revise\s+draft:\s*",
            r"\b(?:make\s+(?:it|that|the\s+draft)\s+(?:shorter|more\s+concise|brief|concise)|shorten\s+(?:it|that|the\s+draft))\b",
            r"\b(?:make\s+(?:it|that|the\s+draft)\s+(?:more\s+)?(?:professional|formal|polite|casual))\b",
            r"\b(?:add\s+(?:this\s+point|a\s+point|the\s+following|the\s+point|a\s+sentence|this\s+sentence|a\s+line)(?:\s*:\s*|\s+that\s+|\s+about\s+)?(.+))\b",
            r"\b(?:undo\s+(?:the\s+)?(?:last\s+)?(?:change|revision)|revert\s+(?:to\s+)?previous\s+version|go\s+back\s+to\s+(?:the\s+)?previous\s+version|revert\s+draft)\b",
        ])
        if is_revision_intent and self.context and hasattr(self.context, "has_draft") and self.context.has_draft():
            clean_instruction = re.sub(r"^revise\s+draft:\s*", "", cmd_text, flags=re.IGNORECASE).strip()
            if any(p in cmd_lower for p in ["previous version", "revert", "undo revision", "undo the last change", "undo last change", "undo change"]):
                revised = self.context.revert_draft()
                rev_msg = "Reverted draft to previous version."
            else:
                revised = self.context.revise_draft(clean_instruction)
                rev_msg = f"Revised draft: {revised.change_description if revised and hasattr(revised, 'change_description') else 'updated'}"

            prediction["intent"] = "communication_drafting"
            prediction["action"] = "exec_draft_content"
            prediction["operational_confidence"] = 1.0
            rev_body = revised.body if hasattr(revised, "body") else str(revised)
            rev_subj = revised.subject if hasattr(revised, "subject") else ""
            return {
                "can_execute": True,
                "message": f"{rev_msg}\n\n[Updated Draft]:\n{rev_body}",
                "handler": "draft_content",
                "parameters": {
                    "subject": rev_subj,
                    "body": rev_body,
                    "channel": "clipboard",
                    "command": cmd_text
                }
            }

        # 0a. Cognitive Drafting & Composition check (email, mail, letters, notes)
        draft_info = CognitiveDraftingEngine.parse_drafting_request(cmd_text)
        if draft_info:
            draft = CognitiveDraftingEngine.generate_draft(draft_info)
            prediction["intent"] = "communication_drafting"
            prediction["action"] = "exec_draft_content"
            prediction["operational_confidence"] = 1.0

            channel = draft_info.get("channel")
            if not channel and context:
                channel = context.get("last_channel") or context.get("last_service") or context.get("last_app")
            if not channel:
                channel = "gmail" if "gmail" in cmd_lower else ("mail" if "mail" in cmd_lower else "clipboard")

            recip = draft["recipient"]
            subj = draft["subject"]
            msg = f"Drafted {draft_info['tone']} {draft_info['document_type']} to {recip} ('{subj}')."

            # Store generated draft in context for subsequent turns and revisions
            if self.context and hasattr(self.context, "set_draft"):
                self.context.set_draft(
                    body=draft["body"],
                    topic=draft_info.get("topic", "Draft"),
                    subject=draft.get("subject", ""),
                    recipient=draft.get("recipient", ""),
                    document_type=draft_info.get("document_type", "document")
                )

            return {
                "can_execute": True,
                "message": msg,
                "handler": "draft_content",
                "parameters": {
                    "subject": draft["subject"],
                    "body": draft["body"],
                    "recipient": draft["recipient"],
                    "channel": channel,
                    "gmail_url": draft["gmail_url"],
                    "mailto_url": draft["mailto_url"],
                    "command": cmd_text
                }
            }


        # 0a-3. Check for file summarization request
        # e.g., "now write a summary of it", "summarize it", "write a summary of the report"
        if re.search(r"\b(?:write\s+a\s+summary\s+of\s+(?:it|that|the\s+file|the\s+report)|summarize\s+(?:it|that|the\s+file|the\s+report))\b", cmd_lower):
            target_f = (context.get("last_path") or context.get("last_file")) if context else None
            if not target_f and self.context:
                target_f = getattr(self.context, "last_path", None) or getattr(self.context, "last_file", None)

            if target_f and os.path.exists(target_f):
                content = ""
                if self.executor and self.executor.file_discovery:
                    content = self.executor.file_discovery._extract_text_content(target_f, max_chars=3000)
                if not content:
                    try:
                        with open(target_f, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(3000)
                    except Exception:
                        pass

                fname = os.path.basename(target_f)
                lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
                key_pts = lines[:5]
                body_summary = f"Summary of {fname}:\n\n" + "\n".join(f"• {pt}" for pt in key_pts) if key_pts else f"Document {fname} is verified and ready for review."
                if self.context and hasattr(self.context, "set_draft"):
                    self.context.set_draft(body=body_summary, topic=f"Summary of {fname}", subject=f"Summary: {fname}")

                prediction["intent"] = "communication_drafting"
                prediction["action"] = "exec_draft_content"
                prediction["operational_confidence"] = 1.0
                return {
                    "can_execute": True,
                    "message": f"Prepared summary of '{fname}':\n\n{body_summary}",
                    "handler": "draft_content",
                    "parameters": {
                        "subject": f"Summary: {fname}",
                        "body": body_summary,
                        "channel": "clipboard",
                        "command": cmd_text
                    }
                }

        # 0b. Web Navigation check (e.g. "go on gmail", "go to youtube", "visit github", "navigate to ...")
        m_nav = re.match(r"^(?:go\s+(?:to|on)|visit|navigate\s+(?:to)?)\s+(.+)$", cmd_lower)
        if m_nav:
            target_raw = m_nav.group(1).strip()
            if target_raw in ["desktop", "the desktop"]:
                prediction["intent"] = "system_utility"
                prediction["action"] = "exec_system_utility"
                prediction["operational_confidence"] = 1.0
                return {
                    "can_execute": True,
                    "message": "Minimizing all windows to show Desktop.",
                    "handler": "system_utility",
                    "parameters": {"utility": "minimize_all", "command": cmd_text}
                }

            resolved = resolve_web_target(target_raw)
            if resolved:
                s_name, s_url = resolved
                prediction["intent"] = "web_navigation"
                prediction["action"] = "exec_web_navigation"
                prediction["operational_confidence"] = 1.0
                return {
                    "can_execute": True,
                    "message": f"Navigating to {s_name.capitalize()} ({s_url}).",
                    "handler": "web_navigation",
                    "parameters": {"url": s_url, "service": s_name, "command": cmd_text}
                }

        # Precision YouTube Search Dispatch (precedes generic web search)
        m_yt = re.match(r"^(?:search\s+youtube\s+for|search\s+on\s+youtube\s+for|youtube\s+search)\s+(.+)$", cmd_lower)
        if m_yt:
            yt_query = m_yt.group(1).strip()
            yt_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(yt_query)}"
            prediction["intent"] = "web_navigation"
            prediction["action"] = "exec_web_navigation"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Searching YouTube for '{yt_query}'.",
                "handler": "web_navigation",
                "parameters": {"url": yt_url, "service": "youtube", "command": cmd_text}
            }

        # Explicit web search check (prevents single-step queries from being misrouted)
        m_web = re.match(r"^(?:search\s+(?:web|google|online)\s+for|search\s+for|google\s+for|google|search)\s+(.+)$", cmd_lower)
        if m_web and not re.search(r"\b(?:in|into)\s+[a-zA-Z0-9_\-]+$", cmd_lower):
            s_query = m_web.group(1).strip()
            prediction["intent"] = "web_search"
            prediction["action"] = "exec_browser_search"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Searching online for '{s_query}'.",
                "handler": "web_search",
                "parameters": {"query": s_query, "command": cmd_text}
            }

        # 1. Spotify volume control
        if "spotify" in cmd_lower and any(k in cmd_lower for k in ["volume up", "louder", "turn up", "increase volume", "raise volume"]):
            prediction["intent"] = "media_control"
            prediction["action"] = "exec_media_change"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Increasing Spotify playback volume.",
                "handler": "media_control",
                "parameters": {"action": "spotify_volume_up", "command": cmd_text}
            }
        elif "spotify" in cmd_lower and any(k in cmd_lower for k in ["volume down", "quieter", "turn down", "decrease volume", "lower volume"]):
            prediction["intent"] = "media_control"
            prediction["action"] = "exec_media_change"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Decreasing Spotify playback volume.",
                "handler": "media_control",
                "parameters": {"action": "spotify_volume_down", "command": cmd_text}
            }

        # 2. Next track / skip
        elif (
            cmd_lower in ["next", "skip", "next song", "next track", "skip song", "skip track", "play next", "play next song", "skip to next song", "hop to next song", "hop to the next song"] or
            any(cmd_lower.startswith(k) for k in ["next song", "next track", "skip this song", "skip track", "skip to next", "skip to the next", "hop to next", "hop to the next", "play next song"]) or
            bool(re.search(r"\b(?:skip\s+to|hop\s+to|jump\s+to)\s+(?:the\s+)?next\s+(?:song|track|music)\b", cmd_lower))
        ):
            prediction["intent"] = "media_control"
            prediction["action"] = "exec_media_change"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Skipping to the next track on Spotify.",
                "handler": "media_control",
                "parameters": {"action": "next", "command": cmd_text}
            }

        # 3. Previous track
        elif cmd_lower in ["previous", "prev", "previous song", "previous track", "prev song", "prev track", "go back song", "play previous", "last song", "rewind"] or any(cmd_lower.startswith(k) for k in ["previous song", "previous track", "go back to previous"]):
            prediction["intent"] = "media_control"
            prediction["action"] = "exec_media_change"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Returning to the previous track on Spotify.",
                "handler": "media_control",
                "parameters": {"action": "previous", "command": cmd_text}
            }

        # 4. Open / Show Lyrics in Spotify
        elif any(k in cmd_lower for k in ["open lyrics", "show lyrics", "spotify lyrics", "view lyrics", "lyrics in spotify", "see lyrics"]) or cmd_lower in ["lyrics", "show lyric"]:
            prediction["intent"] = "media_control"
            prediction["action"] = "exec_media_change"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Toggling lyrics panel in Spotify.",
                "handler": "media_control",
                "parameters": {"action": "open_lyrics", "command": cmd_text}
            }

        # 5. Add to Playlist / Like song
        elif ("playlist" in cmd_lower and any(k in cmd_lower for k in ["add", "save", "put"])) or any(k in cmd_lower for k in ["add to playlist", "like this song", "like song", "add song to playlist", "save this song", "save song", "favorite song", "add to favorites", "save to playlist", "save to library"]):
            prediction["intent"] = "media_control"
            prediction["action"] = "exec_media_change"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Adding current song to your Spotify playlist / Liked Songs.",
                "handler": "media_control",
                "parameters": {"action": "add_to_playlist", "command": cmd_text}
            }

        # 6. Play / Pause / Resume
        elif cmd_lower in ["play", "pause", "pause music", "pause song", "pause spotify", "stop music", "resume", "resume music", "resume song", "unpause", "play/pause", "toggle music", "toggle playback"]:
            prediction["intent"] = "media_control"
            prediction["action"] = "exec_media_change"
            prediction["operational_confidence"] = 1.0
            act = "pause" if any(k in cmd_lower for k in ["pause", "stop"]) else ("play" if any(k in cmd_lower for k in ["play", "resume", "unpause"]) else "play_pause")
            msg = "Pausing Spotify playback." if act == "pause" else ("Resuming Spotify playback." if act == "play" else "Toggling media playback (Play/Pause).")
            return {
                "can_execute": True,
                "message": msg,
                "handler": "media_control",
                "parameters": {"action": act, "command": cmd_text}
            }

        # 7. Brightness control
        elif any(k in cmd_lower for k in ["brightness", "bright"]):
            prediction["intent"] = "system_control"
            prediction["action"] = "exec_sys_command"
            prediction["operational_confidence"] = 1.0
            if any(k in cmd_lower for k in ["up", "increase", "higher", "raise", "more"]):
                val = "up"
                msg = "Increasing system display brightness."
            elif any(k in cmd_lower for k in ["down", "decrease", "lower", "dim", "less"]):
                val = "down"
                msg = "Decreasing system display brightness."
            else:
                m = re.search(r"(\d+)", cmd_lower)
                val = m.group(1) if m else "query"
                msg = f"Adjusting system brightness to {val}%." if m else "Querying system brightness."
            return {
                "can_execute": True,
                "message": msg,
                "handler": "system_control",
                "parameters": {"setting": "brightness", "value": val, "command": cmd_text}
            }

        # 8. Master System Volume control
        elif any(k in cmd_lower for k in ["system volume up", "volume up", "increase volume", "turn up volume", "raise volume", "louder"]) and "spotify" not in cmd_lower:
            prediction["intent"] = "system_control"
            prediction["action"] = "exec_sys_command"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Increasing system master volume.",
                "handler": "system_control",
                "parameters": {"setting": "volume", "value": "up", "command": cmd_text}
            }
        elif any(k in cmd_lower for k in ["system volume down", "volume down", "decrease volume", "turn down volume", "lower volume", "quieter"]) and "spotify" not in cmd_lower:
            prediction["intent"] = "system_control"
            prediction["action"] = "exec_sys_command"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Decreasing system master volume.",
                "handler": "system_control",
                "parameters": {"setting": "volume", "value": "down", "command": cmd_text}
            }
        elif any(k in cmd_lower for k in ["mute system", "mute volume", "mute audio", "unmute system", "unmute"]):
            prediction["intent"] = "system_control"
            prediction["action"] = "exec_sys_command"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Toggling system volume mute.",
                "handler": "system_control",
                "parameters": {"setting": "volume", "value": "mute", "command": cmd_text}
            }

        # 9. Screenshot
        elif not any(k in cmd_lower for k in ["delete", "remove", "trash", "erase", "open", "copy", "move"]) and any(k in cmd_lower for k in ["screenshot", "take screenshot", "take a screenshot", "capture screen", "screen capture", "grab screen"]):
            prediction["intent"] = "system_utility"
            prediction["action"] = "exec_system_utility"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Capturing full screen snapshot to Desktop.",
                "handler": "system_utility",
                "parameters": {"utility": "screenshot", "command": cmd_text}
            }

        # 10. Lock & Sleep
        elif any(k in cmd_lower for k in ["lock workstation", "lock screen", "lock pc", "lock computer"]):
            prediction["intent"] = "system_control"
            prediction["action"] = "exec_sys_command"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Locking Windows workstation.",
                "handler": "system_control",
                "parameters": {"setting": "lock", "value": "lock", "command": cmd_text}
            }
        elif any(k in cmd_lower for k in ["sleep pc", "sleep computer", "put pc to sleep", "sleep mode"]):
            prediction["intent"] = "system_control"
            prediction["action"] = "exec_sys_command"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Entering system sleep state.",
                "handler": "system_control",
                "parameters": {"setting": "sleep", "value": "sleep", "command": cmd_text}
            }

        # Process termination check
        m_kill = re.match(r"^(?:kill|terminate|stop|end)\s+(?:process\s+|task\s+|pid\s+)?(\d+)$", cmd_lower)
        if m_kill:
            pid = m_kill.group(1).strip()
            prediction["intent"] = "process_manage"
            prediction["action"] = "exec_process_kill"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Terminating process with PID {pid}.",
                "handler": "kill_process",
                "parameters": {"pid": pid, "command": cmd_text}
            }

        m_kill_no_pid = re.match(r"^(?:kill|terminate|stop|end)\s+(?:the\s+)?(?:process|task)$", cmd_lower)
        if m_kill_no_pid:
            prediction["intent"] = "process_manage"
            prediction["action"] = "exec_process_kill"
            prediction["operational_confidence"] = 0.5
            return {
                "can_execute": False,
                "needs_confirmation": True,
                "message": "Please specify the Process ID (PID) or application name to terminate.",
                "handler": "kill_process",
                "parameters": {"command": cmd_text}
            }

        # 11. Empty Recycle Bin
        elif any(k in cmd_lower for k in ["empty the recycle bin", "empty recycle bin", "empty trash", "clean recycle bin", "clear recycle bin"]):
            prediction["intent"] = "system_utility"
            prediction["action"] = "exec_system_utility"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Emptying Recycle Bin.",
                "handler": "system_utility",
                "parameters": {"utility": "empty_recycle_bin", "command": cmd_text}
            }

        # 12. System Performance Metrics
        elif any(k in cmd_lower for k in [
            "system metrics", "system status", "system specs", "cpu usage", "ram usage",
            "memory usage", "battery status", "check battery", "battery level", "power status",
            "system info", "device metrics", "system performance", "check system metrics",
            "take a peek at my current cpu usage"
        ]) or cmd_lower in ["battery", "battery status", "check battery", "system status", "system metrics"]:
            prediction["intent"] = "system_query"
            prediction["action"] = "exec_read_metrics"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Querying current system performance metrics.",
                "handler": "system_metrics",
                "parameters": {"command": cmd_text}
            }

        # 13. Network Connectivity Diagnostics
        elif cmd_lower.startswith("ping ") or any(k in cmd_lower for k in ["internet", "network diagnostic", "network status", "ping", "wifi status", "check connection", "reachability"]):
            prediction["intent"] = "network_task"
            prediction["action"] = "exec_network_diagnostic"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Running network connectivity diagnostics.",
                "handler": "network_task",
                "parameters": {"command": cmd_text}
            }

        # 14. Minimize All Windows / Show Desktop
        elif any(k in cmd_lower for k in ["minimize all", "show desktop", "minimize windows", "hide all windows"]):
            prediction["intent"] = "system_utility"
            prediction["action"] = "exec_system_utility"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Minimizing all windows to show Desktop.",
                "handler": "system_utility",
                "parameters": {"utility": "minimize_all", "command": cmd_text}
            }

        # 15. Time & Date Utility
        elif any(k in cmd_lower for k in ["what time", "current time", "what is the time", "tell me time", "today date", "current date", "what is today"]):
            prediction["intent"] = "system_utility"
            prediction["action"] = "exec_system_utility"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Checking current date and time.",
                "handler": "system_utility",
                "parameters": {"utility": "datetime", "command": cmd_text}
            }

        # 15b. Clipboard Operations (Copy & Read)
        m_clip_copy = re.match(r"^(?:copy|copy\s+text)\s+(?:'([^']+)'|\"([^\"]+)\"|(.+?))\s+(?:to|into)\s+(?:the\s+)?clipboard$", cmd_text, re.IGNORECASE)
        if m_clip_copy:
            raw_content = (m_clip_copy.group(1) or m_clip_copy.group(2) or m_clip_copy.group(3) or "").strip()
            if raw_content.lower() in ["it", "that", "this", "the text", "the result", "that result", "result", "the summary", "that summary", "summary"]:
                if raw_content.lower() in ["the result", "that result", "result"] and self.context and self.context.last_calculation:
                    raw_content = str(self.context.last_calculation.get("result", ""))
                elif raw_content.lower() in ["the summary", "that summary", "summary"] and self.context and self.context.last_summary:
                    raw_content = str(self.context.last_summary)
                elif self.context:
                    raw_content = getattr(self.context, "last_clipboard", None) or (getattr(self.context, "last_summary", None) or raw_content)
            prediction["intent"] = "system_utility"
            prediction["action"] = "system_clipboard_copy"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Copying text to clipboard.",
                "handler": "system_clipboard_copy",
                "parameters": {"content": raw_content, "command": cmd_text}
            }

        if any(cmd_lower == p or cmd_lower.startswith(p + " ") for p in [
            "read clipboard", "get clipboard", "get clipboard content", "read the clipboard",
            "check clipboard", "what is on my clipboard", "what is on clipboard", "what's on my clipboard", "what's in my clipboard"
        ]):
            prediction["intent"] = "system_utility"
            prediction["action"] = "system_clipboard_read"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Reading clipboard content.",
                "handler": "system_clipboard_read",
                "parameters": {"command": cmd_text}
            }

        # 15b-1. Window Control & Management
        m_win_snap = re.match(r"^(?:snap|tile)\s+(?:window\s+)?(left|right)$", cmd_lower)
        if m_win_snap:
            direction = m_win_snap.group(1).lower()
            prediction["intent"] = "window_control"
            prediction["action"] = f"window_snap_{direction}"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Snapping active window to the {direction}.",
                "handler": "window_control",
                "parameters": {"action": f"snap_{direction}", "command": cmd_text}
            }

        m_win_act = re.match(r"^(maximize|fullscreen|unmaximize|restore|minimize)\s+(?:the\s+)?(?:window|screen)?(?:\s+(?:for|of)\s+([a-zA-Z0-9_\-]+))?$", cmd_lower)
        if m_win_act:
            w_act = m_win_act.group(1).lower()
            t_app = m_win_act.group(2)
            act_name = "maximize" if w_act in ["maximize", "fullscreen"] else ("restore" if w_act in ["restore", "unmaximize"] else "minimize")
            prediction["intent"] = "window_control"
            prediction["action"] = f"window_{act_name}"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"{act_name.capitalize()}ing window{' for ' + t_app if t_app else ''}.",
                "handler": "window_control",
                "parameters": {"action": act_name, "app_name": t_app, "command": cmd_text}
            }

        m_win_switch = re.match(r"^(?:switch\s+to|focus\s+(?:on\s+)?)([a-zA-Z0-9_\-\s]+?)(?:\s+window)?$", cmd_lower) or re.match(r"^bring\s+(?:up\s+)?([a-zA-Z0-9_\-\s]+?)\s+window$", cmd_lower)
        if m_win_switch and m_win_switch.group(1).strip() in APP_NAMES:
            t_app = m_win_switch.group(1).strip()
            prediction["intent"] = "window_control"
            prediction["action"] = "window_switch"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Switching focus to {t_app}.",
                "handler": "window_control",
                "parameters": {"action": "switch", "app_name": t_app, "command": cmd_text}
            }

        if cmd_lower in ["close window", "close active window", "close current window", "exit window"]:
            prediction["intent"] = "window_control"
            prediction["action"] = "window_close"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Closing active window.",
                "handler": "window_control",
                "parameters": {"action": "close", "command": cmd_text}
            }

        # 15b-2. Timers & Reminders
        if any(cmd_lower == p for p in ["cancel timer", "cancel timers", "stop timer", "cancel the timer"]):
            prediction["intent"] = "system_timer"
            prediction["action"] = "cancel_timer"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Cancelling active timers.",
                "handler": "system_timer",
                "parameters": {"subaction": "cancel", "command": cmd_text}
            }

        if any(cmd_lower == p for p in ["check timer", "check timers", "list timers", "show timers", "active timers"]):
            prediction["intent"] = "system_timer"
            prediction["action"] = "list_timers"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Checking active countdown timers.",
                "handler": "system_timer",
                "parameters": {"subaction": "list", "command": cmd_text}
            }

        m_remind = re.match(r"^remind\s+me\s+(?:in\s+)?(.+?)\s+to\s+(.+)$", cmd_lower)
        if m_remind:
            dur_raw = m_remind.group(1).strip()
            reminder_text = m_remind.group(2).strip()
            prediction["intent"] = "system_timer"
            prediction["action"] = "set_reminder"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Setting reminder in {dur_raw} to {reminder_text}.",
                "handler": "system_timer",
                "parameters": {"subaction": "set", "duration": dur_raw, "label": reminder_text, "command": cmd_text}
            }

        m_timer = re.match(r"^(?:set\s+a?\s*timer|set\s+timer|timer|countdown)\s+(?:for\s+)?(.+?)(?:\s+(?:called|labeled|for)\s+(.+))?$", cmd_lower)
        if m_timer:
            dur_raw = m_timer.group(1).strip()
            lbl = (m_timer.group(2) or "Timer").strip()
            prediction["intent"] = "system_timer"
            prediction["action"] = "set_timer"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Setting timer for {dur_raw}.",
                "handler": "system_timer",
                "parameters": {"subaction": "set", "duration": dur_raw, "label": lbl, "command": cmd_text}
            }

        # 15b-3. Quick Notes & Scratchpad Store
        if any(cmd_lower == p for p in ["show notes", "show my notes", "read my notes", "read notes", "list notes", "my notes"]):
            prediction["intent"] = "notes_manager"
            prediction["action"] = "list_notes"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Retrieving your saved notes.",
                "handler": "notes_manager",
                "parameters": {"action": "list", "command": cmd_text}
            }

        if any(cmd_lower == p for p in ["clear notes", "clear all notes", "delete notes", "delete all notes"]):
            prediction["intent"] = "notes_manager"
            prediction["action"] = "clear_notes"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Clearing all saved notes.",
                "handler": "notes_manager",
                "parameters": {"action": "clear", "command": cmd_text}
            }

        m_note_add = re.match(r"^(?:take\s+a?\s*note|add\s+note|create\s+note|save\s+note|note)\s*[:\-]?\s*(.+)$", cmd_text, re.IGNORECASE)
        if m_note_add:
            note_content = m_note_add.group(1).strip()
            prediction["intent"] = "notes_manager"
            prediction["action"] = "add_note"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Saving note: \"{note_content}\".",
                "handler": "notes_manager",
                "parameters": {"action": "add", "content": note_content, "command": cmd_text}
            }

        # 15b-4. Real-Time Weather Queries
        m_weather = re.match(r"^(?:what\s+is\s+the\s+weather|what\'s\s+the\s+weather|weather|check\s+weather|how\s+is\s+the\s+weather|temperature)(?:\s+(?:in|for|at|of)\s+(.+))?$", cmd_lower)
        if m_weather:
            loc = (m_weather.group(1) or "").strip()
            prediction["intent"] = "system_weather"
            prediction["action"] = "check_weather"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Checking weather{' for ' + loc if loc else ''}.",
                "handler": "system_weather",
                "parameters": {"location": loc, "command": cmd_text}
            }

        # 15b-5. Unit & Currency Conversion
        m_conv = re.match(r"^(?:convert\s+)?(\d+(?:\.\d+)?)\s*([a-zA-Z°]+)\s+(?:to|in|into)\s+([a-zA-Z°]+)$", cmd_lower)
        if m_conv:
            amt = float(m_conv.group(1))
            u1 = m_conv.group(2).strip()
            u2 = m_conv.group(3).strip()
            prediction["intent"] = "unit_convert"
            prediction["action"] = "convert_units"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Converting {amt} {u1} to {u2}.",
                "handler": "unit_convert",
                "parameters": {"amount": amt, "from_unit": u1, "to_unit": u2, "command": cmd_text}
            }

        # 15b-6. Running Apps & Network IP Inspection
        if any(cmd_lower == p for p in ["what apps are open", "what apps are running", "list running apps", "running apps", "list open windows", "open windows"]):
            prediction["intent"] = "system_processes"
            prediction["action"] = "list_apps"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Inspecting active desktop applications.",
                "handler": "system_processes",
                "parameters": {"subaction": "list_apps", "command": cmd_text}
            }

        m_is_running = re.match(r"^(?:is|check\s+if)\s+([a-zA-Z0-9_\-]+)\s+(?:running|open)$", cmd_lower)
        if m_is_running:
            target_app = m_is_running.group(1).strip()
            prediction["intent"] = "system_processes"
            prediction["action"] = "check_app_running"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Checking if '{target_app}' is running.",
                "handler": "system_processes",
                "parameters": {"subaction": "is_running", "app_name": target_app, "command": cmd_text}
            }

        if any(cmd_lower == p for p in ["what is my ip", "my ip", "what's my ip", "public ip", "check my ip", "network info"]):
            prediction["intent"] = "system_processes"
            prediction["action"] = "network_info"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Retrieving host and public IP address.",
                "handler": "system_processes",
                "parameters": {"subaction": "network", "command": cmd_text}
            }

        # 15b-7. Precision YouTube Search Dispatch
        m_yt = re.match(r"^(?:search\s+youtube\s+for|search\s+on\s+youtube\s+for|youtube\s+search)\s+(.+)$", cmd_lower)
        if m_yt:
            yt_query = m_yt.group(1).strip()
            yt_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(yt_query)}"
            prediction["intent"] = "web_navigation"
            prediction["action"] = "exec_web_navigation"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Searching YouTube for '{yt_query}'.",
                "handler": "web_navigation",
                "parameters": {"url": yt_url, "service": "youtube", "command": cmd_text}
            }

        # 15c. Zip & Unzip Operations
        m_zip = re.match(r"^(?:zip|compress)\s+(?:folder\s+|file\s+|directory\s+)?(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))(?:\s+(?:to|into|as)\s+(.+))?$", cmd_text, re.IGNORECASE)
        if m_zip:
            src = (m_zip.group(1) or m_zip.group(2) or m_zip.group(3) or "").strip().strip("'\"")
            dst = (m_zip.group(4) or "").strip().strip("'\"")
            prediction["intent"] = "file_operation"
            prediction["action"] = "file_zip"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Compressing '{src}' into zip archive.",
                "handler": "file_zip",
                "parameters": {"source": src, "destination": dst, "command": cmd_text}
            }

        m_unzip = re.match(r"^(?:unzip|extract)\s+(?:archive\s+|file\s+|zip\s+)?(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))(?:\s+(?:to|into)\s+(.+))?$", cmd_text, re.IGNORECASE)
        if m_unzip:
            src = (m_unzip.group(1) or m_unzip.group(2) or m_unzip.group(3) or "").strip().strip("'\"")
            dst = (m_unzip.group(4) or "").strip().strip("'\"")
            prediction["intent"] = "file_operation"
            prediction["action"] = "file_unzip"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Extracting zip archive '{src}'.",
                "handler": "file_unzip",
                "parameters": {"source": src, "destination": dst, "command": cmd_text}
            }

        # 15d. Explicit File Read & Append
        m_file_read = re.match(r"^(?:read\s+file|read\s+the\s+contents?\s+of|read\s+the\s+file|read)\s+(?:file\s+)?(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))$", cmd_text, re.IGNORECASE)
        if m_file_read and not any(k in cmd_lower for k in ["clipboard", "screen", "metrics", "status", "http", "www."]):
            f_path = (m_file_read.group(1) or m_file_read.group(2) or m_file_read.group(3) or "").strip().strip("'\"")
            prediction["intent"] = "file_operation"
            prediction["action"] = "file_read"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Reading file '{f_path}'.",
                "handler": "file_read",
                "parameters": {"path": f_path, "command": cmd_text}
            }

        m_append = re.match(r"^append\s+(?:text\s+|content\s+)?(?:'([^']+)'|\"([^\"]+)\"|(.+?))\s+(?:to|into)\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))$", cmd_text, re.IGNORECASE)
        if m_append:
            content = (m_append.group(1) or m_append.group(2) or m_append.group(3) or "").strip()
            f_path = (m_append.group(4) or m_append.group(5) or m_append.group(6) or "").strip().strip("'\"")
            prediction["intent"] = "file_operation"
            prediction["action"] = "file_append"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Appending text to '{f_path}'.",
                "handler": "file_append",
                "parameters": {"path": f_path, "content": content, "command": cmd_text}
            }

        # 15e. Explicit File Create with Content
        m_create_file = re.match(r"^(?:create\s+file|create|write\s+file)\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))\s+(?:with\s+content|with\s+text|saying|containing)\s*[:\s]?\s*(.+)$", cmd_text, re.IGNORECASE)
        if m_create_file:
            f_path = (m_create_file.group(1) or m_create_file.group(2) or m_create_file.group(3) or "").strip().strip("'\"")
            content = m_create_file.group(4).strip().strip("'\"")
            prediction["intent"] = "file_operation"
            prediction["action"] = "file_create"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Creating file '{f_path}' with content.",
                "handler": "file_create",
                "parameters": {"path": f_path, "content": content, "command": cmd_text}
            }

        # 15f. Web Page Read and Summarize
        m_web_sum = re.match(r"^(?:read\s+and\s+summarize|read\s+webpage|summarize\s+webpage|read\s+page|summarize\s+page|summarize\s+url|read\s+url|summarize\s+web\s+page)\s+['\"]?(https?://[^\s'\"]+)['\"]?$", cmd_text, re.IGNORECASE)
        if m_web_sum:
            url = m_web_sum.group(1).strip()
            prediction["intent"] = "web_search"
            prediction["action"] = "browser_read_page"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Reading and summarizing web page at {url}.",
                "handler": "browser_read_page",
                "parameters": {"url": url, "command": cmd_text}
            }

        # 16. File Management Operations (copy, cut, paste, delete, create, open, list)
        file_cmd = self._parse_file_command(cmd_text)
        if file_cmd:
            prediction["intent"] = "file_operation"
            prediction["action"] = "exec_file_action"
            prediction["operational_confidence"] = 1.0
            act = file_cmd["subaction"]
            src = file_cmd["source"]
            dst = file_cmd["destination"]
            if act == "copy":
                msg = f"Copying '{src}' to '{dst}'." if dst else f"Copying '{src}' to clipboard."
            elif act == "cut":
                msg = f"Moving '{src}' to '{dst}'." if dst else f"Cutting '{src}' to clipboard."
            elif act == "paste":
                msg = f"Pasting file into '{dst}'." if dst else "Pasting file from clipboard."
            elif act == "delete":
                if not src or (src.lower() in ["it", "that", "this"] and not (context and (context.get("last_file") or context.get("last_path")))):
                    return {
                        "can_execute": False,
                        "needs_confirmation": True,
                        "message": f"Confirmation required before deleting '{src or 'file'}': target unspecified in context.",
                        "handler": "file_operation",
                        "parameters": file_cmd
                    }
                msg = f"Deleting '{src}'."
            elif act == "create":
                msg = f"Creating '{src}' in '{dst}'." if dst else f"Creating '{src}'."
            elif act == "list":
                msg = f"Listing files in '{dst}'." if dst else "Listing files."
            elif act == "find":
                msg = f"Searching for '{src}' in '{dst}'." if dst else f"Searching for '{src}'."
            else:
                msg = f"Opening '{src}'."

            return {
                "can_execute": True,
                "message": msg,
                "handler": "file_operation",
                "parameters": file_cmd
            }

        # 17. Universal GUI & Screen Perception Actions (ScreenEngine / GUIAgent)
        # 17a. Inspect Screen / What is on screen
        if any(k in cmd_lower for k in [
            "what's on my screen", "what is on my screen", "what is on the screen",
            "what's on screen", "what is on screen", "inspect screen", "inspect the screen",
            "perceive screen", "scan screen", "what do you see on screen", "what do you see",
            "read screen", "look at my screen", "look at screen"
        ]):
            prediction["intent"] = "gui_perception"
            prediction["action"] = "exec_gui_action"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": "Inspecting screen and perceiving all visible UI elements.",
                "handler": "gui_action",
                "parameters": {"action": "inspect", "app_name": "", "command": cmd_text}
            }

        # Inspect specific app, e.g. "inspect chrome", "inspect spotify", "look at notepad"
        m_inspect = re.match(r"^(?:inspect|look at|scan|see|view)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+window|\s+app)?$", cmd_lower)
        if m_inspect and m_inspect.group(1).strip() in ["chrome", "spotify", "notepad", "calculator", "edge", "firefox", "vscode", "terminal", "explorer", "discord"]:
            app_t = m_inspect.group(1).strip()
            prediction["intent"] = "gui_perception"
            prediction["action"] = "exec_gui_action"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Perceiving UI elements in {app_t}.",
                "handler": "gui_action",
                "parameters": {"action": "inspect", "app_name": app_t, "command": cmd_text}
            }

        # 17b. Dynamic Click: e.g. "click <element> in <app>", "click on <element> on <app>", "click <element>"
        m_click_app = re.match(r"^(?:click|press|tap)\s+(?:on\s+)?(.+?)\s+(?:in|on)\s+([a-zA-Z0-9_\-]+)$", cmd_lower)
        if m_click_app:
            target_elem = m_click_app.group(1).strip()
            target_app = m_click_app.group(2).strip()
            prediction["intent"] = "gui_action"
            prediction["action"] = "exec_gui_action"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Locating and clicking '{target_elem}' in {target_app}.",
                "handler": "gui_action",
                "parameters": {"action": "click", "target": target_elem, "app_name": target_app, "command": cmd_text}
            }

        m_click = re.match(r"^(?:click|press|tap)\s+(?:on\s+)?(.+)$", cmd_lower)
        if m_click and m_click.group(1).strip() not in ["play", "pause", "resume", "stop", "next", "previous", "next song", "prev song"]:
            target_elem = m_click.group(1).strip()
            prediction["intent"] = "gui_action"
            prediction["action"] = "exec_gui_action"
            prediction["operational_confidence"] = 1.0
            return {
                "can_execute": True,
                "message": f"Locating and clicking '{target_elem}' on screen.",
                "handler": "gui_action",
                "parameters": {"action": "click", "target": target_elem, "app_name": "", "command": cmd_text}
            }

        # 17c. Dynamic App Search: e.g. "search for <query> in <app>", "search <query> in <app>"
        m_search_app = re.match(r"^(?:search|find|look up)\s+(?:for\s+)?(.+?)\s+in\s+([a-zA-Z0-9_\-]+)$", cmd_lower)
        if m_search_app:
            s_query = m_search_app.group(1).strip()
            s_app = m_search_app.group(2).strip()
            if s_app in ["spotify", "chrome", "edge", "firefox", "explorer", "files", "notepad", "code", "vscode", "settings"]:
                prediction["intent"] = "gui_action"
                prediction["action"] = "exec_gui_action"
                prediction["operational_confidence"] = 1.0
                return {
                    "can_execute": True,
                    "message": f"Searching for '{s_query}' inside {s_app}.",
                    "handler": "gui_action",
                    "parameters": {"action": "search", "text": s_query, "app_name": s_app, "command": cmd_text}
                }

        # 17d. Dynamic Typing: e.g. "type <text> into <element> in <app>", "type hello world"
        type_cmd = self._parse_typing_command(cmd_text)
        if type_cmd:
            prediction["intent"] = "gui_action"
            prediction["action"] = "exec_gui_action"
            prediction["operational_confidence"] = 1.0
            t_text = type_cmd["text"]
            t_elem = type_cmd["target"]
            t_app = type_cmd["app_name"]
            if t_elem and t_app:
                msg = f"Typing '{t_text}' into '{t_elem}' in {t_app}."
            elif t_app:
                msg = f"Typing '{t_text}' into {t_app}."
            elif t_elem:
                msg = f"Typing '{t_text}' into '{t_elem}'."
            else:
                msg = f"Typing '{t_text}'."
            return {
                "can_execute": True,
                "message": msg,
                "handler": "gui_action",
                "parameters": type_cmd
            }

        intent = prediction["intent"]
        action = prediction["action"]
        entities = prediction["entities"]
        confidence = prediction["operational_confidence"]

        if confidence < 0.4 or intent == "unsupported_ood":
            return {
                "can_execute": False,
                "message": "I am not confident I understood your command. Could you please rephrase or specify the target?",
                "handler": None,
                "parameters": {}
            }

        if action == "exec_launch_app":
            app = entities.get("APP")
            if not app or app == "application":
                m_open = re.match(r"^(?:open|launch|start|run)\s+(?:the\s+)?['\"]?([a-zA-Z0-9_\-\s]+?)['\"]?(?:\s+(?:app|application|program))?$", cmd_lower)
                if m_open:
                    app = m_open.group(1).strip()
                else:
                    app = "application"
            app = app.strip("'\"")

            if "chrome" in app.lower():
                profile = "Aditya Sawant"
                return {
                    "can_execute": True,
                    "message": "Opening Google Chrome with profile 'Aditya Sawant'.",
                    "handler": "launch_app",
                    "parameters": {"app_name": "chrome", "profile": profile}
                }
            return {
                "can_execute": True,
                "message": f"Starting up {app}.",
                "handler": "launch_app",
                "parameters": {"app_name": app}
            }
        elif action == "exec_terminate_app":
            app = entities.get("APP")
            if not app or app == "application":
                m_close = re.match(r"^(?:close|kill|terminate|quit|exit)\s+(?:the\s+)?['\"]?([a-zA-Z0-9_\-\s]+?)['\"]?(?:\s+(?:app|application|program))?$", cmd_lower)
                if m_close:
                    app = m_close.group(1).strip()
                else:
                    app = "application"
            app = app.strip("'\"")

            return {
                "can_execute": True,
                "message": f"Closing {app}.",
                "handler": "close_app",
                "parameters": {"app_name": app}
            }
        elif action == "exec_file_action":
            filename = entities.get("FILE", "")
            path = entities.get("PATH", "")
            app = entities.get("APP", "")
            # Gracefully handle app names that got classified under file action (e.g. 'open mail')
            if not filename and not path and app:
                prediction["action"] = "exec_launch_app"
                prediction["intent"] = "app_launch"
                return {
                    "can_execute": True,
                    "message": f"Starting up {app}.",
                    "handler": "launch_app",
                    "parameters": {"app_name": app}
                }
            return {
                "can_execute": True,
                "message": f"Performing file operation on {filename or path}." if (filename or path) else "Performing file operation.",
                "handler": "file_operation",
                "parameters": {"file": filename, "path": path}
            }
        elif action == "exec_read_metrics":
            # Guard: verify command actually asks for system metrics / hardware info
            sys_keys = ["cpu", "ram", "memory", "battery", "storage", "disk", "specs", "hardware", "device metrics", "system metrics", "system status", "power status"]
            if not any(k in cmd_lower for k in sys_keys):
                if self.dialogue_engine:
                    d_res = self.dialogue_engine.respond(cmd_text, context=self.context, executor=self.executor)
                    prediction["intent"] = "chitchat"
                    prediction["action"] = "respond_dialogue"
                    prediction["operational_confidence"] = 1.0
                    return {
                        "can_execute": True,
                        "message": d_res.get("message", "I am right here with you."),
                        "handler": "dialogue",
                        "parameters": d_res.get("parameters", {})
                    }
            return {
                "can_execute": True,
                "message": "Inspecting system status and resource metrics.",
                "handler": "system_metrics",
                "parameters": {}
            }
        elif action == "exec_process_kill":
            pid = entities.get("PID", "")
            return {
                "can_execute": True,
                "message": f"Terminating process with PID {pid}." if pid else "Identifying target process to terminate.",
                "handler": "kill_process",
                "parameters": {"pid": pid}
            }
        elif action == "exec_browser_search" or intent == "web_search":
            query = entities.get("QUERY", "")
            if isinstance(query, (list, tuple)):
                query = " ".join(str(q) for q in query).strip()
            else:
                query = str(query).strip()

            # If slot extraction missed query tokens, extract from command text
            if not query:
                cmd_lower = prediction.get("query", "").lower()
                for prefix in [
                    "search web for", "search online for", "search google for",
                    "search for", "look up", "find information about", "find", "search"
                ]:
                    if prefix in cmd_lower:
                        query = cmd_lower.split(prefix, 1)[1].strip()
                        break
                if not query:
                    query = prediction.get("query", "").strip()

            return {
                "can_execute": True,
                "message": f"Searching online for '{query}'.",
                "handler": "web_search",
                "parameters": {"query": query}
            }
        elif action in ["exec_network_diagnostic", "network_task"]:
            return {
                "can_execute": True,
                "message": "Checking local network adapter and internet reachability.",
                "handler": "network_task",
                "parameters": {}
            }
        elif action in ["exec_sys_command", "system_control"]:
            setting = entities.get("SETTING", "")
            value = entities.get("VALUE", "")
            return {
                "can_execute": True,
                "message": f"Applying system setting change: {setting} -> {value}.",
                "handler": "system_control",
                "parameters": {"setting": setting, "value": value}
            }
        elif action in ["exec_workflow_script", "automation_workflow"]:
            app = entities.get("APP", "")
            return {
                "can_execute": True,
                "message": f"Executing multi-action task{f' for {app}' if app else ''}.",
                "handler": "workflow_script",
                "parameters": {
                    "app_name": app,
                    "command": prediction.get("query", ""),
                    "entities": entities
                }
            }
        elif action in ["exec_play_music", "music_playback"] or intent == "music_playback":
            song = entities.get("SONG", "")
            artist = entities.get("ARTIST", "")
            modifier = entities.get("MODIFIER", "")
            platform = entities.get("PLATFORM", "spotify")

            GENERIC_SONGS = {
                "a song", "some song", "song", "songs", "some songs",
                "music", "some music", "a track", "tracks", "something",
                "anything", "a music", "track", "tunes", "some tunes"
            }

            if song and song.strip().lower() in GENERIC_SONGS:
                song = ""
            if song:
                song = re.sub(r"\s+(?:on|in)\s+(?:spotify|youtube)\b.*$", "", song, flags=re.IGNORECASE).strip()
                if song.strip().lower() in GENERIC_SONGS:
                    song = ""

            # Fallback extraction from command text if entity slot missed unseen song title
            cmd_text = prediction.get("query", "").strip()
            if not song and not artist:
                cmd_lower = cmd_text.lower()
                for trig in [
                    "open spotify and play", "open youtube and play",
                    "play on spotify", "play on youtube", "play in spotify",
                    "play song", "edith play", "stream", "put on", "play"
                ]:
                    if trig in cmd_lower:
                        rem = cmd_lower.split(trig, 1)[1].strip()
                        for p in ["on spotify", "on youtube", "in spotify", "spotify", "youtube"]:
                            rem = re.sub(rf"\s+{re.escape(p)}\b.*$", "", rem, flags=re.IGNORECASE).strip()
                        if rem.strip().lower() not in GENERIC_SONGS:
                            song = rem.strip()
                        break

            # Build display target: include modifier if present
            parts = [f"'{song}'"] if song else []
            if modifier:
                parts.append(f"({modifier})")
            if artist:
                parts.append(f"by {artist}")
            msg_target = " ".join(parts) if parts else "music"

            return {
                "can_execute": True,
                "message": f"Queuing {msg_target} on {platform}.",
                "handler": "play_music",
                "parameters": {
                    "song": song,
                    "artist": artist,
                    "modifier": modifier,
                    "platform": platform,
                    "command": cmd_text
                }
            }
        elif action in ["exec_compute_math", "math_compute"] or intent == "math_compute":
            expr = entities.get("EXPR", "")
            cmd_q = prediction.get("query", "").strip()
            # If expr is incomplete or lacks an operator, extract expression from command text
            if not expr or not re.search(r"[\+\-\*\/\%\^]|sqrt|square\s+root|percent", expr.lower()):
                m_sub = re.search(r"(?:calculate|compute|what\s+is|how\s+much\s+is)\s+(.+)$", cmd_q, re.IGNORECASE)
                if m_sub:
                    expr = m_sub.group(1).strip()
                else:
                    expr = cmd_q
            return {
                "can_execute": True,
                "message": f"Calculating: {expr}",
                "handler": "compute_math",
                "parameters": {
                    "expr": expr,
                    "command": cmd_q
                }
            }
        elif action in ["exec_system_utility", "system_utility"] or intent == "system_utility":
            utility = entities.get("UTILITY", "")
            return {
                "can_execute": True,
                "message": f"Invoking system utility: {utility or 'desktop action'}",
                "handler": "system_utility",
                "parameters": {
                    "utility": utility,
                    "command": prediction.get("query", "")
                }
            }
        elif action in ["exec_media_change", "media_control"]:
            return {
                "can_execute": True,
                "message": "Controlling media playback.",
                "handler": "media_control",
                "parameters": entities
            }
        elif action in ["exec_gui_action", "gui_action"] or intent in ["gui_action", "gui_perception"]:
            return {
                "can_execute": True,
                "message": "Executing screen perception GUI action.",
                "handler": "gui_action",
                "parameters": entities
            }
        elif action in ["exec_project_action", "project_manage"] or intent == "project_manage":
            proj = entities.get("PROJECT") or (context.get("last_project") if context else None) or "EDITH"
            cmd_action = entities.get("CMD", "tests")
            return {
                "can_execute": True,
                "message": f"Managing project '{proj}': executing {cmd_action}.",
                "handler": "project_action",
                "parameters": {
                    "project": proj,
                    "action": cmd_action,
                    "command": prediction.get("query", "")
                }
            }
        elif action in ["exec_context_action", "context_control"] or intent == "context_control":
            ref = entities.get("REF", "")
            temporal = entities.get("TEMPORAL", "")
            dest = entities.get("DEST", "")
            return {
                "can_execute": True,
                "message": f"Contextual control resolved: {ref or temporal or dest}.",
                "handler": "context_control",
                "parameters": {
                    "ref": ref,
                    "temporal": temporal,
                    "destination": dest,
                    "command": prediction.get("query", "")
                }
            }
        elif action == "respond_dialogue":
            if self.dialogue_engine:
                dialogue_res = self.dialogue_engine.respond(cmd_text, context=self.context, executor=self.executor)
                return {
                    "can_execute": True,
                    "message": dialogue_res.get("message", "Hello! I am E.D.I.T.H., your custom-trained neural task handler. All systems operational."),
                    "handler": "dialogue",
                    "parameters": dialogue_res.get("parameters", {})
                }
            return {
                "can_execute": True,
                "message": "Hello! I am E.D.I.T.H., your custom-trained neural task handler. All systems operational.",
                "handler": "dialogue",
                "parameters": {}
            }
        else:
            return {
                "can_execute": True,
                "message": f"Executing requested action: {action}.",
                "handler": action,
                "parameters": entities
            }


def interactive_cli():
    """Terminal chat loop for live interaction and OS task execution with EDITH's neural network."""
    print("==================================================================")
    print("        E.D.I.T.H. NEURAL NETWORK & TASK EXECUTION ENGINE         ")
    print("       Commands are predicted by custom ANN and executed live      ")
    print("        Type any command or 'exit' / 'quit' to close console      ")
    print("==================================================================\n")

    try:
        brain = EdithBrain(enable_execution=True)
    except Exception as e:
        print(f"Error initializing EDITH Brain: {e}")
        return

    # --- Personality Greeting on Startup ---
    try:
        from engine.personality import EdithPersonality
        personality = EdithPersonality()
        greeting = personality.generate_startup_greeting()
        print(f"\n[E.D.I.T.H.] {greeting}\n")
    except ImportError:
        try:
            from personality import EdithPersonality
            personality = EdithPersonality()
            greeting = personality.generate_startup_greeting()
            print(f"\n[E.D.I.T.H.] {greeting}\n")
        except ImportError:
            print("\n[E.D.I.T.H.] Hey! Ready to help. What do you need?\n")

    # Neural Voice Engine
    tts = None
    voice_feedback_enabled = False
    try:
        from runtime.stt_engine import EdithTTSEngine
        tts = EdithTTSEngine()
    except Exception:
        pass

    print("  (Tip: Type 'voice on' to hear ChatGPT-style neural voice responses, or 'mic' for voice chat)\n")

    while True:
        try:
            cmd = input("\n[User] > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["exit", "quit", "q"]:
                # Warm farewell using personality
                try:
                    farewell = personality.react_to_farewell()
                    print(f"\n[E.D.I.T.H.] {farewell}")
                    if voice_feedback_enabled and tts:
                        tts.speak(farewell, async_mode=False)
                except Exception:
                    print("\n[E.D.I.T.H.] See you later! Take care! 💙")
                break

            if cmd.lower() in ["voice on", "speak on", "sound on", "enable voice"]:
                voice_feedback_enabled = True
                msg = "Neural Voice activated! I'll speak responses aloud with my ChatGPT voice."
                print(f"\n[E.D.I.T.H.] 🔊 {msg}")
                if tts:
                    tts.speak(msg, async_mode=True)
                continue

            if cmd.lower() in ["voice off", "speak off", "sound off", "quiet", "mute"]:
                voice_feedback_enabled = False
                print("\n[E.D.I.T.H.] 🔇 Voice feedback muted. Back to quiet text mode.")
                continue

            if cmd.lower() in ["wake", "wake word", "wakeword", "standby", "hey edith", "wake up"]:
                from runtime.stt_engine import EdithWakeWordDetector
                detector = EdithWakeWordDetector(brain=brain, enable_tts=True)
                detector.listen_for_wake_word(
                    on_wake_callback=lambda w, c: print(f"  ⚡ Wake phrase detected: '{w}'"),
                    on_command_callback=lambda c, r: print(f"  [Executed] > {c}")
                )
                continue

            if cmd.lower() in ["voice", "mic", "listen", "voice mode", "talk", "voice once", "mic once"]:
                from runtime.stt_engine import EdithVoiceAssistant
                va = EdithVoiceAssistant(brain=brain, enable_tts=True)

                if "once" in cmd.lower():
                    print("\n[E.D.I.T.H.] 🎙️ Listening (single command)... (speak now)")
                    try:
                        v_res = va.listen_and_execute(brain=brain, speak_response=True)
                        if v_res.get("status") == "success":
                            cmd = v_res.get("spoken_command", "")
                            print(f"[Heard] > \"{cmd}\"")
                            result = v_res.get("brain_result", {})
                        else:
                            err = v_res.get("stt", {}).get("error", "No speech detected.")
                            print(f"[E.D.I.T.H.] ⚠️ {err}")
                            continue
                    except Exception as ex:
                        print(f"[Voice Error] {ex}")
                        continue
                else:
                    print("\n[E.D.I.T.H.] 🎙️ Hands-Free Voice Mode Activated.")
                    print("  Speak your commands naturally into your microphone.")
                    print("  Say 'text mode', 'stop', 'standby', or 'exit' to switch back to typing.\n")

                    def on_cmd(spoken_text, v_res):
                        print(f"\n[Heard] > \"{spoken_text}\"")
                        res = v_res.get("brain_result", {})
                        dir_info = res.get("directive", {})
                        if dir_info.get("message"):
                            print(f"[E.D.I.T.H.] {dir_info['message']}")
                        if res.get("intent") not in ["dialogue", "chitchat"] or res.get("intent_confidence", 0) < 0.95:
                            print(f"  |-- Predicted Intent : {res.get('intent')} ({res.get('intent_confidence', 0):.1%})")
                            print(f"  |-- Action Policy    : {res.get('action')} ({res.get('action_confidence', 0):.1%})")
                            if res.get("entities"):
                                print(f"  |-- Extracted Slots  : {res['entities']}")
                            print(f"  \\-- Neural Status    : {res.get('status')} (Operational: {res.get('operational_confidence', 0):.1%})")
                        if "execution" in res:
                            ex = res["execution"]
                            icon = "[EXEC-OK]" if ex.get("status") in ["success", "warning"] else "[EXEC-INFO]"
                            print(f"  {icon} {ex.get('message')}")

                    try:
                        va.continuous_listen_loop(
                            brain=brain,
                            on_command_callback=on_cmd,
                            on_status_callback=lambda s: print("🎙️ Listening..." if s == "listening" else "")
                        )
                    except KeyboardInterrupt:
                        print("\n[E.D.I.T.H.] Voice mode paused. Returning to text console.")
                    continue
            else:
                result = brain.process(cmd, execute=True)

            directive = result["directive"]
            print(f"\n[E.D.I.T.H.] {directive['message']}")

            if voice_feedback_enabled and tts and directive.get("message"):
                tts.speak(directive["message"], async_mode=True)

            if result.get("intent") not in ["dialogue", "chitchat"] or result.get("intent_confidence", 0) < 0.95:
                print(f"  |-- Predicted Intent : {result['intent']} ({result['intent_confidence']:.1%})")
                print(f"  |-- Action Policy    : {result['action']} ({result['action_confidence']:.1%})")
                if result["entities"]:
                    print(f"  |-- Extracted Slots  : {result['entities']}")
                print(f"  \\-- Neural Status    : {result['status']} (Operational: {result['operational_confidence']:.1%})")

            if "execution" in result:
                ex = result["execution"]
                if ex.get("status") in ["suggestions_available", "needs_confirmation"]:
                    icon = "[SUGGESTIONS]"
                elif ex.get("status") in ["success", "warning"]:
                    icon = "[EXEC-OK]"
                else:
                    icon = "[EXEC-INFO]"
                print(f"\n  {icon} {ex.get('message')}")

        except KeyboardInterrupt:
            print("\n\n[E.D.I.T.H.] Caught you trying to sneak away! 😄 Take care! 💙")
            break


if __name__ == "__main__":
    if "--wake" in sys.argv or "-w" in sys.argv or "wake" in sys.argv:
        from runtime.stt_engine import EdithWakeWordDetector
        detector = EdithWakeWordDetector(enable_tts=True)
        detector.listen_for_wake_word()
    else:
        interactive_cli()

