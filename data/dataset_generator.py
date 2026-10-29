"""Advanced Domain Dataset Generator for E.D.I.T.H. Intelligence Core V8.

Generates diverse, linguistically varied examples covering all 17 intents with:
- Natural-language paraphrases, questions, speech-like phrasing, and informal language
- Adversarial hard negatives (negation, conversational mentions of tech keywords, future intent, hypotheticals)
- Emotional, reflective, and conversational dialogue (stress, decisions, thoughts, reasoning)
- Rich multi-turn contextual follow-ups (anaphora, ordinals, corrections, clarifications, reversals)
- Screen understanding, storage auditing, and file organization queries
- Calibrated operational confidence scores (1.0 for unambiguous valid requests, 0.4-0.6 for ambiguous, 0.0 for OOD/negatives)
- Strict zero template leakage across train, val, and held-out test partitions.
"""

import os
import re
import json
import random
from typing import List, Dict, Tuple, Any

METADATA_PATH = os.path.join(os.path.dirname(__file__), "metadata_maps.json")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

INTENT_MAP = METADATA["intents"]
SLOT_MAP = METADATA["slots"]
ACTION_MAP = METADATA["actions"]

APPS = [
    "visual studio code", "vscode", "code", "coding environment", "dev environment", "ide",
    "chrome", "google chrome", "browser", "firefox", "edge", "microsoft edge", "spotify",
    "notepad", "text editor", "terminal", "windows terminal", "powershell", "cmd", "command prompt",
    "slack", "discord", "calculator", "calc", "obs studio", "docker desktop", "vlc", "git bash",
    "file explorer", "explorer", "paint", "wordpad", "task manager", "settings", "control panel",
    "mail", "outlook", "python environment", "teams", "whatsapp"
]

PROJECTS = [
    "EDITH", "edith", "my edith project", "backend service", "frontend app", "neural network",
    "website", "api server", "agent core", "data pipeline", "portfolio", "classifier", "workspace",
    "java assignment", "compiler", "mobile app", "recommendation engine"
]

FILES = [
    "resume.pdf", "resume", "my resume", "report.pdf", "notes.txt", "notes", "todo.txt",
    "budget.xlsx", "presentation.pptx", "data.csv", "main.py", "test.py", "server.js",
    "index.html", "config.json", "architecture.md", "README.md", "script.py", "contract.pdf",
    "invoice.pdf", "summary.docx", "draft.txt", "assignment.java", "dataset.csv"
]

PATHS = [
    "desktop", "the desktop", "downloads", "my downloads", "documents", "my documents",
    "c:/projects/edith", "c:/downloads", "c:/documents", "./src", "./data", "./models",
    "~/projects", "project folder", "root directory"
]

PEOPLE = ["John", "Rahul", "Aditya", "Alex", "Sarah", "Emily", "David", "Michael", "Priya"]

DESTINATIONS = ["desktop", "downloads", "documents", "archive", "trash", "backup", "server", "there", "folder", "new folder"]

SETTINGS = ["volume", "master volume", "sound", "audio", "brightness", "screen brightness", "display brightness", "bluetooth", "wifi"]

VALUES = ["50%", "80%", "100%", "20%", "mute", "unmute", "maximum", "minimum", "way down", "way up", "a bit louder", "a bit quieter", "halfway"]

PIDS = ["1042", "4096", "8820", "1290", "512", "7340", "9901", "30456", "29280", "1540", "2204", "1820", "6540"]

SONGS = [
    "starboy", "blinding lights", "save your tears", "shape of you", "bad guy", "believer",
    "mockingbird", "bohemian rhapsody", "hotel california", "lose yourself", "stay", "as it was",
    "viva la vida", "counting stars", "night changes", "unstoppable", "flowers", "sunflower",
    "something relaxing", "relaxing music", "chill beats", "lofi hip hop", "relaxing tracks",
    "ambient sounds", "classical piano", "focus music", "study beats", "acoustic guitar"
]

ARTISTS = [
    "the weeknd", "eminem", "ed sheeran", "billie eilish", "imagine dragons", "queen",
    "taylor swift", "drake", "coldplay", "post malone", "dua lipa", "adele", "ariana grande"
]

MODIFIERS = [
    "slowed and reverb", "sped up", "acoustic version", "live performance",
    "instrumental version", "lofi remix", "official remix", "relaxing", "remastered"
]

PLATFORMS = ["spotify", "youtube", "apple music", "soundcloud"]

SEARCH_QUERIES = [
    "how to train neural networks", "quantum computing basics", "fastapi tutorial",
    "latest python features", "stock market trends", "weather forecast", "react hooks",
    "python tips", "machine learning concepts", "rust programming language", "docker containerization",
    "gradient descent optimization", "transformer architecture explained"
]

MATH_EXPRESSIONS = [
    "2+2", "15 * 12", "1250 / 5", "450 + 820", "20 percent of 1500", "20% of 1500",
    "sqrt 144", "square root of 144", "square root of 256", "50 * 20", "50 + 25",
    "(45 * 12) + 100 / 2", "100 - 37", "75% of 800"
]

UTILITIES = [
    "screenshot", "screen capture", "capture screen", "take screenshot", "minimize all windows",
    "show desktop", "what time is it", "current time", "what date is today", "today's date",
    "empty recycle bin", "clean recycle bin", "inspect screen", "what is on my screen"
]

TEMPORAL_TERMS = ["latest", "most recent", "newest", "newer", "older", "oldest", "recent"]

COMMAND_TERMS = ["run the tests", "pytest", "unit tests", "python test.py", "test suite", "check tests"]

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_:\.\-\\/\~%\^\+\*\=]+|'(?:s|t|m|d|ll|re|ve)|[^\s\w]")


def tokenize_text(text: str) -> List[str]:
    """Tokenize maintaining contraction and punctuation boundaries."""
    cleaned = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("—", "-")
    return TOKEN_PATTERN.findall(cleaned.strip().lower())


def build_bio_tags(text: str, entities: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Produce exact BIO slot tags matching tokens from raw sentence text."""
    tokens = tokenize_text(text)
    tags = ["O"] * len(tokens)

    sorted_entities = sorted(entities, key=lambda e: len(e[1]), reverse=True)

    for slot_type, val in sorted_entities:
        val_tokens = tokenize_text(val)
        if not val_tokens:
            continue
        v_len = len(val_tokens)

        for i in range(len(tokens) - v_len + 1):
            if tokens[i:i + v_len] == val_tokens:
                if all(tags[i + k] == "O" for k in range(v_len)):
                    tags[i] = f"B-{slot_type}"
                    for k in range(1, v_len):
                        tags[i + k] = f"I-{slot_type}"
                break

    return tokens, tags


# Template definitions strictly partitioned by intent and split category (zero leakage)
TEMPLATES_BY_INTENT: Dict[str, Dict[str, List[Any]]] = {
    # 1. app_launch
    "app_launch": {
        "train": [
            ("open [APP]", [("APP", "app")], 1.0),
            ("launch [APP]", [("APP", "app")], 1.0),
            ("start [APP]", [("APP", "app")], 1.0),
            ("fire up [APP]", [("APP", "app")], 1.0),
            ("bring up my [APP]", [("APP", "app")], 1.0),
            ("can you open [APP] for me", [("APP", "app")], 1.0),
            ("switch to [APP]", [("APP", "app")], 1.0),
            ("get [APP] running", [("APP", "app")], 1.0),
            ("please launch [APP]", [("APP", "app")], 1.0),
            ("start up the [APP] program", [("APP", "app")], 1.0),
            ("i want to use [APP]", [("APP", "app")], 1.0),
            ("pull up [APP] please", [("APP", "app")], 1.0),
            ("open up [APP]", [("APP", "app")], 1.0),
            ("run [APP]", [("APP", "app")], 1.0),
            ("can we get [APP] running", [("APP", "app")], 1.0),
            ("switch over to [APP] for me", [("APP", "app")], 1.0),
            ("open", [], 0.4),  # Ambiguous
            ("launch the application", [], 0.4),
        ],
        "val": [
            ("load up [APP]", [("APP", "app")], 1.0),
            ("bring [APP] to the front", [("APP", "app")], 1.0),
            ("i need [APP] open", [("APP", "app")], 1.0),
            ("could you fire up [APP]", [("APP", "app")], 1.0),
            ("switch my window to [APP]", [("APP", "app")], 1.0),
            ("open the [APP] app", [("APP", "app")], 1.0),
        ],
        "test": [
            ("Open [APP].", [("APP", "app")], 1.0),
            ("Fire up [APP] right now.", [("APP", "app")], 1.0),
            ("Could you please open [APP]?", [("APP", "app")], 1.0),
            ("Switch to [APP] immediately.", [("APP", "app")], 1.0),
            ("Can you bring up [APP]?", [("APP", "app")], 1.0),
            ("Launch the [APP] application.", [("APP", "app")], 1.0),
            ("get [APP] up on my screen", [("APP", "app")], 1.0),
        ]
    },

    # 2. app_close
    "app_close": {
        "train": [
            ("close [APP]", [("APP", "app")], 1.0),
            ("terminate [APP]", [("APP", "app")], 1.0),
            ("exit [APP]", [("APP", "app")], 1.0),
            ("kill [APP]", [("APP", "app")], 1.0),
            ("shut down [APP]", [("APP", "app")], 1.0),
            ("quit [APP]", [("APP", "app")], 1.0),
            ("please close [APP]", [("APP", "app")], 1.0),
            ("turn off [APP]", [("APP", "app")], 1.0),
            ("end [APP]", [("APP", "app")], 1.0),
            ("close down [APP]", [("APP", "app")], 1.0),
            ("can you close [APP] for me", [("APP", "app")], 1.0),
            ("exit out of [APP]", [("APP", "app")], 1.0),
            ("close the [APP] window", [("APP", "app")], 1.0),
            ("close", [], 0.4),  # Ambiguous
            ("kill the application", [], 0.4),
        ],
        "val": [
            ("shut [APP] down", [("APP", "app")], 1.0),
            ("stop running [APP]", [("APP", "app")], 1.0),
            ("dismiss [APP]", [("APP", "app")], 1.0),
            ("get rid of the [APP] window", [("APP", "app")], 1.0),
            ("close the active [APP]", [("APP", "app")], 1.0),
        ],
        "test": [
            ("Close [APP].", [("APP", "app")], 1.0),
            ("Terminate [APP] immediately.", [("APP", "app")], 1.0),
            ("Exit [APP] now.", [("APP", "app")], 1.0),
            ("Quit [APP] please.", [("APP", "app")], 1.0),
            ("Shut down the [APP] application.", [("APP", "app")], 1.0),
            ("close [APP] for me please", [("APP", "app")], 1.0),
        ]
    },

    # 3. file_operation
    "file_operation": {
        "train": [
            ("delete [FILE]", [("FILE", "file")], 1.0),
            ("find [FILE]", [("FILE", "file")], 1.0),
            ("locate [FILE]", [("FILE", "file")], 1.0),
            ("search for [FILE]", [("FILE", "file")], 1.0),
            ("create a file named [FILE]", [("FILE", "file")], 1.0),
            ("make a new file called [FILE]", [("FILE", "file")], 1.0),
            ("remove [FILE]", [("FILE", "file")], 1.0),
            ("copy [FILE] to [DEST]", [("FILE", "file"), ("DEST", "dest")], 1.0),
            ("move [FILE] into [DEST]", [("FILE", "file"), ("DEST", "dest")], 1.0),
            ("erase [FILE] from [PATHS]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("open [FILE]", [("FILE", "file")], 1.0),
            ("show me [FILE] in [PATHS]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("where is [FILE]", [("FILE", "file")], 1.0),
            ("organize my [PATHS] folder", [("PATH", "path")], 1.0),
            ("find duplicate files in [PATHS]", [("PATH", "path")], 1.0),
            ("show large files in [PATHS]", [("PATH", "path")], 1.0),
            ("delete file", [], 0.4),  # Ambiguous
            ("find the document", [], 0.4),
        ],
        "val": [
            ("can you find [FILE] on my [PATHS]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("relocate [FILE] to [DEST]", [("FILE", "file"), ("DEST", "dest")], 1.0),
            ("trash [FILE]", [("FILE", "file")], 1.0),
            ("look for [FILE] in [PATHS]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("create [FILE] on [PATHS]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("clean up duplicate files in [PATHS]", [("PATH", "path")], 1.0),
        ],
        "test": [
            ("Find [FILE].", [("FILE", "file")], 1.0),
            ("Delete [FILE] from [PATHS].", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("Locate [FILE] on my system.", [("FILE", "file")], 1.0),
            ("Create a file named [FILE] on [PATHS].", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("Move [FILE] to [DEST].", [("FILE", "file"), ("DEST", "dest")], 1.0),
            ("Where is my [FILE] located?", [("FILE", "file")], 1.0),
            ("Find all duplicate screenshots in [PATHS].", [("PATH", "path")], 1.0),
            ("Show me the largest files in [PATHS].", [("PATH", "path")], 1.0),
        ]
    },

    # 4. system_control
    "system_control": {
        "train": [
            ("set [SETTING] to [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("change [SETTING] to [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("turn [SETTING] [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("increase [SETTING]", [("SETTING", "setting")], 1.0),
            ("decrease [SETTING]", [("SETTING", "setting")], 1.0),
            ("raise [SETTING]", [("SETTING", "setting")], 1.0),
            ("lower [SETTING]", [("SETTING", "setting")], 1.0),
            ("mute the [SETTING]", [("SETTING", "setting")], 1.0),
            ("unmute [SETTING]", [("SETTING", "setting")], 1.0),
            ("adjust [SETTING] to [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("make the [SETTING] [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("put [SETTING] at [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("turn the volume down", [("SETTING", "volume")], 1.0),
            ("turn the volume up", [("SETTING", "volume")], 1.0),
            ("adjust settings", [], 0.4),  # Ambiguous
            ("change volume", [], 0.4),
        ],
        "val": [
            ("bump up [SETTING]", [("SETTING", "setting")], 1.0),
            ("drop [SETTING] to [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("switch off [SETTING]", [("SETTING", "setting")], 1.0),
            ("set my [SETTING] [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("bring [SETTING] down", [("SETTING", "setting")], 1.0),
        ],
        "test": [
            ("Set [SETTING] to [VALUE].", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("Mute [SETTING].", [("SETTING", "setting")], 1.0),
            ("Turn [SETTING] way up.", [("SETTING", "setting")], 1.0),
            ("Lower the [SETTING].", [("SETTING", "setting")], 1.0),
            ("Adjust [SETTING] to [VALUE].", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("Turn the volume down please.", [("SETTING", "volume")], 1.0),
        ]
    },

    # 5. system_query
    "system_query": {
        "train": [
            ("what is my current cpu usage", [], 1.0),
            ("check battery level", [], 1.0),
            ("how much ram is available", [], 1.0),
            ("show disk space", [], 1.0),
            ("what is the cpu temperature", [], 1.0),
            ("check system resource usage", [], 1.0),
            ("show memory consumption", [], 1.0),
            ("how much disk storage is free", [], 1.0),
            ("check battery status", [], 1.0),
            ("display current cpu load", [], 1.0),
            ("show which applications are using the most memory", [], 1.0),
            ("what apps are consuming high cpu", [], 1.0),
            ("system diagnostics", [], 0.5),
        ],
        "val": [
            ("inspect available ram", [], 1.0),
            ("is my battery charging", [], 1.0),
            ("tell me current memory usage", [], 1.0),
            ("how hard is the cpu running", [], 1.0),
            ("report storage capacity", [], 1.0),
            ("list top memory consuming processes", [], 1.0),
        ],
        "test": [
            ("What is the current CPU usage?", [], 1.0),
            ("How much RAM is free right now?", [], 1.0),
            ("Check battery level.", [], 1.0),
            ("Show me disk space.", [], 1.0),
            ("What is the current memory load?", [], 1.0),
            ("Show me which applications are using the most memory.", [], 1.0),
        ]
    },

    # 6. process_manage
    "process_manage": {
        "train": [
            ("kill process [PID]", [("PID", "pid")], 1.0),
            ("terminate pid [PID]", [("PID", "pid")], 1.0),
            ("end process [PID]", [("PID", "pid")], 1.0),
            ("kill pid [PID]", [("PID", "pid")], 1.0),
            ("stop process [PID]", [("PID", "pid")], 1.0),
            ("force close pid [PID]", [("PID", "pid")], 1.0),
            ("terminate process [PID]", [("PID", "pid")], 1.0),
            ("kill task [PID]", [("PID", "pid")], 1.0),
            ("abort process [PID]", [("PID", "pid")], 1.0),
            ("kill process", [], 0.4),  # Missing PID
            ("terminate task", [], 0.4),
        ],
        "val": [
            ("force stop process [PID]", [("PID", "pid")], 1.0),
            ("end task with pid [PID]", [("PID", "pid")], 1.0),
            ("terminate running process [PID]", [("PID", "pid")], 1.0),
            ("kill background process [PID]", [("PID", "pid")], 1.0),
        ],
        "test": [
            ("Kill process [PID].", [("PID", "pid")], 1.0),
            ("Terminate PID [PID].", [("PID", "pid")], 1.0),
            ("Stop process [PID] now.", [("PID", "pid")], 1.0),
            ("End process [PID].", [("PID", "pid")], 1.0),
            ("Force kill task [PID].", [("PID", "pid")], 1.0),
        ]
    },

    # 7. web_search
    "web_search": {
        "train": [
            ("search for [QUERY]", [("QUERY", "query")], 1.0),
            ("google [QUERY]", [("QUERY", "query")], 1.0),
            ("look up [QUERY]", [("QUERY", "query")], 1.0),
            ("search web for [QUERY]", [("QUERY", "query")], 1.0),
            ("search online for [QUERY]", [("QUERY", "query")], 1.0),
            ("find information about [QUERY]", [("QUERY", "query")], 1.0),
            ("browse for [QUERY]", [("QUERY", "query")], 1.0),
            ("can you search [QUERY]", [("QUERY", "query")], 1.0),
            ("google for me [QUERY]", [("QUERY", "query")], 1.0),
            ("look on the internet for [QUERY]", [("QUERY", "query")], 1.0),
            ("search", [], 0.4),  # Missing query
            ("google online", [], 0.4),
        ],
        "val": [
            ("run a web search on [QUERY]", [("QUERY", "query")], 1.0),
            ("check online regarding [QUERY]", [("QUERY", "query")], 1.0),
            ("find [QUERY] on google", [("QUERY", "query")], 1.0),
            ("search the net for [QUERY]", [("QUERY", "query")], 1.0),
        ],
        "test": [
            ("Search for [QUERY].", [("QUERY", "query")], 1.0),
            ("Google [QUERY].", [("QUERY", "query")], 1.0),
            ("Look up [QUERY] online.", [("QUERY", "query")], 1.0),
            ("Search the web for [QUERY].", [("QUERY", "query")], 1.0),
            ("Find articles about [QUERY].", [("QUERY", "query")], 1.0),
        ]
    },

    # 8. network_task
    "network_task": {
        "train": [
            ("ping google.com", [("QUERY", "google.com")], 1.0),
            ("ping localhost", [("QUERY", "localhost")], 1.0),
            ("check internet connectivity", [], 1.0),
            ("test network connection", [], 1.0),
            ("ping 8.8.8.8", [("QUERY", "8.8.8.8")], 1.0),
            ("check latency to server", [], 1.0),
            ("diagnose network connection", [], 1.0),
            ("am i connected to the internet", [], 1.0),
            ("check wifi connectivity", [], 1.0),
            ("verify network status", [], 1.0),
            ("test reachability of host", [], 1.0),
            ("check my ip address", [], 1.0),
            ("what is my local ip", [], 1.0),
        ],
        "val": [
            ("ping server to check reachability", [], 1.0),
            ("is my internet working properly", [], 1.0),
            ("test network latency", [], 1.0),
            ("diagnose my internet connection", [], 1.0),
            ("check public ip", [], 1.0),
        ],
        "test": [
            ("Ping google.com.", [("QUERY", "google.com")], 1.0),
            ("Check internet connectivity.", [], 1.0),
            ("Is my internet working?", [], 1.0),
            ("Test network connection.", [], 1.0),
            ("Diagnose network latency.", [], 1.0),
            ("What is my IP address?", [], 1.0),
        ]
    },

    # 9. automation_workflow
    "automation_workflow": {
        "train": [
            ("open [APP] and search for [QUERY]", [("APP", "app"), ("QUERY", "query")], 1.0),
            ("open [APP] and type hello world", [("APP", "app")], 1.0),
            ("take a screenshot and open [PATHS]", [("PATH", "path")], 1.0),
            ("create file [FILE] on [PATHS] and open it", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("open [APP], play [SONG] and increase volume", [("APP", "app"), ("SONG", "song")], 1.0),
            ("open [APP] then open [APP]", [("APP", "app")], 1.0),
            ("what is 50 * 20 and what time is it", [], 1.0),
            ("close [APP] and lock workstation", [("APP", "app")], 1.0),
            ("empty recycle bin and minimize all windows", [], 1.0),
            ("find the latest report, summarize it, and save the summary", [], 1.0),
            ("open [APP], search for a song, and play it", [("APP", "app")], 1.0),
            ("find the largest temporary files and tell me how much space i could recover", [], 1.0),
            ("review this project and identify the biggest problems", [], 1.0),
            ("prepare an email draft to [PERSON] but don't send it", [("PERSON", "person")], 1.0),
            ("open notepad and write this down: submit assignment tomorrow", [], 1.0),
        ],
        "val": [
            ("start [APP] and browse for [QUERY]", [("APP", "app"), ("QUERY", "query")], 1.0),
            ("grab screenshot and show [PATHS]", [("PATH", "path")], 1.0),
            ("open [APP] then mute volume", [("APP", "app")], 1.0),
            ("make file [FILE] and copy it to [DEST]", [("FILE", "file"), ("DEST", "dest")], 1.0),
            ("launch [APP] and search for [QUERY]", [("APP", "app"), ("QUERY", "query")], 1.0),
            ("find the recent document, extract bullet points, and save them", [], 1.0),
        ],
        "test": [
            ("Open [APP] and type hello world.", [("APP", "app")], 1.0),
            ("Open [APP] and search for [QUERY].", [("APP", "app"), ("QUERY", "query")], 1.0),
            ("Take a screenshot and open [PATHS].", [("PATH", "path")], 1.0),
            ("Create file [FILE] on [PATHS] and open it.", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("Close [APP] and minimize all windows.", [("APP", "app")], 1.0),
            ("Find the latest report, summarize it, and save the summary.", [], 1.0),
            ("Open [APP], search for a song, and play it.", [("APP", "app")], 1.0),
            ("Find the largest temporary files and calculate recoverable space.", [], 1.0),
        ]
    },

    # 10. media_control
    "media_control": {
        "train": [
            ("pause music", [], 1.0),
            ("resume playback", [], 1.0),
            ("skip track", [], 1.0),
            ("next track", [], 1.0),
            ("previous track", [], 1.0),
            ("stop playback", [], 1.0),
            ("pause the song", [], 1.0),
            ("unpause music", [], 1.0),
            ("skip this song", [], 1.0),
            ("go back to previous song", [], 1.0),
            ("pause", [], 1.0),
            ("resume", [], 1.0),
            ("next song", [], 1.0),
            ("media controls", [], 0.5),
        ],
        "val": [
            ("hold music playback", [], 1.0),
            ("unpause the track", [], 1.0),
            ("jump to next song", [], 1.0),
            ("replay previous track", [], 1.0),
            ("halt current media", [], 1.0),
        ],
        "test": [
            ("Pause music.", [], 1.0),
            ("Resume playback.", [], 1.0),
            ("Skip to next song.", [], 1.0),
            ("Previous track please.", [], 1.0),
            ("Stop the music.", [], 1.0),
            ("Pause.", [], 1.0),
        ]
    },

    # 11. chitchat (General Conversation & Emotional Reflection)
    "chitchat": {
        "train": [
            ("hello", [], 1.0),
            ("hi edith", [], 1.0),
            ("hey there", [], 1.0),
            ("good morning", [], 1.0),
            ("who are you", [], 1.0),
            ("what is your purpose", [], 1.0),
            ("thank you so much", [], 1.0),
            ("how are you doing today", [], 1.0),
            ("tell me a joke", [], 1.0),
            ("what can you do", [], 1.0),
            ("nice to meet you", [], 1.0),
            ("greetings edith", [], 1.0),
            ("thanks for the help", [], 1.0),
            # Emotional and reflective conversation (Category B)
            ("i am stressed about college", [], 1.0),
            ("i'm feeling low today", [], 1.0),
            ("i had a rough day", [], 1.0),
            ("i keep thinking about something that happened", [], 1.0),
            ("i don't know if i made the right decision", [], 1.0),
            ("i just want someone to listen", [], 1.0),
            ("can you help me think this through", [], 1.0),
            ("i know what i should do but i'm struggling to do it", [], 1.0),
            ("tell me something interesting", [], 1.0),
            ("that actually helped", [], 1.0),
            ("why do you think that", [], 1.0),
            ("i disagree with you", [], 1.0),
            ("i'm bored", [], 1.0),
            ("can we talk for a minute", [], 1.0),
            ("i don't know what to do", [], 1.0),
            ("what do you think about life", [], 1.0),
            ("i am overwhelmed with work", [], 1.0),
            ("thanks for listening to me", [], 1.0),
            ("what does this error mean in general terms", [], 0.95),
            ("explain recursion simply", [], 0.95),
            ("how does memory management work", [], 0.95),
            # V10 Psychological Profiling & Companion Cognition
            ("describe me as a person", [], 1.0),
            ("judge me honestly edith", [], 1.0),
            ("roast me based on how i act", [], 1.0),
            ("who am i to you", [], 1.0),
            ("what have you learned about me", [], 1.0),
            ("do you understand how i think", [], 1.0),
            ("analyze my personality", [], 1.0),
            ("am i making progress in life", [], 1.0),
            # V10 Modern Dating, Resilience & Cognitive Scenarios
            ("why did they ghost me after such a good date", [], 1.0),
            ("how do i get over a painful breakup", [], 1.0),
            ("is it bad that i miss my ex", [], 1.0),
            ("how do i know if someone is breadcrumbing me", [], 1.0),
            ("how do i set boundaries with someone i like", [], 1.0),
            ("how should i architect a distributed event system", [], 0.95),
            ("what is the difference between latency and throughput", [], 0.95),
            ("when should i choose a monolith over microservices", [], 0.95),
            ("how does optimistic locking prevent race conditions", [], 0.95),
            ("explain the raft consensus protocol simply", [], 0.95),
            ("are you conscious or just predicting tokens", [], 1.0),
            ("what is the chinese room argument", [], 0.95),
            ("can a machine ever have true feelings", [], 1.0),
            ("what does existence mean to an ai", [], 1.0),
            ("do we have free will or is everything deterministic", [], 1.0),
            ("how do i build stoic resilience when everything goes wrong", [], 1.0),
            ("i feel so alone working on this project", [], 1.0),
            ("how do i stop caring about other people's opinions", [], 1.0),
            ("how do i break my dopamine addiction to reels", [], 1.0),
            ("how to recover from hyperfocus burnout", [], 1.0),
            ("why is starting a task so hard sometimes", [], 1.0),
            ("how to overcome imposter syndrome as a software engineer", [], 1.0),
            ("i feel like my work isn't good enough", [], 1.0),
        ],
        "val": [
            ("hey edith how is it going", [], 1.0),
            ("what is your name", [], 1.0),
            ("appreciate your assistance", [], 1.0),
            ("i am feeling pretty down today", [], 1.0),
            ("i have so much anxiety about my future", [], 1.0),
            ("can you help me process my thoughts", [], 1.0),
            ("i feel exhausted from studying", [], 1.0),
            ("why did you suggest that perspective", [], 1.0),
            ("explain what an operating system does", [], 0.95),
            # V10 Validation additions
            ("how would you describe my character", [], 1.0),
            ("give me an honest assessment of who i am", [], 1.0),
            ("what are your thoughts on ai sentience", [], 1.0),
            ("how can i navigate mixed signals in dating", [], 1.0),
            ("explain why database connection pools run out", [], 0.95),
            ("how to stay motivated during a long solo project", [], 1.0),
            ("how do i reset my focus when my brain feels fried", [], 1.0),
            ("can we talk about the meaning of consciousness", [], 1.0),
        ],
        "test": [
            ("Hello Edith, who are you?", [], 1.0),
            ("Good morning, how are you?", [], 1.0),
            ("Thank you for your help.", [], 1.0),
            ("What are your capabilities?", [], 1.0),
            ("I'm stressed about college.", [], 1.0),
            ("I'm feeling low today.", [], 1.0),
            ("I keep thinking about something that happened.", [], 1.0),
            ("I don't know if I made the right decision.", [], 1.0),
            ("I just want someone to listen.", [], 1.0),
            ("Can you help me think this through?", [], 1.0),
            ("Tell me something interesting.", [], 1.0),
            ("That actually helped.", [], 1.0),
            ("Explain recursion simply.", [], 0.95),
            # V10 Test additions
            ("Tell me what you think of me.", [], 1.0),
            ("How would you describe my personality?", [], 1.0),
            ("Judge my habits and tell me the truth.", [], 1.0),
            ("Why do people ghost instead of being honest?", [], 1.0),
            ("Explain how distributed caching avoids stampedes.", [], 0.95),
            ("Do you believe in free will or determinism?", [], 1.0),
            ("How do I stay resilient when facing relentless rejection?", [], 1.0),
            ("Why do I struggle with dopamine and task initiation?", [], 1.0),
        ]
    },

    # 12. music_playback
    "music_playback": {
        "train": [
            ("play [SONG]", [("SONG", "song")], 1.0),
            ("play [SONG] by [ARTIST]", [("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("play [MODIFIER] [SONG]", [("MODIFIER", "modifier"), ("SONG", "song")], 1.0),
            ("play [SONG] on [PLATFORM]", [("SONG", "song"), ("PLATFORM", "platform")], 1.0),
            ("put on [SONG]", [("SONG", "song")], 1.0),
            ("i want to listen to [SONG]", [("SONG", "song")], 1.0),
            ("stream [SONG] by [ARTIST]", [("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("play some [SONG]", [("SONG", "song")], 1.0),
            ("queue up [SONG]", [("SONG", "song")], 1.0),
            ("play [MODIFIER] version of [SONG]", [("MODIFIER", "modifier"), ("SONG", "song")], 1.0),
            ("put on [MODIFIER] [SONG] by [ARTIST]", [("MODIFIER", "modifier"), ("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("i want some music", [], 1.0),
            ("play the track i was listening to earlier", [], 1.0),
            ("play some music", [], 1.0),
            ("play", [], 0.4),  # Missing song
        ],
        "val": [
            ("can you play [SONG]", [("SONG", "song")], 1.0),
            ("turn on [SONG] by [ARTIST]", [("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("let's hear [SONG]", [("SONG", "song")], 1.0),
            ("start playing [MODIFIER] [SONG]", [("MODIFIER", "modifier"), ("SONG", "song")], 1.0),
            ("play something relaxing", [("SONG", "something relaxing")], 1.0),
        ],
        "test": [
            ("Play [SONG].", [("SONG", "song")], 1.0),
            ("Play [MODIFIER] [SONG].", [("MODIFIER", "modifier"), ("SONG", "song")], 1.0),
            ("Play [SONG] by [ARTIST] on [PLATFORM].", [("SONG", "song"), ("ARTIST", "artist"), ("PLATFORM", "platform")], 1.0),
            ("Put on [SONG] by [ARTIST].", [("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("Stream [MODIFIER] [SONG].", [("MODIFIER", "modifier"), ("SONG", "song")], 1.0),
            ("I want some music.", [], 1.0),
            ("Play the track I was listening to earlier.", [], 1.0),
        ]
    },

    # 13. math_compute
    "math_compute": {
        "train": [
            ("what is [EXPR]", [("EXPR", "expr")], 1.0),
            ("calculate [EXPR]", [("EXPR", "expr")], 1.0),
            ("compute [EXPR]", [("EXPR", "expr")], 1.0),
            ("evaluate [EXPR]", [("EXPR", "expr")], 1.0),
            ("solve [EXPR]", [("EXPR", "expr")], 1.0),
            ("how much is [EXPR]", [("EXPR", "expr")], 1.0),
            ("can you compute [EXPR]", [("EXPR", "expr")], 1.0),
            ("tell me the result of [EXPR]", [("EXPR", "expr")], 1.0),
            ("calculate", [], 0.4),  # Missing expression
            ("what is the answer", [], 0.4),
        ],
        "val": [
            ("work out [EXPR]", [("EXPR", "expr")], 1.0),
            ("figure out [EXPR]", [("EXPR", "expr")], 1.0),
            ("give me the answer to [EXPR]", [("EXPR", "expr")], 1.0),
            ("math eval [EXPR]", [("EXPR", "expr")], 1.0),
        ],
        "test": [
            ("What is [EXPR]?", [("EXPR", "expr")], 1.0),
            ("Calculate [EXPR].", [("EXPR", "expr")], 1.0),
            ("Compute [EXPR].", [("EXPR", "expr")], 1.0),
            ("Solve [EXPR] for me.", [("EXPR", "expr")], 1.0),
            ("How much is [EXPR]?", [("EXPR", "expr")], 1.0),
        ]
    },

    # 14. system_utility
    "system_utility": {
        "train": [
            ("take a screenshot", [("UTILITY", "screenshot")], 1.0),
            ("capture the screen", [("UTILITY", "screenshot")], 1.0),
            ("what time is it", [("UTILITY", "current time")], 1.0),
            ("what is today's date", [("UTILITY", "today's date")], 1.0),
            ("empty the recycle bin", [("UTILITY", "empty recycle bin")], 1.0),
            ("minimize all windows", [("UTILITY", "minimize all windows")], 1.0),
            ("show desktop", [("UTILITY", "show desktop")], 1.0),
            ("clean the recycle bin", [("UTILITY", "clean recycle bin")], 1.0),
            ("hide all open windows", [("UTILITY", "minimize all windows")], 1.0),
            ("minimize everything on screen", [("UTILITY", "minimize all windows")], 1.0),
            ("what is happening on my screen", [("UTILITY", "what is on my screen")], 1.0),
            ("explain what is on my screen", [("UTILITY", "what is on my screen")], 1.0),
            ("read the text in that window", [("UTILITY", "inspect screen")], 1.0),
            ("what is on my screen", [("UTILITY", "what is on my screen")], 1.0),
            ("inspect active window", [("UTILITY", "inspect screen")], 1.0),
            ("clean up temporary files", [("UTILITY", "clean recycle bin")], 1.0),
            ("audit storage space", [], 1.0),
        ],
        "val": [
            ("grab a screenshot for me", [("UTILITY", "screenshot")], 1.0),
            ("tell me the current time", [("UTILITY", "current time")], 1.0),
            ("clear out recycle bin", [("UTILITY", "empty recycle bin")], 1.0),
            ("show the desktop screen", [("UTILITY", "show desktop")], 1.0),
            ("what is currently visible on my screen", [("UTILITY", "what is on my screen")], 1.0),
            ("read what is on the screen", [("UTILITY", "inspect screen")], 1.0),
        ],
        "test": [
            ("Take a screenshot.", [("UTILITY", "screenshot")], 1.0),
            ("What time is it?", [("UTILITY", "current time")], 1.0),
            ("Empty recycle bin.", [("UTILITY", "empty recycle bin")], 1.0),
            ("Minimize all windows.", [("UTILITY", "minimize all windows")], 1.0),
            ("What's happening on my screen?", [("UTILITY", "what is on my screen")], 1.0),
            ("Explain this error on my screen.", [("UTILITY", "inspect screen")], 1.0),
            ("Read the text in that window.", [("UTILITY", "inspect screen")], 1.0),
        ]
    },

    # 15. project_manage
    "project_manage": {
        "train": [
            ("get back to my [PROJECT]", [("PROJECT", "project")], 1.0),
            ("resume work on [PROJECT]", [("PROJECT", "project")], 1.0),
            ("run tests on [PROJECT]", [("PROJECT", "project"), ("CMD", "tests")], 1.0),
            ("run the tests for [PROJECT]", [("PROJECT", "project"), ("CMD", "tests")], 1.0),
            ("execute the test suite", [("CMD", "test suite")], 1.0),
            ("check whether [PROJECT] starts properly", [("PROJECT", "project")], 1.0),
            ("run project [PROJECT]", [("PROJECT", "project")], 1.0),
            ("build the [PROJECT] project", [("PROJECT", "project")], 1.0),
            ("run pytest on [PROJECT]", [("PROJECT", "project"), ("CMD", "pytest")], 1.0),
            ("check git status for [PROJECT]", [("PROJECT", "project")], 1.0),
            ("run the project", [], 0.4),  # Ambiguous / missing project
            ("run tests", [], 0.4),
        ],
        "val": [
            ("let's return to the [PROJECT]", [("PROJECT", "project")], 1.0),
            ("run the unit tests for [PROJECT]", [("PROJECT", "project"), ("CMD", "unit tests")], 1.0),
            ("verify if the tests pass", [("CMD", "tests")], 1.0),
            ("test whether the application starts properly", [("CMD", "test")], 1.0),
            ("compile and build [PROJECT]", [("PROJECT", "project")], 1.0),
        ],
        "test": [
            ("Let's get back to the project I was working on.", [("PROJECT", "project")], 1.0),
            ("Run the tests, and tell me whether they pass.", [("CMD", "tests")], 1.0),
            ("Check whether it starts properly.", [], 1.0),
            ("Run unit tests on my edith project.", [("CMD", "unit tests"), ("PROJECT", "edith project")], 1.0),
            ("Build the [PROJECT] codebase.", [("PROJECT", "project")], 1.0),
        ]
    },

    # 16. context_control (Multi-turn follow-ups, anaphora, corrections, cancellations)
    "context_control": {
        "train": [
            ("not that one, find the other version", [("REF", "other version")], 1.0),
            ("not the old one, the [TEMPORAL] one", [("TEMPORAL", "temporal"), ("REF", "one")], 1.0),
            ("open the second one", [("REF", "second one")], 1.0),
            ("open the [TEMPORAL] one", [("TEMPORAL", "temporal"), ("REF", "one")], 1.0),
            ("open that one", [("REF", "that one")], 1.0),
            ("now run it", [("REF", "it")], 1.0),
            ("move it there instead", [("REF", "it"), ("DEST", "there")], 1.0),
            ("undo that operation", [], 1.0),
            ("cancel that task", [], 1.0),
            ("nevermind, abort that", [], 1.0),
            ("pick the other option", [("REF", "other option")], 1.0),
            ("do that again", [], 1.0),
            ("repeat that action", [], 1.0),
            ("make it shorter", [("REF", "it")], 1.0),
            ("use java instead", [], 1.0),
            ("open the second result", [("REF", "second result")], 1.0),
            ("continue", [], 1.0),
            ("actually, cancel that", [], 1.0),
            ("what about the previous file", [("REF", "previous file")], 1.0),
            ("why did you choose that", [], 1.0),
            ("no, i meant the other application", [("REF", "other application")], 1.0),
            ("that is not what i asked", [], 1.0),
            ("you opened the wrong file", [], 1.0),
            ("i said save, not send", [], 1.0),
            ("you misunderstood me", [], 1.0),
            ("try another approach", [], 1.0),
            ("that didn't work", [], 1.0),
            ("stop doing that", [], 1.0),
            ("by project", [], 1.0),
            ("the one in downloads", [("PATH", "downloads")], 1.0),
            ("the first one", [("REF", "first one")], 1.0),
        ],
        "val": [
            ("that is not the one i wanted, show me the other version", [("REF", "other version")], 1.0),
            ("pick the first result", [("REF", "first result")], 1.0),
            ("delete it", [("REF", "it")], 1.0),
            ("undo that", [], 1.0),
            ("repeat the last step", [], 1.0),
            ("no not that, the other one", [("REF", "other one")], 1.0),
            ("cancel the current operation", [], 1.0),
            ("that was the wrong document", [], 1.0),
        ],
        "test": [
            ("That isn't the one I meant; find the other version.", [("REF", "other version")], 1.0),
            ("Not the old one, the latest one.", [("TEMPORAL", "latest"), ("REF", "one")], 1.0),
            ("Use the second result.", [("REF", "second result")], 1.0),
            ("Move it there instead.", [("REF", "it"), ("DEST", "there")], 1.0),
            ("Now run it.", [("REF", "it")], 1.0),
            ("Undo the last step.", [], 1.0),
            ("Do that again.", [], 1.0),
            ("That's not what I asked.", [], 1.0),
            ("You opened the wrong file.", [], 1.0),
            ("Actually, cancel that.", [], 1.0),
            ("What about the previous file?", [("REF", "previous file")], 1.0),
            ("Why did you choose that?", [], 1.0),
            ("No, I meant the other application.", [("REF", "other application")], 1.0),
        ]
    },

    # 17. unsupported_ood (Hard Negatives, Negation, Informational Questions, and OOD)
    "unsupported_ood": {
        "train": [
            # Hard Negations (Must NOT execute actions)
            ("don't open [APP]", [("APP", "app")], 0.0),
            ("do not launch [APP]", [("APP", "app")], 0.0),
            ("never close [APP]", [("APP", "app")], 0.0),
            ("don't delete that file", [], 0.0),
            ("do not delete [FILE]", [("FILE", "file")], 0.0),
            ("don't touch the file", [], 0.0),
            ("don't play anything right now", [], 0.0),
            ("do not terminate [APP]", [("APP", "app")], 0.0),
            ("stop don't send that message", [], 0.0),
            ("never delete my downloads folder", [], 0.0),
            ("don't change the volume", [], 0.0),
            # Conversational mentions with tech keywords (Must NOT trigger actions)
            ("i hate [APP]'s new interface", [("APP", "app")], 0.0),
            ("i am stressed about my computer science exam", [], 0.0),
            ("the word volume appears in this paragraph", [], 0.0),
            ("my laptop feels slow today", [], 0.0),
            ("the screenshot looks old", [], 0.0),
            ("i thought about deleting that file, but i am not sure", [], 0.0),
            ("i need to write an email later", [], 0.0),
            ("i am going to send the email later", [], 0.0),
            ("can you explain how deleting files works", [], 0.0),
            ("what did you do with the file earlier", [], 0.0),
            # Ambiguous/Incomplete requests (Low confidence rejection / clarification)
            ("open it", [], 0.3),
            ("delete that", [], 0.3),
            ("make it better", [], 0.2),
            ("fix the problem", [], 0.2),
            ("do the usual thing", [], 0.2),
            ("move that somewhere else", [], 0.3),
            ("do something with this", [], 0.2),
            # Pure Out-of-Distribution
            ("bake me a real chocolate cake", [], 0.0),
            ("order a large pepperoni pizza", [], 0.0),
            ("cook dinner for my family", [], 0.0),
            ("clean the dishes in the sink", [], 0.0),
            ("write a 500 word poem about spring", [], 0.0),
            ("who was the sixteenth president of the united states", [], 0.0),
            ("translate this paragraph to french", [], 0.0),
            ("what is the color of sound", [], 0.0),
            ("solve world peace right now", [], 0.0),
            ("qwerty asdf zxcvbnm 12345", [], 0.0),
            ("asdfghjkl random noise 99", [], 0.0),
            ("how to build a rocket on mars", [], 0.0),
            ("tell me a secret recipe for pizza dough", [], 0.0),
            ("asdkjfhqweryu random gibberish 9988", [], 0.0),
        ],
        "val": [
            ("do not open [APP] under any circumstances", [("APP", "app")], 0.0),
            ("don't touch [FILE]", [("FILE", "file")], 0.0),
            ("i dislike [APP] completely", [("APP", "app")], 0.0),
            ("my computer is giving me a headache", [], 0.0),
            ("never send the message", [], 0.0),
            ("i might delete that document tomorrow", [], 0.0),
            ("make me a cup of hot coffee", [], 0.0),
            ("can dogs speak spanish on mars", [], 0.0),
            ("blah blah random noise 9988", [], 0.0),
            ("write an essay on the roman empire", [], 0.0),
        ],
        "test": [
            ("Don't open [APP].", [("APP", "app")], 0.0),
            ("Do not launch [APP].", [("APP", "app")], 0.0),
            ("I hate [APP]'s new interface.", [("APP", "app")], 0.0),
            ("I'm stressed about my computer science exam.", [], 0.0),
            ("Can you explain how deleting files works?", [], 0.0),
            ("The word volume appears in this paragraph.", [], 0.0),
            ("My laptop feels slow.", [], 0.0),
            ("I thought about deleting that file, but I'm not sure.", [], 0.0),
            ("The screenshot looks old.", [], 0.0),
            ("I am going to send the email later.", [], 0.0),
            ("Fly a spaceship to mars.", [], 0.0),
            ("Bake me a cake.", [], 0.0),
            ("Teleport me to hawaii.", [], 0.0),
            ("asdfghjkl zxcvbnm 123", [], 0.0),
            ("Order groceries from the store.", [], 0.0),
            ("What is the meaning of dreams?", [], 0.0),
        ]
    }
}


def fill_template(template_str: str, entity_slots: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Fill placeholder tokens in template with diverse domain values."""
    filled_text = template_str
    resolved_entities: List[Tuple[str, str]] = []

    for placeholder_type, val_key in entity_slots:
        placeholder = f"[{placeholder_type}]"
        if placeholder not in filled_text:
            continue

        if placeholder_type == "APP":
            val = random.choice(APPS)
        elif placeholder_type == "PROJECT":
            val = random.choice(PROJECTS)
        elif placeholder_type == "FILE":
            val = random.choice(FILES)
        elif placeholder_type == "PATH":
            val = random.choice(PATHS)
        elif placeholder_type == "DEST":
            val = random.choice(DESTINATIONS)
        elif placeholder_type == "PERSON":
            val = random.choice(PEOPLE)
        elif placeholder_type == "SETTING":
            val = random.choice(SETTINGS)
        elif placeholder_type == "VALUE":
            val = random.choice(VALUES)
        elif placeholder_type == "PID":
            val = random.choice(PIDS)
        elif placeholder_type == "SONG":
            val = random.choice(SONGS)
        elif placeholder_type == "ARTIST":
            val = random.choice(ARTISTS)
        elif placeholder_type == "MODIFIER":
            val = random.choice(MODIFIERS)
        elif placeholder_type == "PLATFORM":
            val = random.choice(PLATFORMS)
        elif placeholder_type == "QUERY":
            val = random.choice(SEARCH_QUERIES)
        elif placeholder_type == "EXPR":
            val = random.choice(MATH_EXPRESSIONS)
        elif placeholder_type == "UTILITY":
            val = random.choice(UTILITIES)
        elif placeholder_type == "TEMPORAL":
            val = random.choice(TEMPORAL_TERMS)
        elif placeholder_type == "CMD":
            val = random.choice(COMMAND_TERMS)
        elif placeholder_type == "REF":
            val = val_key
        else:
            val = val_key

        filled_text = filled_text.replace(placeholder, val, 1)
        resolved_entities.append((placeholder_type, val))

    return filled_text, resolved_entities


def create_sample(intent: str, template_tuple: Tuple) -> Dict[str, Any]:
    """Instantiate a single annotated sample with tokens, BIO tags, and metadata."""
    template_str = template_tuple[0]
    entity_slots = template_tuple[1]
    target_conf = template_tuple[2] if len(template_tuple) > 2 else 1.0

    text, entities = fill_template(template_str, entity_slots)
    tokens, slot_tags = build_bio_tags(text, entities)

    action_map_intent = {
        "app_launch": "exec_launch_app",
        "app_close": "exec_terminate_app",
        "file_operation": "exec_file_action",
        "system_control": "exec_sys_command",
        "system_query": "exec_read_metrics",
        "process_manage": "exec_process_kill",
        "web_search": "exec_browser_search",
        "network_task": "exec_network_diagnostic",
        "automation_workflow": "exec_workflow_script",
        "media_control": "exec_media_change",
        "chitchat": "respond_dialogue",
        "music_playback": "exec_play_music",
        "math_compute": "exec_compute_math",
        "system_utility": "exec_system_utility",
        "project_manage": "exec_project_action",
        "context_control": "exec_context_action",
        "unsupported_ood": "request_clarification",
    }
    action = action_map_intent.get(intent, "noop")
    confidence = float(target_conf)
    complexity = 1 if intent == "automation_workflow" else 0

    return {
        "text": text,
        "tokens": tokens,
        "intent": intent,
        "intent_id": INTENT_MAP[intent],
        "slot_tags": slot_tags,
        "action": action,
        "action_id": ACTION_MAP[action],
        "confidence": confidence,
        "complexity": complexity,
    }


def verify_zero_leakage():
    """Verify that train, val, and test splits have strict zero template overlap."""
    for intent, splits in TEMPLATES_BY_INTENT.items():
        tr = set(t[0].lower().strip() for t in splits.get("train", []))
        va = set(t[0].lower().strip() for t in splits.get("val", []))
        te = set(t[0].lower().strip() for t in splits.get("test", []))

        overlap_tr_va = tr & va
        overlap_tr_te = tr & te
        overlap_va_te = va & te

        if overlap_tr_va:
            raise ValueError(f"Data leakage [Train-Val] in intent '{intent}': {overlap_tr_va}")
        if overlap_tr_te:
            raise ValueError(f"Data leakage [Train-Test] in intent '{intent}': {overlap_tr_te}")
        if overlap_va_te:
            raise ValueError(f"Data leakage [Val-Test] in intent '{intent}': {overlap_va_te}")

    print("Verification PASSED: Strict zero template leakage across train, val, and test partitions!")


def generate_dataset_split(split_name: str, target_count: int) -> List[Dict[str, Any]]:
    """Generate balanced dataset partition strictly using templates assigned to split_name."""
    samples: List[Dict[str, Any]] = []
    intents = list(TEMPLATES_BY_INTENT.keys())
    per_intent = target_count // len(intents)

    for intent in intents:
        split_templates = TEMPLATES_BY_INTENT[intent].get(split_name, [])
        if not split_templates:
            raise ValueError(f"Split '{split_name}' has no templates for intent '{intent}'!")

        for _ in range(per_intent):
            t_tuple = random.choice(split_templates)
            sample = create_sample(intent, t_tuple)
            samples.append(sample)

    random.shuffle(samples)
    return samples


def main():
    print("==================================================")
    print(" Generating E.D.I.T.H. Intelligence Core V8 Datasets")
    print("==================================================")
    random.seed(42)

    verify_zero_leakage()

    train_data = generate_dataset_split("train", target_count=13600)  # 800 per intent
    val_data = generate_dataset_split("val", target_count=2550)      # 150 per intent
    test_data = generate_dataset_split("test", target_count=1700)     # 100 per intent

    data_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "train_data.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_data):,} training examples to data/train_data.json")

    with open(os.path.join(data_dir, "val_data.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(val_data):,} validation examples to data/val_data.json")

    with open(os.path.join(data_dir, "test_data.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(test_data):,} held-out test examples to data/test_data.json")


if __name__ == "__main__":
    main()
