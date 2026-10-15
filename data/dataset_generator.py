"""Advanced Domain Dataset Generator for E.D.I.T.H. Intelligence Core V2.1.

Generates diverse, linguistically varied examples covering all 17 intents:
- Natural-language paraphrases and indirect intent expressions
- Rich entity values across applications, files, people, destinations, projects, environments, temporal tags, PIDs, settings
- Multi-step compound workflows and dependency sequences
- Context-dependent follow-ups, anaphora, ordinals, user corrections, and reversals
- Ambiguous and out-of-distribution adversarial examples with calibrated confidence (1.0 for valid, 0.5 for ambiguous, 0.0 for OOD)
- Strict partition-based splitting (train / val / test) preventing template leakage
"""

import os
import re
import json
import random
from typing import List, Dict, Tuple, Any, Optional

with open("data/metadata_maps.json", "r", encoding="utf-8") as f:
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
    "mail", "outlook", "python environment"
]

PROJECTS = [
    "EDITH", "edith", "my edith project", "backend service", "frontend app", "neural network",
    "website", "api server", "agent core", "data pipeline", "portfolio", "classifier", "workspace"
]

FILES = [
    "resume.pdf", "resume", "my resume", "report.pdf", "notes.txt", "notes", "todo.txt",
    "budget.xlsx", "presentation.pptx", "data.csv", "main.py", "test.py", "server.js",
    "index.html", "config.json", "architecture.md", "README.md", "script.py", "contract.pdf",
    "invoice.pdf", "summary.docx", "draft.txt"
]

PATHS = [
    "desktop", "the desktop", "downloads", "my downloads", "documents", "my documents",
    "c:/projects/edith", "c:/downloads", "c:/documents", "./src", "./data", "./models",
    "~/projects", "project folder", "root directory"
]

PEOPLE = ["John", "Rahul", "Aditya", "Alex", "Sarah", "Emily", "David", "Michael", "Priya"]

DESTINATIONS = ["desktop", "downloads", "documents", "archive", "trash", "backup", "server", "there", "folder"]

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
    "python tips", "machine learning concepts", "rust programming language", "docker containerization"
]

MATH_EXPRESSIONS = [
    "2+2", "15 * 12", "1250 / 5", "450 + 820", "20 percent of 1500", "20% of 1500",
    "sqrt 144", "square root of 144", "square root of 256", "50 * 20", "50 + 25",
    "(45 * 12) + 100 / 2", "100 - 37", "75% of 800"
]

UTILITIES = [
    "screenshot", "screen capture", "capture screen", "take screenshot", "minimize all windows",
    "show desktop", "what time is it", "current time", "what date is today", "today's date",
    "empty recycle bin", "clean recycle bin"
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

    # Sort entities longest first to prevent subphrase hijacking
    sorted_entities = sorted(entities, key=lambda e: len(e[1]), reverse=True)

    for slot_type, val in sorted_entities:
        val_tokens = tokenize_text(val)
        if not val_tokens:
            continue
        v_len = len(val_tokens)

        # Match contiguous slice in tokens
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
            ("edith open [APP]", [("APP", "app")], 1.0),
            ("pull up the [APP]", [("APP", "app")], 1.0),
            ("start up [APP]", [("APP", "app")], 1.0),
            ("open", [], 0.5),  # Ambiguous / incomplete
        ],
        "val": [
            ("please open up [APP]", [("APP", "app")], 1.0),
            ("could you launch [APP] for me", [("APP", "app")], 1.0),
            ("bring up [APP]", [("APP", "app")], 1.0),
            ("can you start [APP]", [("APP", "app")], 1.0),
        ],
        "test": [
            ("could you bring up my coding environment?", [("APP", "coding environment")], 1.0),
            ("fire up my dev environment", [("APP", "dev environment")], 1.0),
            ("open the ide", [("APP", "ide")], 1.0),
            ("launch visual studio code for me", [("APP", "visual studio code")], 1.0),
        ]
    },

    # 2. app_close
    "app_close": {
        "train": [
            ("close [APP]", [("APP", "app")], 1.0),
            ("kill [APP]", [("APP", "app")], 1.0),
            ("terminate [APP]", [("APP", "app")], 1.0),
            ("exit [APP]", [("APP", "app")], 1.0),
            ("shut down [APP]", [("APP", "app")], 1.0),
            ("close down [APP]", [("APP", "app")], 1.0),
            ("quit [APP]", [("APP", "app")], 1.0),
            ("end [APP]", [("APP", "app")], 1.0),
            ("can you close [APP]", [("APP", "app")], 1.0),
            ("force close [APP]", [("APP", "app")], 1.0),
            ("close the application", [], 0.5),  # Ambiguous / missing app
        ],
        "val": [
            ("please terminate [APP]", [("APP", "app")], 1.0),
            ("could you close [APP] for me", [("APP", "app")], 1.0),
            ("shut off [APP]", [("APP", "app")], 1.0),
            ("exit out of [APP]", [("APP", "app")], 1.0),
        ],
        "test": [
            ("dismiss [APP]", [("APP", "app")], 1.0),
            ("kill the [APP] window", [("APP", "app")], 1.0),
            ("close the active [APP]", [("APP", "app")], 1.0),
            ("terminate [APP] application", [("APP", "app")], 1.0),
        ]
    },

    # 3. file_operation
    "file_operation": {
        "train": [
            ("find the [TEMPORAL] version of my [FILE]", [("TEMPORAL", "temporal"), ("FILE", "file")], 1.0),
            ("find [TEMPORAL] version of [FILE]", [("TEMPORAL", "temporal"), ("FILE", "file")], 1.0),
            ("locate the [TEMPORAL] [FILE]", [("TEMPORAL", "temporal"), ("FILE", "file")], 1.0),
            ("find my [FILE] on the [PATH]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("locate [FILE] in [PATH]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("create file [FILE] at [PATH]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("delete [FILE] from [PATH]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("delete [FILE]", [("FILE", "file")], 1.0),
            ("copy [FILE] to [DEST]", [("FILE", "file"), ("DEST", "dest")], 1.0),
            ("move [FILE] to [DEST]", [("FILE", "file"), ("DEST", "dest")], 1.0),
            ("delete the file", [], 0.5),  # Ambiguous
        ],
        "val": [
            ("search for my [TEMPORAL] [FILE]", [("TEMPORAL", "temporal"), ("FILE", "file")], 1.0),
            ("look for [FILE] in [PATH]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("create a new file named [FILE] on [PATH]", [("FILE", "file"), ("PATH", "path")], 1.0),
            ("remove [FILE] from [PATH]", [("FILE", "file"), ("PATH", "path")], 1.0),
        ],
        "test": [
            ("Find the most recent version of my resume.", [("TEMPORAL", "most recent"), ("FILE", "resume")], 1.0),
            ("find the latest version of report.pdf", [("TEMPORAL", "latest"), ("FILE", "report.pdf")], 1.0),
            ("locate the newest notes file", [("TEMPORAL", "newest"), ("FILE", "notes")], 1.0),
            ("get rid of the old draft", [("FILE", "draft")], 1.0),
        ]
    },

    # 4. system_control
    "system_control": {
        "train": [
            ("turn [SETTING] to [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("adjust [SETTING] to [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("set [SETTING] to [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("make the [SETTING] [VALUE]", [("SETTING", "setting"), ("VALUE", "value")], 1.0),
            ("mute the [SETTING]", [("SETTING", "setting")], 1.0),
            ("unmute the [SETTING]", [("SETTING", "setting")], 1.0),
            ("put system to sleep", [], 1.0),
            ("lock the pc", [], 1.0),
            ("adjust the setting", [], 0.5),  # Ambiguous
        ],
        "val": [
            ("increase master volume", [("SETTING", "volume"), ("VALUE", "way up")], 1.0),
            ("turn brightness down to 20%", [("SETTING", "brightness"), ("VALUE", "20%")], 1.0),
            ("unmute system sound", [("SETTING", "sound")], 1.0),
            ("sleep the computer", [], 1.0),
        ],
        "test": [
            ("turn the volume way down", [("SETTING", "volume"), ("VALUE", "way down")], 1.0),
            ("make it a bit louder", [("SETTING", "volume"), ("VALUE", "a bit louder")], 1.0),
            ("lower brightness to 50%", [("SETTING", "brightness"), ("VALUE", "50%")], 1.0),
            ("lock workstation", [], 1.0),
        ]
    },

    # 5. system_query
    "system_query": {
        "train": [
            ("what is the system cpu load", [], 1.0),
            ("show available ram memory", [], 1.0),
            ("check battery percentage", [], 1.0),
            ("how much storage capacity is left", [], 1.0),
            ("show system performance and memory", [], 1.0),
            ("display resource utilization", [], 1.0),
            ("read system diagnostics", [], 1.0),
        ],
        "val": [
            ("inspect system performance metrics", [], 1.0),
            ("what is my battery level at", [], 1.0),
            ("show storage capacity", [], 1.0),
            ("inspect memory metrics", [], 1.0),
        ],
        "test": [
            ("what is the current cpu usage", [], 1.0),
            ("how much storage do I have left", [], 1.0),
            ("check cpu and ram usage", [], 1.0),
            ("how much ram is free", [], 1.0),
        ]
    },

    # 6. process_manage
    "process_manage": {
        "train": [
            ("kill process [PID]", [("PID", "pid")], 1.0),
            ("terminate process with pid [PID]", [("PID", "pid")], 1.0),
            ("force close pid [PID]", [("PID", "pid")], 1.0),
            ("end task [PID]", [("PID", "pid")], 1.0),
            ("stop process [PID]", [("PID", "pid")], 1.0),
            ("kill pid [PID]", [("PID", "pid")], 1.0),
            ("terminate task [PID]", [("PID", "pid")], 1.0),
            ("kill the process", [], 0.5),  # Ambiguous / missing PID
            ("terminate the task", [], 0.5),
        ],
        "val": [
            ("force kill process [PID]", [("PID", "pid")], 1.0),
            ("terminate the process running under pid [PID]", [("PID", "pid")], 1.0),
            ("stop pid [PID] immediately", [("PID", "pid")], 1.0),
            ("kill task [PID]", [("PID", "pid")], 1.0),
        ],
        "test": [
            ("kill process 4096", [("PID", "4096")], 1.0),
            ("terminate pid 1042", [("PID", "1042")], 1.0),
            ("end task 8820", [("PID", "8820")], 1.0),
            ("kill the process with pid 1290", [("PID", "1290")], 1.0),
        ]
    },

    # 7. web_search
    "web_search": {
        "train": [
            ("search web for [QUERY]", [("QUERY", "query")], 1.0),
            ("look up [QUERY] on google", [("QUERY", "query")], 1.0),
            ("search online for [QUERY]", [("QUERY", "query")], 1.0),
            ("google [QUERY]", [("QUERY", "query")], 1.0),
            ("find information about [QUERY]", [("QUERY", "query")], 1.0),
            ("search internet for [QUERY]", [("QUERY", "query")], 1.0),
            ("browse the web for [QUERY]", [("QUERY", "query")], 1.0),
            ("search online for", [], 0.5),  # Ambiguous / missing query
            ("google for", [], 0.5),
        ],
        "val": [
            ("look up [QUERY]", [("QUERY", "query")], 1.0),
            ("search for [QUERY] on google", [("QUERY", "query")], 1.0),
            ("search online regarding [QUERY]", [("QUERY", "query")], 1.0),
        ],
        "test": [
            ("search online for quantum computing", [("QUERY", "quantum computing")], 1.0),
            ("look up latest machine learning models", [("QUERY", "latest machine learning models")], 1.0),
            ("search for cats and dogs", [("QUERY", "cats and dogs")], 1.0),
            ("google information about space exploration", [("QUERY", "space exploration")], 1.0),
        ]
    },

    # 8. network_task
    "network_task": {
        "train": [
            ("ping [DEST]", [("DEST", "dest")], 1.0),
            ("test connection to [DEST]", [("DEST", "dest")], 1.0),
            ("check network latency to [DEST]", [("DEST", "dest")], 1.0),
            ("trace network to [DEST]", [("DEST", "dest")], 1.0),
            ("check internet connectivity", [], 1.0),
            ("diagnose network status", [], 1.0),
            ("ping host [DEST]", [("DEST", "dest")], 1.0),
            ("check if [DEST] is reachable", [("DEST", "dest")], 1.0),
            ("ping the server", [], 0.5),  # Ambiguous
        ],
        "val": [
            ("verify connection to [DEST]", [("DEST", "dest")], 1.0),
            ("run network ping to [DEST]", [("DEST", "dest")], 1.0),
            ("is [DEST] online", [("DEST", "dest")], 1.0),
            ("diagnose wifi connection", [], 1.0),
        ],
        "test": [
            ("ping google.com", [("DEST", "google.com")], 1.0),
            ("check reachability of 8.8.8.8", [("DEST", "8.8.8.8")], 1.0),
            ("test network connection to server", [("DEST", "server")], 1.0),
            ("check internet status", [], 1.0),
        ]
    },

    # 9. automation_workflow
    "automation_workflow": {
        "train": [
            ("open [APP] and type hello world", [("APP", "app")], 1.0),
            ("open [APP] and search for [QUERY]", [("APP", "app"), ("QUERY", "query")], 1.0),
            ("take a screenshot and open downloads", [("UTILITY", "screenshot"), ("PATH", "downloads")], 1.0),
            ("create file [FILE] on desktop and open it", [("FILE", "file"), ("PATH", "desktop")], 1.0),
            ("open spotify, play [SONG] and increase volume", [("APP", "spotify"), ("SONG", "song")], 1.0),
            ("close [APP] and lock workstation", [("APP", "app")], 1.0),
            ("empty recycle bin and minimize all windows", [("UTILITY", "empty recycle bin")], 1.0),
            ("launch [APP] then type [QUERY]", [("APP", "app")], 1.0),
        ],
        "val": [
            ("open [APP] and look up [QUERY]", [("APP", "app"), ("QUERY", "query")], 1.0),
            ("take a screenshot and save it to desktop", [("UTILITY", "screenshot"), ("PATH", "desktop")], 1.0),
            ("open calculator then open notepad", [("APP", "calculator")], 1.0),
            ("launch [APP] and minimize all windows", [("APP", "app")], 1.0),
        ],
        "test": [
            ("Find my latest EDITH project, open it in VS Code, run the tests, and tell me whether they pass.", [("PROJECT", "EDITH"), ("APP", "VS Code")], 1.0),
            ("Open my project, run it, and check whether it starts properly.", [("PROJECT", "project")], 1.0),
            ("open notepad and write hello world", [("APP", "notepad")], 1.0),
            ("open chrome and search for artificial intelligence", [("APP", "chrome"), ("QUERY", "artificial intelligence")], 1.0),
        ]
    },

    # 10. media_control
    "media_control": {
        "train": [
            ("pause music", [], 1.0),
            ("pause the playback", [], 1.0),
            ("resume music", [], 1.0),
            ("unpause the track", [], 1.0),
            ("skip this song", [], 1.0),
            ("play next track", [], 1.0),
            ("skip to next song", [], 1.0),
            ("go back to previous song", [], 1.0),
            ("rewind the track", [], 1.0),
            ("stop music playback", [], 1.0),
        ],
        "val": [
            ("pause Spotify playback", [], 1.0),
            ("resume playing music", [], 1.0),
            ("skip forward to the next song", [], 1.0),
            ("play the previous track", [], 1.0),
        ],
        "test": [
            ("pause playback", [], 1.0),
            ("resume the song", [], 1.0),
            ("skip track", [], 1.0),
            ("previous song", [], 1.0),
        ]
    },

    # 11. chitchat
    "chitchat": {
        "train": [
            ("hello edith", [], 1.0),
            ("hi edith", [], 1.0),
            ("good morning edith", [], 1.0),
            ("good afternoon edith", [], 1.0),
            ("how are you doing today", [], 1.0),
            ("who are you", [], 1.0),
            ("tell me who you are", [], 1.0),
            ("thank you edith", [], 1.0),
            ("thanks for the help", [], 1.0),
            ("what can you do for me", [], 1.0),
        ],
        "val": [
            ("greetings edith", [], 1.0),
            ("how are you feeling", [], 1.0),
            ("what is your purpose", [], 1.0),
            ("much appreciated edith", [], 1.0),
        ],
        "test": [
            ("hello edith, how are you?", [], 1.0),
            ("who are you and what do you do?", [], 1.0),
            ("good evening edith", [], 1.0),
            ("thanks a lot edith", [], 1.0),
        ]
    },

    # 12. music_playback
    "music_playback": {
        "train": [
            ("put on [SONG] while i work", [("SONG", "song")], 1.0),
            ("play [SONG]", [("SONG", "song")], 1.0),
            ("play [SONG] by [ARTIST]", [("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("play [SONG] on [PLATFORM]", [("SONG", "song"), ("PLATFORM", "platform")], 1.0),
            ("stream [SONG] [MODIFIER]", [("SONG", "song"), ("MODIFIER", "modifier")], 1.0),
            ("play [MODIFIER] [SONG]", [("MODIFIER", "modifier"), ("SONG", "song")], 1.0),
            ("queue up [SONG] by [ARTIST] on [PLATFORM]", [("SONG", "song"), ("ARTIST", "artist"), ("PLATFORM", "platform")], 1.0),
            ("start playing [SONG]", [("SONG", "song")], 1.0),
            ("listen to [SONG] by [ARTIST]", [("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("play some music", [], 0.5),  # Ambiguous / generic
        ],
        "val": [
            ("can you play some relaxing music while i work", [("SONG", "relaxing music")], 1.0),
            ("put on [SONG] by [ARTIST]", [("SONG", "song"), ("ARTIST", "artist")], 1.0),
            ("play [SONG] [MODIFIER] on [PLATFORM]", [("SONG", "song"), ("MODIFIER", "modifier"), ("PLATFORM", "platform")], 1.0),
            ("queue up [SONG] for me", [("SONG", "song")], 1.0),
        ],
        "test": [
            ("Put on something relaxing while I work.", [("SONG", "something relaxing"), ("MODIFIER", "relaxing")], 1.0),
            ("play slowed and reverb believer", [("MODIFIER", "slowed and reverb"), ("SONG", "believer")], 1.0),
            ("play blinding lights on youtube", [("SONG", "blinding lights"), ("PLATFORM", "youtube")], 1.0),
            ("stream chill beats on spotify", [("SONG", "chill beats"), ("PLATFORM", "spotify")], 1.0),
        ]
    },

    # 13. math_compute
    "math_compute": {
        "train": [
            ("calculate [EXPR]", [("EXPR", "expr")], 1.0),
            ("what is [EXPR]", [("EXPR", "expr")], 1.0),
            ("solve [EXPR]", [("EXPR", "expr")], 1.0),
            ("evaluate [EXPR]", [("EXPR", "expr")], 1.0),
            ("compute [EXPR]", [("EXPR", "expr")], 1.0),
            ("calculate", [], 0.5),  # Ambiguous / missing expression
        ],
        "val": [
            ("compute [EXPR] for me", [("EXPR", "expr")], 1.0),
            ("tell me [EXPR]", [("EXPR", "expr")], 1.0),
            ("what does [EXPR] equal", [("EXPR", "expr")], 1.0),
        ],
        "test": [
            ("what is 20 percent of 1500", [("EXPR", "20 percent of 1500")], 1.0),
            ("calculate (45 * 12) + 100 / 2", [("EXPR", "(45 * 12) + 100 / 2")], 1.0),
            ("what is the square root of 144", [("EXPR", "square root of 144")], 1.0),
            ("what is 50 + 25", [("EXPR", "50 + 25")], 1.0),
        ]
    },

    # 14. system_utility
    "system_utility": {
        "train": [
            ("grab screen snapshot", [("UTILITY", "screenshot")], 1.0),
            ("take a snapshot of the screen", [("UTILITY", "screenshot")], 1.0),
            ("display current time", [("UTILITY", "current time")], 1.0),
            ("show me today's date", [("UTILITY", "today date")], 1.0),
            ("clean the recycle bin", [("UTILITY", "clean recycle bin")], 1.0),
            ("hide all open windows", [("UTILITY", "minimize all windows")], 1.0),
            ("minimize everything on screen", [("UTILITY", "minimize all windows")], 1.0),
        ],
        "val": [
            ("grab a screenshot for me", [("UTILITY", "screenshot")], 1.0),
            ("tell me the current time", [("UTILITY", "current time")], 1.0),
            ("clear out recycle bin", [("UTILITY", "empty recycle bin")], 1.0),
            ("show the desktop screen", [("UTILITY", "show desktop")], 1.0),
        ],
        "test": [
            ("take a screenshot", [("UTILITY", "screenshot")], 1.0),
            ("what time is it", [("UTILITY", "current time")], 1.0),
            ("empty recycle bin", [("UTILITY", "empty recycle bin")], 1.0),
            ("minimize all windows", [("UTILITY", "minimize all windows")], 1.0),
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
            ("run the project", [], 0.5),  # Ambiguous / missing project
            ("run tests", [], 0.5),
        ],
        "val": [
            ("let's return to the [PROJECT]", [("PROJECT", "project")], 1.0),
            ("run the unit tests for [PROJECT]", [("PROJECT", "project"), ("CMD", "unit tests")], 1.0),
            ("verify if the tests pass", [("CMD", "tests")], 1.0),
            ("test whether the application starts properly", [("CMD", "test")], 1.0),
        ],
        "test": [
            ("Let's get back to the project I was working on.", [("PROJECT", "project")], 1.0),
            ("run the tests, and tell me whether they pass", [("CMD", "tests")], 1.0),
            ("check whether it starts properly", [], 1.0),
            ("run unit tests on my edith project", [("CMD", "unit tests"), ("PROJECT", "edith project")], 1.0),
        ]
    },

    # 16. context_control
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
        ],
        "val": [
            ("that is not the one i wanted, show me the other version", [("REF", "other version")], 1.0),
            ("pick the first result", [("REF", "first result")], 1.0),
            ("delete it", [("REF", "it")], 1.0),
            ("undo that", [], 1.0),
        ],
        "test": [
            ("That isn't the one I meant; find the other version.", [("REF", "other version")], 1.0),
            ("Not the old one, the latest one.", [("TEMPORAL", "latest"), ("REF", "one")], 1.0),
            ("Use the second result.", [("REF", "second result")], 1.0),
            ("Move it there instead.", [("REF", "it"), ("DEST", "there")], 1.0),
            ("Now run it.", [("REF", "it")], 1.0),
            ("Undo the last step.", [], 1.0),
        ]
    },

    # 17. unsupported_ood
    "unsupported_ood": {
        "train": [
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
            ("can dogs speak spanish on mars", [], 0.0),
            ("make me a cup of hot coffee", [], 0.0),
            ("blah blah random noise 9988", [], 0.0),
            ("write an essay on the roman empire", [], 0.0),
        ],
        "test": [
            ("fly a spaceship to mars", [], 0.0),
            ("bake me a cake", [], 0.0),
            ("teleport me to hawaii", [], 0.0),
            ("asdfghjkl zxcvbnm 123", [], 0.0),
            ("order groceries from the store", [], 0.0),
            ("what is the meaning of dreams", [], 0.0),
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

    print("Verification PASSED: Zero template leakage across train, val, and test partitions!")


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
    print(" Generating E.D.I.T.H. Intelligence Core V2.1 Datasets")
    print("==================================================")
    random.seed(42)

    # 1. Audit template leakage
    verify_zero_leakage()

    # 2. Generate balanced datasets across all 17 intents
    train_data = generate_dataset_split("train", target_count=13600)  # 800 per intent
    val_data = generate_dataset_split("val", target_count=2550)      # 150 per intent
    test_data = generate_dataset_split("test", target_count=1700)     # 100 per intent

    os.makedirs("data", exist_ok=True)

    with open("data/train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_data):,} training examples to data/train_data.json")

    with open("data/val_data.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(val_data):,} validation examples to data/val_data.json")

    with open("data/test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(test_data):,} held-out test examples to data/test_data.json")


if __name__ == "__main__":
    main()
