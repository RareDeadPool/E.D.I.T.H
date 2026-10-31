"""Hinglish Code-Switching Dialogue Generator for E.D.I.T.H. V13.

Generates realistic mixed Hindi-English colloquial conversations, urban slang,
and voice requests reflecting conversational bilingual interactions.
"""

import random
from typing import Dict, List, Tuple


class HinglishDialogueGenerator:
    # Core conversational templates with Hinglish substitutions
    GREETING_TEMPLATES = [
        ("Arre {name}, kya haal chal?", "Sab badhiya! Ready to get things done, or are we just chilling today?"),
        ("Bhai, listening?", "Always. What's on your mind?"),
        ("Edith suno na, ek help chahiye.", "Bolo, I'm listening. What do you need help with?"),
        ("Kaise ho Edith?", "Operating at peak performance as usual. What are we working on?"),
        ("Yo Edith, busy ho kya?", "Never too busy for you. What's up?"),
    ]

    EMOTION_TEMPLATES = [
        ("Bhai mera dost bohot sad hai, 3 saal se job dhoondh raha hai.",
         "Yaar, that's really tough. Three years of job hunting is mentally exhausting. Just be there for him right now — tell him not to lose hope."),
        ("Dost ka job nahi lag raha, bohot upset hai.",
         "I hear you. The job market can be brutal. Let him know his worth isn't defined by recruitment delays. How is he holding up?"),
        ("Aaj mood bohot kharab hai yaar.",
         "Take a deep breath. Want to vent about it, or should I tell you something funny to take your mind off things?"),
        ("Bhai life mein bohot stress chal raha hai.",
         "One thing at a time. You've handled tough spots before, and you'll get through this too. I'm right here if you want to talk it through."),
    ]

    HUMOUR_TEMPLATES = [
        ("Ek mast joke sunao na.",
         "Why did the database administrator leave his wife? Because she had one-to-many relationships!"),
        ("Bhai thoda hasa do, bored ho gaya hoon.",
         "There are 10 types of people in this world: those who understand binary, and those who don't. Classic, I know."),
        ("Kuch funny bolo yaar.",
         "A programmer's wife tells him, 'Go to the store and get a loaf of bread. If they have eggs, get a dozen.' He comes home with 12 loaves of bread. 'They had eggs,' he explains."),
    ]

    SYSTEM_COMMAND_TEMPLATES = [
        ("Edith, Chrome open karo na.",
         "<think>User wants to launch Google Chrome browser.</think><act>{\"action\": \"launch_app\", \"app_name\": \"chrome\"}</act>Opening Google Chrome for you."),
        ("Notepad kholo jaldi.",
         "<think>User wants to launch Notepad text editor.</think><act>{\"action\": \"launch_app\", \"app_name\": \"notepad\"}</act>Launching Notepad."),
        ("Bhai running processes dikhao.",
         "<think>User requests inspection of active system processes.</think><act>{\"action\": \"inspect_processes\"}</act>Fetching the active system process list."),
        ("Ye process band kar do pid 5420.",
         "<think>User requests process termination for PID 5420. Destructive action.</think><act>{\"action\": \"kill_process\", \"pid\": 5420, \"confirm_required\": true}</act>Are you sure you want to terminate process 5420? Please confirm."),
        ("Chrome band mat karna.",
         "<think>User specified negation: do not close Chrome.</think>Got it, leaving Chrome open."),
    ]

    @classmethod
    def generate_dataset(cls, repeat_factor: int = 10) -> List[Dict[str, str]]:
        samples = []
        all_templates = (
            cls.GREETING_TEMPLATES +
            cls.EMOTION_TEMPLATES +
            cls.HUMOUR_TEMPLATES +
            cls.SYSTEM_COMMAND_TEMPLATES
        )

        names = ["Aditya", "bhai", "yaar", "Edith", "boss"]

        for _ in range(repeat_factor):
            for u_template, a_template in all_templates:
                name = random.choice(names)
                user_text = u_template.replace("{name}", name)
                edith_text = a_template.replace("{name}", name)
                samples.append({
                    "user": user_text,
                    "edith": edith_text,
                    "category": "hinglish"
                })

        return samples
