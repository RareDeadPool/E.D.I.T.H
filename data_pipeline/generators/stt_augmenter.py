"""STT Acoustic Slip & Homophone Augmentation Generator for E.D.I.T.H. V13.

Generates realistic acoustic speech recognition errors (homophones, phonetic slips,
dropped function words, slurred contractions) reflecting Whisper and Pocket-VAD failure modes.
"""

import random
from typing import Dict, List, Tuple


class STTAcousticAugmenter:
    # Common acoustic confusion pairs observed in Whisper / Pocket-VAD on laptop mics
    PHONETIC_CONFUSIONS: Dict[str, List[str]] = {
        "tell": ["help", "del", "dell", "sell", "hell", "tel"],
        "joke": ["choke", "jok", "yoke", "smoke"],
        "close": ["clothes", "cloze", "rose", "glows"],
        "chrome": ["roam", "home", "crome", "comb"],
        "open": ["oatman", "hope in", "opin", "open up"],
        "kill": ["chill", "gill", "bill", "pill"],
        "process": ["progress", "prosess", "prowess"],
        "friend": ["frend", "fiend", "fried"],
        "sad": ["bad", "sat", "said"],
        "years": ["ears", "tears"],
        "jobs": ["jaws", "chops", "dobs"],
        "show": ["shoe", "slow", "sew"],
        "system": ["cistern", "sistem"],
        "screen": ["scream", "stream"],
        "harshan": ["darshan", "harsh an", "harshan"],
        "terminate": ["germinate", "determinate"],
        "file": ["vial", "phial", "fail"],
        "notepad": ["note bad", "oatpad", "boatpad"],
    }

    # Common spoken fillers and prefixes that STT captures
    SPEECH_FILLERS = [
        "come on, ",
        "hey edith, ",
        "uh, ",
        "um, ",
        "yo, ",
        "wait, ",
        "so, ",
        "actually, ",
        "please, ",
        "edith, ",
    ]

    @classmethod
    def augment_text(cls, text: str, slip_probability: float = 0.35) -> str:
        """Applies acoustic slips and optional filler prefixes to a sentence."""
        words = text.split()
        augmented_words = []

        for word in words:
            clean_w = word.lower().strip(".,!?;:\"'")
            punct = word[len(clean_w):] if len(word) > len(clean_w) else ""

            if clean_w in cls.PHONETIC_CONFUSIONS and random.random() < slip_probability:
                confused = random.choice(cls.PHONETIC_CONFUSIONS[clean_w])
                # Preserve capitalization
                if word[0].isupper():
                    confused = confused.capitalize()
                augmented_words.append(confused + punct)
            else:
                augmented_words.append(word)

        result = " ".join(augmented_words)

        # 20% chance to prepend acoustic filler prefix
        if random.random() < 0.20 and not any(result.lower().startswith(f) for f in cls.SPEECH_FILLERS):
            prefix = random.choice(cls.SPEECH_FILLERS)
            result = prefix + result[0].lower() + result[1:]

        return result

    @classmethod
    def generate_slip_dataset(cls, seed_pairs: List[Tuple[str, str]], augmentations_per_seed: int = 5) -> List[Dict[str, str]]:
        """Generates augmented dialogue pairs mapping acoustic slips to intended answers."""
        samples = []
        for user_msg, edith_reply in seed_pairs:
            # 1. Clean pair
            samples.append({"user": user_msg, "edith": edith_reply, "augmented": False})
            # 2. Augmented acoustic variants
            for _ in range(augmentations_per_seed):
                augmented_user = cls.augment_text(user_msg, slip_probability=0.45)
                if augmented_user != user_msg:
                    samples.append({
                        "user": augmented_user,
                        "edith": edith_reply,
                        "augmented": True,
                        "intended": user_msg
                    })
        return samples
