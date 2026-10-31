"""Social Introduction & Entity Memory Generator for E.D.I.T.H. V13.

Generates social introductions ("I have a friend named X", "Meet my colleague Y")
paired with structured <mem> tags and warm conversational responses.
"""

import random
from typing import Dict, List, Tuple


class SocialIntroGenerator:
    NAMES = [
        "Harshan", "Rohan", "Sneha", "Aarav", "Pooja", "Vikram",
        "Ananya", "Karan", "Tanvi", "Siddharth", "Rahul", "Priya",
        "Alex", "David", "Sarah", "Michael", "Elena", "Marcus"
    ]

    RELATIONS = [
        ("friend", "a friend", "friend"),
        ("best friend", "my best friend", "friend"),
        ("colleague", "a colleague from work", "colleague"),
        ("roommate", "my roommate", "roommate"),
        ("cousin", "my cousin", "family"),
        ("brother", "my brother", "family"),
        ("sister", "my sister", "family"),
    ]

    TEMPLATES = [
        ("I have a {rel_phrase} named {name}.",
         "<mem>{{\"entity\": \"{name}\", \"relation\": \"{rel_tag}\"}}</mem>Nice to meet {name} through you! How long have you two known each other?"),
        
        ("Meet my {rel_phrase}, {name}.",
         "<mem>{{\"entity\": \"{name}\", \"relation\": \"{rel_tag}\"}}</mem>Hey {name}! Great to meet you. Any friend of Aditya's is someone I'm glad to know."),

        ("So I was talking to {name}, who is my {rel_phrase}.",
         "<mem>{{\"entity\": \"{name}\", \"relation\": \"{rel_tag}\"}}</mem>Got it, logged {name} as your {rel_phrase}. What were you two discussing?"),

        ("Do you remember {name}? He's my {rel_phrase}.",
         "<mem>{{\"entity\": \"{name}\", \"relation\": \"{rel_tag}\"}}</mem>Of course, {name} your {rel_phrase}. What's the latest with him?"),
    ]

    @classmethod
    def generate_dataset(cls, count: int = 200) -> List[Dict[str, str]]:
        samples = []
        for _ in range(count):
            name = random.choice(cls.NAMES)
            rel_name, rel_phrase, rel_tag = random.choice(cls.RELATIONS)
            u_tmpl, a_tmpl = random.choice(cls.TEMPLATES)

            user_text = u_tmpl.format(name=name, rel_phrase=rel_phrase)
            edith_text = a_tmpl.format(name=name, rel_phrase=rel_phrase, rel_tag=rel_tag)

            samples.append({
                "user": user_text,
                "edith": edith_text,
                "category": "social_intro"
            })
        return samples
