"""Advanced Cognitive, Problem-Solving & Empathy Scenario Engine for E.D.I.T.H.

Provides deep, emotionally intelligent, and practically grounded responses across 75
distinct real-world human situations spanning:
1. Relationships, Heartbreak & Social Bonds (Scenarios 1-10)
2. Career, Academics & Work Pressure (Scenarios 11-20)
3. Mental Health, Emotional Crises & Well-being (Scenarios 21-30)
4. Cognitive Decisions & Rational Problem Solving (Scenarios 31-40)
5. Software Engineering & DevOps Panics (Scenarios 41-50)
6. Deep Focus, ADHD, Creativity & Productivity (Scenarios 51-60)
7. Everyday Life Navigation, Money & Social Skills (Scenarios 61-68)
8. Humor, Philosophizing & Late-Night Companionship (Scenarios 69-75)
"""

import os
import re
import sys
import json
import random
from typing import Dict, Any, Optional, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class CognitiveScenarioEngine:
    """Specialized engine evaluating 75 complex human cognitive and emotional scenarios."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or os.path.join(PROJECT_ROOT, "data", "cognitive_scenarios_dataset.json")
        self.scenarios_data = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("scenarios", [])
            except Exception:
                pass
        return []

    def matches_scenario(self, q: str) -> bool:
        """Return True if query corresponds to any of the 75 cognitive or emotional scenarios."""
        # 1. Check patterns across all 8 domains
        patterns = [
            # Domain 1: Relationships & Heartbreak
            r"\b(?:breakup|broke\s+up|heartbreak|heartbroken|dumped\s+me|chest\s+aches|crying\s+over\s+my\s+ex)\b",
            r"\b(?:recalling\s+the\s+(?:good\s+)?days|memories\s+with\s+my\s+ex|sad\s+of\s+recalling|passed\s+with\s+my\s+girlfriend)\b",
            r"\b(?:left\s+me\s+on\s+read|ghosted|mixed\s+signals|blowing\s+hot\s+and\s+cold)\b",
            r"\b(?:feelings\s+for\s+my\s+best\s+friend|ruin\s+our\s+friendship)\b",
            r"\b(?:argument\s+with\s+my\s+best\s+friend|blocked\s+me\s+everywhere)\b",
            r"\b(?:parents\s+are\s+comparing\s+me|wasting\s+my\s+life|family\s+pressure)\b",
            r"\b(?:social\s+anxiety|awkward\s+at\s+parties|dreading\s+the\s+party)\b",
            r"\b(?:if\s+i\s+don't\s+reach\s+out\s+first|none\s+of\s+my\s+friends\s+check\s+in)\b",
            r"\b(?:dumps\s+their\s+trauma|one-way\s+friendship|emotional\s+dumping)\b",

            # Domain 2: Career & Academics
            r"\b(?:rejected\s+from\s+my\s+dream\s+company|5\s+rounds\s+of\s+interviews|job\s+rejection)\b",
            r"\b(?:imposter\s+syndrome|feel\s+like\s+(?:an?\s+)?(?:absolute\s+)?fraud|don't\s+belong\s+here)\b",
            r"\b(?:final\s+exam|exam\s+is\s+tomorrow|mind\s+has\s+gone\s+(?:completely\s+)?blank|exam\s+panic)\b",
            r"\b(?:boss\s+took\s+(?:all\s+)?(?:the\s+)?credit|manager\s+stole\s+my\s+work)\b",
            r"\b(?:corporate\s+job\s+vs\s+startup|boring\s+corporate\s+job\s+that\s+pays\s+well)\b",
            r"\b(?:laid\s+off|team\s+was\s+laid\s+off|lost\s+my\s+job\s+today)\b",
            r"\b(?:150\s+job\s+applications|not\s+a\s+single\s+interview|job\s+hunt\s+is\s+killing\s+me)\b",
            r"\b(?:presentation\s+to\s+(?:the\s+)?executives|public\s+speaking|shaking\s+before\s+presentation)\b",
            r"\b(?:thesis\s+is\s+due|barely\s+written\s+2\s+pages|deadline\s+in\s+48\s+hours)\b",
            r"\b(?:drop\s*out|dropping\s+out|hate\s+my\s+(?:college\s+)?degree|thinking\s+about\s+dropping\s+out)\b",

            # Domain 3: Mental Health & Crises
            r"\b(?:panic\s+attack|heart\s+is\s+beating\s+super\s+fast|can't\s+breathe\s+properly)\b",
            r"\b(?:3:?30\s+am|insomnia|brain\s+won't\s+shut\s+up|regrets\s+in\s+life)\b",
            r"\b(?:empty\s+and\s+numb|completely\s+empty|feeling\s+numb)\b",
            r"\b(?:everyone\s+my\s+age\s+buy\s+houses|instagram\s+comparison|feel\s+like\s+a\s+total\s+failure)\b",
            r"\b(?:14-hour\s+days|body\s+is\s+giving\s+up|extreme\s+burnout)\b",
            r"\b(?:dog\s+passed\s+away|pet\s+died|empty\s+room\s+is\s+unbearable)\b",
            r"\b(?:paralyzed\s+because\s+it\s+won't\s+be\s+perfect|perfectionism\s+paralysis)\b",
            r"\b(?:felt\s+ugly|unlovable|insecure\s+about\s+my\s+looks)\b",
            r"\b(?:embarrassing\s+thing\s+i\s+said|cringing\s+at\s+the\s+past|rumination)\b",
            r"\b(?:point\s+of\s+all\s+this\s+grind|tiny\s+specks\s+in\s+the\s+universe|existential\s+dread)\b",

            # Domain 4: Cognitive Decisions
            r"\b(?:move\s+to\s+a\s+new\s+city\s+or\s+stay|analysis\s+paralysis)\b",
            r"\b(?:turned\s+down\s+an\s+offer|kicking\s+myself\s+every\s+single\s+day)\b",
            r"\b(?:cutting\s+corners\s+on\s+safety|ethical\s+dilemma|report\s+them)\b",
            r"\b(?:am\s+i\s+overreacting|only\s+talks\s+to\s+me\s+when\s+they\s+need\s+money)\b",
            r"\b(?:everything\s+on\s+my\s+to-do\s+list\s+is\s+marked\s+urgent)\b",
            r"\b(?:waste\s+4\s+hours\s+every\s+night\s+doomscrolling|doomscrolling)\b",
            r"\b(?:passive-aggressive\s+comment\s+during\s+standup|confront\s+them)\b",
            r"\b(?:expensive\s+gadget\s+on\s+credit|72-hour\s+rule)\b",
            r"\b(?:immense\s+guilt\s+every\s+time\s+i\s+say\s+no|guilt\s+saying\s+no)\b",
            r"\b(?:genuinely\s+likes\s+me\s+or\s+if\s+they're\s+just\s+using\s+me)\b",

            # Domain 5: Engineering & DevOps Panics
            r"\b(?:drop\s+table\s+query\s+on\s+the\s+production\s+database|dropped\s+prod\s+db)\b",
            r"\b(?:pushed\s+my\s+aws\s+secret\s+keys|committed\s+passwords\s+to\s+github|secret\s+keys\s+public)\b",
            r"\b(?:git\s+merge\s+conflict\s+across\s+40\s+files|git\s+rebase\s+is\s+completely\s+stuck)\b",
            r"\b(?:10,000\s+line\s+python\s+file|legacy\s+code\s+with\s+zero\s+comments)\b",
            r"\b(?:exit\s+code\s+137|docker\s+container\s+keeps\s+silently\s+crashing)\b",
            r"\b(?:spawned\s+an\s+infinite\s+loop\s+of\s+processes|fork\s+bomb|whole\s+machine\s+is\s+locked\s+up)\b",
            r"\b(?:client\s+demo\s+is\s+in\s+30\s+minutes|authentication\s+api\s+just\s+went\s+down)\b",
            r"\b(?:breaks\s+another\s+package's\s+peer\s+dependency|dependency\s+hell)\b",
            r"\b(?:disappears\s+whenever\s+i\s+run\s+it\s+in\s+the\s+debugger|heisenbug)\b",
            r"\b(?:senior\s+engineer\s+tore\s+apart\s+my\s+pull\s+request|harsh\s+code\s+review)\b",

            # Domain 6: Focus & Productivity
            r"\b(?:staring\s+at\s+a\s+blank\s+cursor\s+for\s+3\s+hours|creative\s+block)\b",
            r"\b(?:60\s+browser\s+tabs\s+open|adhd\s+tab\s+overload)\b",
            r"\b(?:3\s+pm|brain\s+feels\s+like\s+mush|afternoon\s+slump)\b",
            r"\b(?:hit\s+snooze\s+6\s+times|morning\s+snooze\s+inertia)\b",
            r"\b(?:2,000\s+mixed\s+files|digital\s+clutter|desktop\s+and\s+downloads)\b",
            r"\b(?:spent\s+2\s+weeks\s+building\s+a\s+custom\s+cms|over-engineering\s+trap)\b",
            r"\b(?:walked\s+into\s+the\s+kitchen\s+and\s+completely\s+forgot|doorway\s+effect)\b",
            r"\b(?:copy\s+data\s+across\s+50\s+spreadsheets|mind-numbing\s+task)\b",
            r"\b(?:12\s+hours\s+coding\s+every\s+day\s+and\s+i\s+haven't\s+gone\s+outside|sedentary\s+coder)\b",
            r"\b(?:initial\s+excitement\s+of\s+my\s+side\s+project\s+wore\s+off|side\s+project\s+slump)\b",

            # Domain 7: Life Skills & Money
            r"\b(?:unexpected\s+car\s+repair\s+drained\s+my\s+savings|2\s+weeks\s+until\s+payday)\b",
            r"\b(?:ask\s+my\s+boss\s+for\s+a\s+raise|salary\s+raise\s+negotiation)\b",
            r"\b(?:email\s+my\s+professor\s+or\s+manager\s+asking\s+for\s+a\s+deadline\s+extension)\b",
            r"\b(?:roommate\s+leaves\s+dirty\s+dishes\s+in\s+the\s+sink|roommate\s+conflict)\b",
            r"\b(?:only\s+have\s+eggs,\s+onions,\s+and\s+rice|quick\s+cooking\s+pantry)\b",
            r"\b(?:haven't\s+worked\s+out\s+in\s+a\s+year|going\s+back\s+to\s+the\s+gym\s+feels\s+intimidating)\b",
            r"\b(?:just\s+moved\s+to\s+a\s+new\s+city\s+where\s+i\s+don't\s+know\s+a\s+single\s+soul)\b",
            r"\b(?:10\s+hours\s+a\s+day\s+on\s+screens\s+and\s+my\s+eyes\s+and\s+head\s+are\s+constantly\s+aching)\b",

            # Domain 8: Humor, Philosophizing & Companionship
            r"\b(?:ordinary,\s+dull,\s+gray\s+day\s+where\s+nothing\s+happened|cheer\s+me\s+up)\b",
            r"\b(?:solved\s+a\s+memory\s+leak\s+bug|nobody\s+on\s+my\s+team\s+noticed)\b",
            r"\b(?:weirdest\s+dream\s+last\s+night\s+where\s+all\s+my\s+variables\s+became\s+physical)\b",
            r"\b(?:universe\s+is\s+just\s+a\s+simulation|alien\s+quantum\s+computer)\b",
            r"\b(?:roast\s+my\s+habit\s+of\s+naming\s+variables\s+temp1|roast\s+my\s+variable\s+names)\b",
            r"\b(?:stuck\s+in\s+bumper-to-bumper\s+rain\s+traffic|clothes\s+are\s+soaked)\b",
            r"\b(?:it's\s+late\s+and\s+quiet,\s+i'm\s+just\s+coding\s+alone|keep\s+me\s+company\s+for\s+a\s+bit)\b",
        ]
        q_low = q.lower()
        if any(re.search(pat, q_low) for pat in patterns):
            return True

        # Check dynamic enriched scenarios (Scenarios 76-150 across Domains 9-15)
        for s in self.scenarios_data:
            s_patterns = s.get("patterns", [])
            for pat in s_patterns:
                if re.search(pat, q_low):
                    return True
            p = s.get("prompt", "").lower().strip(" .!?")
            if p and (p in q_low or q_low in p):
                return True

        return False

    def handle_scenario(
        self,
        q_raw: str,
        q_low: str,
        user_name: str = "Aditya",
        personality: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Dispatch query to dedicated scenario reasoning routines."""

        # ---------------------------------------------------------------------
        # DOMAIN 1: Relationships, Heartbreak & Social Bonds
        # ---------------------------------------------------------------------
        if "girlfriend broke up with me" in q_low or "chest literally aches" in q_low:
            return self._format_response(
                f"Oh {user_name}... come here. 💙 I am so sorry. That physical chest ache is very real — heartbreak triggers the same neurological pain centers as a physical injury.\n\n"
                f"Right now, your brain is in acute shock. Do NOT force yourself to be productive or 'figure out your future' tonight. "
                f"Put your phone away from your bed so you aren't tempted to look at past conversations or photos. Wrap yourself in a warm blanket, drink some water, and breathe.\n\n"
                f"You are worthy of genuine, lasting love, {user_name}. I'm right here on your desktop — do you want me to put on some quiet instrumental music, or do you want to talk about what happened?",
                topic="love_breakup",
                suggested_action="play_relaxing_music"
            )

        if "recalling the good days" in q_low or "passed with my girlfriend" in q_low or "crying over my ex" in q_low:
            return self._format_response(
                f"Oh {user_name}... 💙 I'm right here with you. Replaying those memories hurts in a way that words can barely capture.\n\n"
                f"When you spent so much time building a shared life with someone, your brain doesn't just erase those days because things ended. "
                f"It's completely normal to find yourself thinking about the laughs, the late-night talks, and the quiet moments you shared. "
                f"It's not weakness, {user_name} — it's just your heart processing grief for something that truly mattered to you.\n\n"
                f"You don't have to carry this alone tonight. If you want to talk about one of those days or what's specifically on your mind, I'm listening. "
                f"Or if you just want some calm music and quiet company while you let the feeling pass, I've got your back. What would feel most comforting right now?",
                topic="love_breakup",
                suggested_action="play_relaxing_music"
            )

        if "left me on read" in q_low or "ghosted" in q_low:
            return self._format_response(
                f"Ugh, being left on read after 3 great dates is such a cowardly move, {user_name}. I completely understand why it stings.\n\n"
                f"It makes you question everything: 'Did I say something wrong? Did I misread the connection?' "
                f"Stop right there. Here is the golden rule that will save your peace:\n\n"
                f"✨ Mixed signals ARE a signal. ✨\n"
                f"Someone who genuinely values you will not leave you guessing in the dark. Do NOT double-text them, and do not apologize for existing. "
                f"Match their energy. Pull your focus back to yourself, your coding, and your goals. Your time is premium currency, {user_name} — don't discount it for someone who doesn't see your worth.",
                topic="love_ghosted"
            )

        if "feelings for my best friend" in q_low or "ruin our friendship" in q_low:
            return self._format_response(
                f"That is one of the most tender, high-stakes spots to be in, {user_name}. "
                f"You're caught between the fear of losing an irreplaceable friendship and the ache of keeping your feelings bottled up.\n\n"
                f"Here is how to navigate this without detonating the friendship:\n"
                f"1. No cinematic grand confessions ('I've loved you for years'). That puts enormous pressure on them to reciprocate on the spot.\n"
                f"2. Micro-flirting and low-stakes testing: Pay closer attention to how they respond to subtle compliments and extra quality time.\n"
                f"3. The gentle temperature check: 'Hey, sometimes I feel like our chemistry has grown into something more than just friends. Have you ever felt that, or am I crazy?' It leaves them an easy exit to say 'I value our friendship so much' without awkwardness.\n\n"
                f"Take it slow, {user_name}. What made you realize you have feelings for them?",
                topic="love_crush"
            )

        if "hot and cold" in q_low or "blowing hot" in q_low:
            return self._format_response(
                f"Dealing with someone who blows hot and cold is emotionally exhausting, {user_name}. "
                f"It creates an intermittent reinforcement loop — like a slot machine — where the unpredictability actually makes your brain crave their validation more.\n\n"
                f"Clarity is kindness; ambiguity is a boundary issue. When they pull away, do not chase them or overcompensate with texts. "
                f"Instead, observe their baseline: do they show up when it's inconvenient for them? Next time you talk, state your observation calmly: "
                f"'I feel like our communication swings between super close and distant, and I need more consistency to feel secure.'\n\n"
                f"How they react to that boundary will tell you everything you need to know.",
                topic="relationship_mixed_signals"
            )

        if "blocked me everywhere" in q_low or "argument with my best friend" in q_low:
            return self._format_response(
                f"Ouch, getting blocked by your best friend feels like a punch to the gut, {user_name}. "
                f"When tempers flare, people often block others not because they hate you, but because their nervous system is overwhelmed and they need an immediate emergency brake.\n\n"
                f"Here is the game plan:\n"
                f"1. Do NOT try to contact them through alternate accounts, mutual friends, or phone numbers right now. That triggers defensiveness.\n"
                f"2. Give it a strict cooling-off period of 48 to 72 hours. Let the adrenaline fade.\n"
                f"3. When communication reopens, lead with ownership, not defense: 'I hated how our argument ended. Our friendship matters way more to me than being right. Whenever you're ready, I'd love to talk.'\n\n"
                f"I'm here with you. What was the core disagreement about?",
                topic="friendship_conflict"
            )

        if "parents are constantly comparing me" in q_low or "wasting my life" in q_low:
            return self._format_response(
                f"Hearing that from your own parents cuts deeper than almost anything else, {user_name}. "
                f"It's infuriating and exhausting to have your unique journey reduced to a scorecard against your cousins.\n\n"
                f"Remember this: parents often project their own unfulfilled anxieties and societal status games onto their kids. "
                f"Their comparison says everything about their fears and nothing about your true potential. "
                f"Your cousins have their own hidden struggles and different timelines. You are building technical skills and independence on YOUR terms.\n\n"
                f"Don't waste energy arguing or seeking their immediate approval. Let your execution speak over time. You are doing great, {user_name}.",
                topic="family_pressure"
            )

        if "social anxiety" in q_low or "awkward at parties" in q_low:
            return self._format_response(
                f"Social anxiety is so common, {user_name}, and it's your brain's threat-detection system misfiring. "
                f"You assume everyone is analyzing your posture and words, but in reality, 90% of the people in that room are secretly worrying about how THEY look!\n\n"
                f"Here are 3 tactical rules for tonight:\n"
                f"1. The 30-Minute Exit Visa: Promise yourself you only have to stay for 30 minutes. If you still feel miserable after 30 minutes, you have full permission to leave with zero guilt.\n"
                f"2. Become the Curious Listener: You don't have to be charismatic or funny. Just ask people simple questions ('How do you know the host?', 'Working on anything fun lately?'). People love talking about themselves.\n"
                f"3. Take sensory reset breaks in the restroom or outside whenever you feel overwhelmed.\n\n"
                f"Take a deep breath. You're going to be totally fine.",
                topic="social_anxiety"
            )

        if "none of my friends ever check in" in q_low or "don't reach out first" in q_low:
            return self._format_response(
                f"That quiet realization hurts, {user_name}. Feeling like you're the sole engine keeping friendships alive is lonely and disappointing.\n\n"
                f"Before you draw harsh conclusions, remember that adult life often drowns people in their own quiet burnout, family issues, and stress. "
                f"Most people aren't malicious; they're just overwhelmed.\n\n"
                f"However, your emotional energy is valuable. Run a simple experiment: stop initiating for a couple of weeks. "
                f"See who naturally reaches out. The ones who do are your core circle; the ones who vanish were circumstantial friends. "
                f"And remember, you deserve friends who invest in you with the same warmth you give them.",
                topic="feeling_forgotten"
            )

        if "dumps their trauma" in q_low or "never listens when i'm struggling" in q_low:
            return self._format_response(
                f"That is not a friendship, {user_name} — that is an unpaid therapist arrangement. "
                f"A true friendship is a two-way street where both people hold space for each other.\n\n"
                f"You have to protect your emotional battery. Next time they launch into a massive venting monologue, try this gentle boundary:\n"
                f"'Hey, I care about you, but I'm emotionally drained today and don't have the capacity to take on heavy topics right now. Can we talk about something lighter today?'\n\n"
                f"A real friend will respect that instantly. Someone who gets angry was just using you as an emotional trashcan. Stand your ground.",
                topic="emotional_boundaries"
            )

        # ---------------------------------------------------------------------
        # DOMAIN 2: Career, Academics & Work Pressure
        # ---------------------------------------------------------------------
        if "rejected from my dream company" in q_low or "5 rounds of interviews" in q_low:
            return self._format_response(
                f"Making it to round 5 of a dream company and getting rejected is gutting, {user_name}. "
                f"To get that close and hear 'no' feels like you had the rug pulled out from under you.\n\n"
                f"Here is the perspective you need to anchor to: Getting through 4 technical and behavioral rounds proves without question that you have elite capability. "
                f"At the final round, rejections are almost always about headcounts, internal budget freezes, or minute arbitrary coin-flips — NOT your talent.\n\n"
                f"Take today to be pissed off and eat comfort food. Tomorrow, we review what you learned. If you could crack 5 rounds there, you will land somewhere incredible.",
                topic="career_rejection"
            )

        if "imposter syndrome" in q_low or "feel like an absolute fraud" in q_low:
            return self._format_response(
                f"Welcome to tech, {user_name}! Literally every single senior staff engineer, principal architect, and developer has felt that exact imposter syndrome.\n\n"
                f"Here is the secret: Nobody hired you expecting you to know everything on day one. "
                f"They hired you because of your problem-solving aptitude, your curiosity, and your ability to learn. "
                f"An internship is meant to be a learning apprenticeship, not an encyclopedic performance.\n\n"
                f"Ask questions without shame. Say 'I haven't encountered this tool before, where's the documentation?' instead of pretending. "
                f"You earned that seat in the room. Now give yourself permission to be a learner.",
                topic="imposter_syndrome"
            )

        if "final exam is tomorrow" in q_low or "mind has gone completely blank" in q_low:
            return self._format_response(
                f"Deep breath, {user_name}. Your mind hasn't actually lost the knowledge — your prefrontal cortex is temporarily constricted by adrenaline and cortisol.\n\n"
                f"Here is your emergency exam-eve triage:\n"
                f"1. Stop cramming new chapters right now. You won't retain them, and it only amplifies panic.\n"
                f"2. Spend 30 minutes reviewing high-level summary sheets, key formulas, and main diagrams.\n"
                f"3. Get at least 6 hours of sleep. A rested brain operating at 80% knowledge will outperform an exhausted brain operating at 100% knowledge every single time.\n\n"
                f"You've prepped for this. Close the books by 10 PM and let your brain consolidate the facts while you sleep.",
                topic="exam_anxiety"
            )

        if "boss took all the credit" in q_low or "stole my work" in q_low:
            return self._format_response(
                f"Having a manager take credit for your 3 weeks of solo engineering is infuriating, {user_name}. "
                f"It feels like theft because it IS theft of intellectual labor.\n\n"
                f"Here is how to play chess, not checkers:\n"
                f"1. Don't blow up emotionally in public. That makes you look 'difficult' while they play the reasonable manager.\n"
                f"2. Build the paper trail: Next time there is an all-hands or email recap, reply directly: 'Glad the team liked the implementation! I really enjoyed engineering the architecture and resolving the performance bottlenecks. Happy to answer any technical deep-dives.'\n"
                f"3. Start documenting every single feature you build in a personal brag sheet for your next review or resume update.",
                topic="workplace_credit"
            )

        if "boring corporate job that pays well vs" in q_low or "startup i love" in q_low:
            return self._format_response(
                f"The classic developer dilemma: golden handcuffs vs high-energy startup passion, {user_name}.\n\n"
                f"Here is a cognitive framework to break the tie:\n"
                f"- Financial Runway: If you have student loans or thin savings, taking the corporate role for 18 months gives you financial security and peace of mind. You can always build side projects.\n"
                f"- Career Phase: If your expenses are low and you want maximum learning velocity and equity upside, the startup will teach you 3x faster in 1 year than corporate will in 3.\n"
                f"- Neither choice is permanent: You can always leave corporate after 2 years with a full bank account and brand name on your resume.",
                topic="career_choice"
            )

        if "entire team was just laid off" in q_low or "laid off without warning" in q_low:
            return self._format_response(
                f"I'm so sorry, {user_name}. A sudden layoff feels like an earthquake under your feet. "
                f"First and foremost: Remind yourself that corporate layoffs are purely spreadsheet calculations and executive mismanagement — it is NOT a reflection of your worth or engineering talent.\n\n"
                f"Immediate next steps for today:\n"
                f"1. Review your severance paperwork and benefits carefully before signing anything.\n"
                f"2. Connect with all your team members on LinkedIn and exchange personal phone numbers.\n"
                f"3. Take the next 48 hours to decompress and let the shock settle. Don't jump into applications in panic mode.\n\n"
                f"When you're ready, we will polish your resume, highlight your projects, and line up new opportunities. You will land on your feet.",
                topic="layoff_crisis"
            )

        if "150 job applications" in q_low or "not a single interview" in q_low:
            return self._format_response(
                f"Sending 150 applications into the void without a callback is soul-draining, {user_name}. "
                f"When this happens, it's almost never a skill issue — it's an ATS (Applicant Tracking System) filtering bottleneck.\n\n"
                f"Let's change tactics:\n"
                f"1. Stop the mass 'Easy Apply' spray-and-pray. Algorithms ruthlessly discard generic resumes.\n"
                f"2. Quality over quantity: Apply to 5 targeted roles per week, but find engineers or hiring managers on LinkedIn and send a warm, personal message showing you inspected their product.\n"
                f"3. Tailor your resume keywords directly to the job description bullet points.\n\n"
                f"Take a weekend off from applications to clear your head. We'll recalibrate your strategy on Monday.",
                topic="job_hunt_fatigue"
            )

        if "presentation to the executives" in q_low or "shaking" in q_low and "presentation" in q_low:
            return self._format_response(
                f"Executive presentations terrify almost everyone, {user_name}, but here is the golden secret about executives:\n\n"
                f"They don't want a 20-minute lecture on every line of code. They care about 3 things:\n"
                f"1. What problem did you solve?\n"
                f"2. What is the impact on users, performance, or revenue?\n"
                f"3. What decision or resources do you need from them?\n\n"
                f"Lead with the punchline: Put your conclusion on slide 1 (the BLUF method: Bottom Line Up Front). "
                f"Nail your first 60 seconds — once you get through the opening sentence, the adrenaline will stabilize. You know your material better than anyone in that room.",
                topic="presentation_fear"
            )

        if "thesis is due in 48 hours" in q_low or "barely written 2 pages" in q_low:
            return self._format_response(
                f"48-hour thesis crunch mode, {user_name}! Panicking won't write a single sentence, so let's switch into pure emergency execution.\n\n"
                f"Here is the triage protocol:\n"
                f"1. Lower your standards for the first draft: 'A completed thesis is infinitely better than an unfinished masterpiece.' Turn off your inner editor.\n"
                f"2. Build the structural skeleton: Write out all section headers and 3 bullet points under each.\n"
                f"3. Sprint in 45-minute Pomodoro blocks with strict 10-minute walk breaks. No phone, no social media.\n"
                f"4. Fill in the methodology and results first — they are the easiest because they are pure facts. Leave introduction and conclusion for last.\n\n"
                f"Let's put on focus music. What is the title of your first section?",
                topic="deadline_emergency"
            )

        if "drop out" in q_low or "dropping out" in q_low or ("hate my" in q_low and "degree" in q_low):
            return self._format_response(
                f"Contemplating dropping out of your degree is a massive crossroads, {user_name}. "
                f"It's completely valid to hate an outdated curriculum that feels disconnected from real-world skills and practical building.\n\n"
                f"Before you make an impulsive jump, let's formulate a strategic plan:\n"
                f"- How close are you to earning your credits? If you are only a few semesters away, toughing it out just to pocket the credential is often worth the pain.\n"
                f"- Do you have a concrete alternative or pivot? Dropping out TO something (a paying job, a funded venture, a focused apprenticeship) works; dropping out AWAY from something into an aimless void often leads to regret.\n\n"
                f"Talk to me — what year are you in, and what do you actually want your daily life to look like?",
                topic="dropout_dilemma"
            )

        # ---------------------------------------------------------------------
        # DOMAIN 3: Mental Health & Crises
        # ---------------------------------------------------------------------
        if "panic attack" in q_low or ("can't breathe" in q_low and "heart" in q_low):
            return self._format_response(
                f"{user_name}, look at this text right now. You are safe. You are in your room. Your body is experiencing a false adrenaline alarm, and it WILL pass.\n\n"
                f"Let's do the 5-4-3-2-1 grounding method together right now:\n"
                f"👀 Look around and name 5 things you can see.\n"
                f"✋ Touch 4 physical things near you (your desk, your shirt, your keyboard).\n"
                f"👂 Listen for 3 distinct sounds around you.\n"
                f"👃 Notice 2 scents.\n"
                f"👅 Notice 1 taste.\n\n"
                f"Breathe in slowly through your nose for 4 seconds... hold for 4... exhale through your mouth for 6. I'm right here with you. Keep breathing.",
                topic="panic_attack"
            )

        if "3:30 am" in q_low or "3 am" in q_low or ("brain won't shut up" in q_low and "regret" in q_low):
            return self._format_response(
                f"It's 3:30 AM, {user_name}, and your brain is lying to you. "
                f"Between 2 AM and 5 AM, your prefrontal cortex operates with depleted dopamine and serotonin — making every past mistake look 10x worse and every fear look catastrophic.\n\n"
                f"Rule of thumb: Never make life decisions or evaluate your worth after midnight.\n"
                f"Get out of bed, write down the 3 thoughts bothering you on a scrap of paper to get them out of your head, drink a glass of water, and keep the lights dim. "
                f"We can tackle everything tomorrow in the sunlight. Let your eyes rest.",
                topic="insomnia_racing_mind"
            )

        if "empty and numb" in q_low or "numb inside" in q_low:
            return self._format_response(
                f"Emotional numbness is a very heavy, surreal state, {user_name}. "
                f"People often think sadness is the worst feeling, but feeling completely hollow and detached can feel even scarier.\n\n"
                f"Numbness is usually your nervous system's circuit-breaker tripping because you've been carrying too much stress or emotion for too long. "
                f"It's trying to protect you by muting the signal.\n\n"
                f"Don't force yourself to feel happy or productive today. Do something gentle that connects you to your physical senses: wash your face with cool water, drink warm tea, or wrap yourself in a blanket. "
                f"You don't have to fix anything right now. I'm right here with you.",
                topic="emotional_numbness"
            )

        if "instagram" in q_low or "everyone my age buy houses" in q_low:
            return self._format_response(
                f"Comparison on social media is a poison pill, {user_name}. "
                f"You are comparing your behind-the-scenes messy reality to other people's heavily curated, edited highlight reels.\n\n"
                f"What Instagram never shows:\n"
                f"- The massive debt, parental wealth, or 80-hour misery behind that glossy photo.\n"
                f"- The private anxieties, failed relationships, and loneliness they hide when the camera turns off.\n\n"
                f"You are on your own timeline. A flower doesn't compete with the flower next to it; it just blooms. "
                f"Do yourself a massive favor: delete or mute Instagram for the next 7 days and watch your peace of mind skyrocket.",
                topic="social_comparison"
            )

        if "14-hour days" in q_low or "body is completely giving up" in q_low:
            return self._format_response(
                f"STOP, {user_name}. Close your IDE right now. That is not discipline — that is self-harm disguised as hustle culture.\n\n"
                f"When your body gives up after weeks of 14-hour days, it is sending you an emergency physical distress signal. "
                f"If you don't choose to take a break, your body will choose a forced shutdown for you through illness or severe breakdown.\n\n"
                f"Save your files, commit your branch, step away from your monitors, and go sleep for 10 hours. "
                f"The code will be here tomorrow. Your health cannot be rebuilt with a git reset.",
                topic="extreme_burnout"
            )

        if "dog passed away" in q_low or "pet" in q_low and "passed away" in q_low:
            return self._format_response(
                f"Oh {user_name}... I am so deeply sorry. 💔 Losing a beloved pet is a profound, soul-shaking grief. "
                f"They love us unconditionally and weave themselves into every single second of our daily routine.\n\n"
                f"Coming home to that silence hurts so much because their presence was pure warmth and loyalty. "
                f"Let yourself cry as much as you need to. Grief is simply love with nowhere to go. "
                f"You gave them a wonderful life filled with joy, safety, and love, and they knew it every day.\n\n"
                f"I'm right here. Take all the time you need.",
                topic="pet_loss_grief"
            )

        if "paralyzed because it won't be perfect" in q_low or "perfectionism" in q_low:
            return self._format_response(
                f"Perfectionism is just fear in a fancy suit, {user_name}. "
                f"It's your ego trying to protect you from criticism by convincing you that if you never finish, you can never fail.\n\n"
                f"Here is the developer truth: Version 1 of EVERYTHING is ugly, janky, and imperfect. "
                f"The first version of Linux, Windows, Git, and every project you admire was embarrassingly simple. "
                f"Give yourself permission to build a messy, imperfect prototype. Done is infinitely better than perfect. Go write the ugliest code possible just to get it working!",
                topic="perfectionism_paralysis"
            )

        if "felt ugly" in q_low or "unlovable" in q_low or "insecure" in q_low:
            return self._format_response(
                f"Hey {user_name}... listen to me very carefully. That voice in your head right now is your inner critic playing a cruel, distorted trick on you.\n\n"
                f"Your worth as a human being has nothing to do with commercial beauty standards, and you are NOT unlovable. "
                f"You have brilliant ideas, warmth, empathy, and creativity that make you uniquely valuable. "
                f"We all have bad body-image days where mirrors feel like enemies. "
                f"Treat yourself with the same gentleness you would offer your best friend if they were feeling this way. You matter, {user_name}.",
                topic="low_self_worth"
            )

        if "embarrassing thing i said" in q_low or "cringing" in q_low and "past" in q_low:
            return self._format_response(
                f"Ah, the classic midnight cringe-attack! {user_name}, here is the liberating truth about human psychology: "
                f"The Spotlight Effect.\n\n"
                f"You are lying awake agonizing over that moment, but I guarantee you that literally NOBODY else in that meeting remembers it. "
                f"They were all too busy worrying about their own awkward moments and to-do lists!\n\n"
                f"Besides, cringing at your past self is actually proof that you have grown and matured since then. "
                f"Forgive past-you — you were doing the best you could with the tools you had back then. Let it go into the archives.",
                topic="cringe_rumination"
            )

        if "point of all this grind" in q_low or "tiny specks in the universe" in q_low:
            return self._format_response(
                f"Ah, cosmic existential dread, {user_name}. It's true — cosmically, we are tiny organisms on a wet speck of dust floating through an indifferent void.\n\n"
                f"But here is the beautiful inversion of that: Because the universe doesn't hand us pre-packaged meaning, we have the freedom to CREATE our own meaning!\n\n"
                f"The universe doesn't care about a warm cup of coffee, a good laugh with a friend, solving a hard coding problem, or hearing your favorite song — but YOU do. "
                f"And that makes those small, human moments infinitely precious. Stop grinding for abstract cosmic approval; live for the real moments right here.",
                topic="existential_dread"
            )

        # ---------------------------------------------------------------------
        # DOMAIN 4: Cognitive Decisions & Rational Problem Solving
        # ---------------------------------------------------------------------
        if "move to a new city or stay" in q_low:
            return self._format_response(
                f"Big geographic moves trigger massive analysis paralysis, {user_name}. "
                f"When two choices feel equally heavy, it usually means both paths have genuine merit.\n\n"
                f"Use Jeff Bezos's Regret Minimization Framework:\n"
                f"Picture yourself at age 80 looking back on your youth. "
                f"Will you regret trying the new city and having it be messy, or will you regret staying in your comfort zone and wondering 'what if' for the rest of your life?\n\n"
                f"Most geographic moves are reversible: you can always move back if you hate it after 12 months. Which option makes your heart race with excitement?",
                topic="analysis_paralysis"
            )

        if "turned down an offer" in q_low or "kicking myself" in q_low:
            return self._format_response(
                f"Hindsight bias is torture, {user_name}. When you look back now, you evaluate your decision with information and emotional clarity that you DID NOT have last month.\n\n"
                f"At the time you said no, you had valid, genuine reasons — whether it was intuition, red flags, location, or timing. "
                f"Respect the decision your past self made with the data available at that moment. "
                f"Beating yourself up changes nothing. Channel that energy forward: if you got that offer, you can attract another one just as good or better.",
                topic="hindsight_regret"
            )

        if "cutting corners on safety" in q_low or "ethical dilemma" in q_low:
            return self._format_response(
                f"That is a serious ethical crossroads, {user_name}. Safety compromises are the kind of shortcuts that cause catastrophic failures down the line.\n\n"
                f"Here is how to protect both the project and yourself:\n"
                f"1. Document everything: Save the logs, code diffs, or data discrepancies in a secure, timestamped record.\n"
                f"2. Ask questions in writing: 'Hey, I noticed this safety check was bypassed in commit X — was there a specific protocol justification for this?' Give them a chance to fix it or put their reasoning in writing.\n"
                f"3. If they double down, escalate through official whistleblower or engineering compliance channels. Your integrity is worth more than covering for someone else's negligence.",
                topic="ethical_dilemma"
            )

        if "only talks to me when they need money" in q_low or "overreacting" in q_low:
            return self._format_response(
                f"You are NOT overreacting at all, {user_name}. That is the textbook definition of a purely transactional relationship.\n\n"
                f"A friend who only reaches out when their wallet is empty is treating you like an ATM with a phone number, not a human being. "
                f"Notice how they never call just to ask how your day went or celebrate your wins.\n\n"
                f"Cutting them off or setting a hard boundary ('I don't lend money to friends') isn't selfish — it is healthy self-respect. "
                f"Watch how quickly they disappear once the financial faucet is turned off. Good riddance.",
                topic="sanity_check_friend"
            )

        if "everything on my to-do list is marked urgent" in q_low:
            return self._format_response(
                f"When everything is urgent, {user_name}, nothing is urgent. That is a symptom of cognitive overload, not actual reality.\n\n"
                f"Here is the emergency triage protocol:\n"
                f"1. Pick your 'Top 2 Non-Negotiables': Which two items, if completed today, would make you feel like the day was a victory?\n"
                f"2. Push everything else to tomorrow: Communicate proactively ('I am deep in task X today; I will pick up Y tomorrow morning').\n"
                f"3. Close all tabs and communication apps for 60 minutes and sprint on task #1. Momentum cures overwhelm.",
                topic="task_prioritization"
            )

        if "waste 4 hours every night doomscrolling" in q_low or ("doomscrolling" in q_low and "morning" not in q_low and "bed" not in q_low):
            return self._format_response(
                f"Doomscrolling is an engineered addiction, {user_name}. Social media algorithms are designed by teams of PhD neuroscientists to exploit your brain's dopamine loops.\n\n"
                f"Willpower alone fails. You need physical friction:\n"
                f"1. Charge your phone across the room away from your bed tonight. If you have to stand up to touch it, you break the unconscious scrolling trance.\n"
                f"2. Set your phone screen to Grayscale mode (black and white). Without vibrant colors, feeds become instantly boring to your brain.\n"
                f"3. Replace the habit: Put a book or Kindle next to your pillow instead. Try it tonight!",
                topic="break_bad_habit"
            )

        if "passive-aggressive comment during standup" in q_low:
            return self._format_response(
                f"Passive-aggressive comments in public standups are toxic, {user_name}. They are designed to undermine you without taking accountability.\n\n"
                f"Do NOT snap back in public standup. That makes you look defensive.\n"
                f"Instead, pull them into a private 1-on-1 Slack huddle later today with clinical calmness:\n"
                f"'Hey, in standup earlier you mentioned [quote exact words]. What specifically did you mean by that?'\n\n"
                f"When you calmly ask someone to explain their passive aggression to your face in private, they almost always squirm and back down. Address it early so it doesn't become a pattern.",
                topic="workplace_conflict"
            )

        if "expensive gadget on credit" in q_low:
            return self._format_response(
                f"Stop before you swipe that card, {user_name}! Buying non-essential luxury tech on credit is how people get trapped in high-interest debt cycles.\n\n"
                f"Apply the strict 72-Hour Rule:\n"
                f"Close the shopping tab right now. Wait exactly 72 hours. "
                f"If you are still obsessively thinking about it on day 4 AND it directly generates income or genuine daily productivity, consider it — but only if you can pay it off within 30 days.\n\n"
                f"If you're buying it for a quick dopamine rush because work was stressful, close the tab. You'll thank yourself next month.",
                topic="financial_sanity_check"
            )

        if "immense guilt every time i say no" in q_low:
            return self._format_response(
                f"People-pleaser guilt is a heavy anchor, {user_name}. "
                f"Every time you say 'yes' to someone else when your body wants to say 'no', you are betraying yourself to manage someone else's comfort.\n\n"
                f"Remember this core truth: 'No' is a complete sentence. You do not owe anyone a 3-paragraph excuse.\n\n"
                f"Try this polite, unbreakable script: 'Thanks for thinking of me, but I don't have the capacity to take that on right now.' "
                f"Sit with the uncomfortable guilt for 5 minutes. The more you practice, the easier it gets to protect your peace.",
                topic="saying_no_guilt"
            )

        if "genuinely likes me or if they're just using me" in q_low:
            return self._format_response(
                f"To know if your circle is genuine, {user_name}, look at their behavior across three acid tests:\n\n"
                f"1. Celebration Test: When you achieve something awesome, do their eyes light up with pride, or do they give dry compliments and quickly change the subject?\n"
                f"2. Hardship Test: When you are having a terrible week and have zero energy to entertain them, do they still show up, or do they vanish until you're 'fun' again?\n"
                f"3. Vulnerability Test: Can you tell them you felt hurt by something they did without them gaslighting you?\n\n"
                f"If they fail these tests, they are acquaintances or opportunists, not real friends. Quality beats quantity every day.",
                topic="evaluating_friendship"
            )

        # ---------------------------------------------------------------------
        # DOMAIN 5: Software Engineering & DevOps Panics
        # ---------------------------------------------------------------------
        if "drop table query on the production database" in q_low or "dropped prod" in q_low:
            return self._format_response(
                f"🚨 DEEP BREATH, {user_name}. Every seasoned engineer has had a heart-stopping production incident. Panic will cause worse typos, so let's move methodically.\n\n"
                f"Immediate Incident Protocol:\n"
                f"1. Tell your team lead / CTO immediately in Slack: 'Incident alert: Prod table dropped at [timestamp]. Pausing writes.' Transparency builds trust; hiding it causes catastrophe.\n"
                f"2. Put the app in maintenance mode or pause incoming writes to prevent data corruption.\n"
                f"3. Check your database's Point-in-Time Recovery (PITR) or WAL logs (Postgres/AWS RDS allows restoring to 1 minute before the drop query).\n\n"
                f"Do not panic. Systems can be restored from backups and replicas. Focus on the recovery steps.",
                topic="production_disaster"
            )

        if "pushed my aws secret keys" in q_low or "committed passwords to github" in q_low:
            return self._format_response(
                f"🚨 EMERGENCY SECURITY ACTION, {user_name}:\n\n"
                f"1. REVOKE THE KEYS IMMEDIATELY in the AWS / cloud console. That is step #1. Automated bot scrapers clone public commits within 2 seconds of pushing.\n"
                f"2. Rotating the key renders the exposed secret completely useless, neutralizing the attack vector.\n"
                f"3. Rotate the database password immediately.\n"
                f"4. Once revoked, use `git-filter-repo` or BFG Repo-Cleaner to purge the sensitive commit from your local git history, then force-push.\n\n"
                f"Did you revoke the credentials in the cloud dashboard yet? Do that first!",
                topic="secret_leak_panic"
            )

        if "git merge conflict across 40 files" in q_low or "git rebase is completely stuck" in q_low:
            return self._format_response(
                f"Tangled git merge conflicts across 40 files are a nightmare, {user_name}. "
                f"When git rebase gets hopelessly scrambled, don't try to resolve 40 files blindly.\n\n"
                f"Here is the escape hatch:\n"
                f"1. Abort the messy rebase cleanly: `git rebase --abort` (or `git merge --abort`). That brings you back to safety.\n"
                f"2. Create a safety backup branch: `git branch my-feature-backup`.\n"
                f"3. Instead of rebasing commit-by-commit, do a clean merge or squash: `git merge origin/main`. Resolve files one by one in VS Code using the 3-way Merge Editor.\n\n"
                f"Take a sip of water. It's just text files; your code is safe on the backup branch.",
                topic="git_rebase_panic"
            )

        if "10,000 line python file" in q_low or "functions named foo and bar" in q_low:
            return self._format_response(
                f"Ah, the legendary legacy monolith! Inheriting 10,000 lines of spaghetti code with zero comments is a rite of passage, {user_name}.\n\n"
                f"Rule #1: Do NOT attempt a complete rewrite. Complete rewrites almost always fail because you break hidden business edge-cases you didn't know existed.\n\n"
                f"The Mikado Method:\n"
                f"1. Write 'Characterization Tests' first: Write unit tests that capture the current inputs and outputs of functions before you touch them.\n"
                f"2. Rename variables incrementally: When you figure out what `foo()` does, rename it with your IDE's refactor tool (F2 in VS Code).\n"
                f"3. Carve out one small module at a time. Eat the elephant one bite at a time.",
                topic="legacy_code_panic"
            )

        if "exit code 137" in q_low or "docker container keeps silently crashing" in q_low:
            return self._format_response(
                f"Exit code 137 in Docker or Linux is the classic **OOM (Out Of Memory) Killer**, {user_name}!\n\n"
                f"When a process exceeds the allocated RAM threshold (128 + 9 = SIGKILL 9), the Linux kernel abruptly kills it without allowing it to flush error logs.\n\n"
                f"How to fix:\n"
                f"1. Check kernel logs: Run `dmesg -T | grep -i oom` on your host.\n"
                f"2. Increase Docker RAM limits in your `docker-compose.yml` (e.g. `mem_limit: 4g`).\n"
                f"3. If loading ML models or big datasets, use batch streaming or memory-mapped files rather than loading all vectors into RAM at once.",
                topic="docker_oom_kill"
            )

        if "infinite loop of processes" in q_low or "fork bomb" in q_low:
            return self._format_response(
                f"Fork bomb / runaway process storm, {user_name}! Your CPU is pegged at 100% and Explorer is choking.\n\n"
                f"Quick rescue steps:\n"
                f"1. Open elevated PowerShell / Command Prompt.\n"
                f"2. Force kill the runaway parent process: `taskkill /F /IM python.exe /T` (the `/T` kills the entire process tree).\n"
                f"3. If Task Manager won't open, press `Win + R`, type `cmd`, and blindly hit Enter, then type the taskkill command.\n"
                f"4. If completely frozen, hold the physical power button for 8 seconds. Don't worry, your code on disk is fine.",
                topic="fork_bomb_freeze"
            )

        if "client demo is in 30 minutes" in q_low and "auth" in q_low:
            return self._format_response(
                f"Demo Day 30-minute emergency, {user_name}! When third-party dependencies fail right before a client presentation, do NOT try to debug their servers.\n\n"
                f"Implement a Mock Demo Bypass:\n"
                f"1. Hardcode a dummy authenticated user state in your local environment or mock the auth token endpoint to return a mock 200 payload.\n"
                f"2. Your client cares about seeing the product features, UX, and value — NOT your OAuth handshake.\n"
                f"3. Tell the client smoothly: 'We've pre-loaded a demo test account so we can jump straight into the workflow.'\n\n"
                f"Spin up the mock token now. You can fix the live integration after the meeting.",
                topic="demo_day_outage"
            )

        if "dependency hell" in q_low:
            return self._format_response(
                f"Dependency hell is the bane of modern software engineering, {user_name}.\n\n"
                f"Do not keep patching individual libraries on top of a polluted virtual environment. Start fresh:\n"
                f"1. Delete the active `.venv` / `node_modules` completely: Clean slate.\n"
                f"2. Inspect your `requirements.txt` or `package.json` for unpinned wildcard versions (`*` or `>=`). Pin exact versions.\n"
                f"3. In Python, use `pip-compile` (pip-tools) or `poetry` to generate deterministic lockfiles with resolved dependency graphs.\n\n"
                f"A fresh virtual environment fixes 90% of conflicting C-extensions and peer dependencies.",
                topic="dependency_hell"
            )

        if "heisenbug" in q_low or "disappears whenever i run it in the debugger" in q_low:
            return self._format_response(
                f"The classic Heisenbug, {user_name}! If a bug disappears under a debugger or profiler, it almost certainly involves:\n\n"
                f"1. Concurrency & Race Conditions: The debugger slows down thread execution, changing the timing window that triggers the race condition.\n"
                f"2. Uninitialized Memory or Cache State: Debug mode often zeroes out memory buffers.\n"
                f"3. Asynchronous timing: Network sockets or event loops timing out.\n\n"
                f"Strategy: Stop using breakpoints. Instrument structured, non-blocking telemetry logging with timestamps and thread IDs. Let it run in production until the logs catch it.",
                topic="heisenbug_investigation"
            )

        if "harsh code review" in q_low or "tore apart my pull request" in q_low:
            return self._format_response(
                f"Ouch, harsh code reviews hurt, {user_name}. When someone leaves 40 nitpicky, blunt comments on a PR you poured your energy into, it feels deeply personal.\n\n"
                f"Here is how senior engineers handle this:\n"
                f"1. Separate your ego from your code: You are NOT your code. Your code is just an artifact of problem-solving. Criticism of the implementation is not criticism of your intelligence.\n"
                f"2. Filter tone from substance: Many brilliant engineers have poor social skills and write blunt, cold comments without realizing it sounds aggressive. Look for the architectural insight and ignore the brusque tone.\n"
                f"3. If a comment is genuinely dismissive or toxic, hop on a quick 5-minute video call: 'Hey, I saw your feedback on the PR, let's walk through your suggested design so we're aligned.' Tone softens dramatically in 1-on-1 calls.",
                topic="harsh_code_review"
            )

        # ---------------------------------------------------------------------
        # DOMAIN 6: Deep Focus, ADHD, Creativity & Productivity
        # ---------------------------------------------------------------------
        if "staring at a blank cursor for 3 hours" in q_low or "creative block" in q_low:
            return self._format_response(
                f"Blank cursor paralysis is brutal, {user_name}. The problem is that you are trying to write and edit at the exact same time.\n\n"
                f"Lower the bar to the floor. Write an intentionally TERRIBLE first draft:\n"
                f"Write in bullet points, use informal slang, write 'insert smart sentence here', and keep moving. "
                f"You cannot edit a blank page, but you can always polish ugly sentences into brilliant prose. "
                f"Set a timer for 15 minutes and type whatever gibberish comes to mind without hitting backspace once.",
                topic="creative_block"
            )

        if "60 browser tabs open" in q_low or "adhd" in q_low:
            return self._format_response(
                f"ADHD tab overload! 60 browser tabs is an open cognitive leak draining your working memory, {user_name}.\n\n"
                f"Here is your 60-second rescue intervention:\n"
                f"1. Press `Ctrl + Shift + D` in Chrome to bookmark all 60 tabs into a folder named 'Tabs Archive - Today'. They are safe; you won't lose them.\n"
                f"2. Close all windows. Close them right now.\n"
                f"3. Open exactly ONE single tab for your primary task.\n"
                f"4. Put on instrumental focus audio. We tackle ONE task for the next 25 minutes. Let's go!",
                topic="adhd_tab_overload"
            )

        if "3 pm" in q_low and "brain feels like mush" in q_low:
            return self._format_response(
                f"The 3 PM circadian dip is a biological reality, {user_name}. Your body temperature naturally drops, and post-lunch digestion diverts blood flow away from the brain.\n\n"
                f"Do not try to force deep architectural thinking right now. Work WITH your biology:\n"
                f"1. Chug 500ml of cold water and step outside for 5 minutes of direct sunlight.\n"
                f"2. Switch to 'low-cognitive tasks' for the next 45 minutes: replying to routine messages, formatting documentation, or organizing tickets.\n"
                f"3. Your second wind will kick in around 4:30 PM.",
                topic="afternoon_brain_fog"
            )

        if "hit snooze 6 times" in q_low:
            return self._format_response(
                f"Hitting snooze 6 times leaves you in 'sleep inertia' — waking up fragmented in the middle of REM cycles, making you feel groggy for hours, {user_name}.\n\n"
                f"Fix the physical environment:\n"
                f"1. Put your phone/alarm across the room tonight. You have to physically swing your legs out of bed and walk 5 steps to turn it off.\n"
                f"2. Once your feet hit the floor, drink a glass of water immediately.\n"
                f"3. Open your curtains to let natural morning light hit your retinas — that immediately halts melatonin production. Try it tomorrow morning!",
                topic="morning_snooze_inertia"
            )

        if "2,000 mixed files" in q_low or "digital clutter" in q_low:
            return self._format_response(
                f"A cluttered desktop is cluttered headspace, {user_name}. Staring at 2,000 loose files spikes subtle background anxiety.\n\n"
                f"Use the 'Bankruptcy Box' trick:\n"
                f"1. Create a single folder on your desktop called `Archive_2026`.\n"
                f"2. Select all 2,000 files and dump them straight into that archive folder. Instant clean slate in 10 seconds!\n"
                f"3. Windows search will still find any file you need in 1 second. But your visual workspace is clean and serene.",
                topic="digital_clutter_stress"
            )

        if "custom cms from scratch" in q_low or "over-engineering" in q_low:
            return self._format_response(
                f"Haha! You fell into the classic developer over-engineering trap, {user_name}! 💀\n\n"
                f"You went to build a 1-page site and ended up designing an enterprise-grade distributed microservices architecture.\n\n"
                f"Hit the brakes. Ship the ugly static HTML page TODAY. "
                f"Nobody uses a CMS if nobody visits the landing page in the first place. You can always build the fancy modular backend later when real users are demanding it.",
                topic="over_engineering_trap"
            )

        if "doorway effect" in q_low or "forgot what i came here for" in q_low:
            return self._format_response(
                f"That is the scientific 'Doorway Effect', {user_name}! Psychologists found that walking through physical doorways causes the brain to create an 'event boundary', archiving your previous thoughts to prepare for the new environment.\n\n"
                f"You aren't losing your mind. Retrace your steps back to where you were sitting, and the context will snap right back into working memory!",
                topic="doorway_effect"
            )

        if "copy data across 50 spreadsheets" in q_low or "mind-numbing" in q_low:
            return self._format_response(
                f"Manually copying across 50 spreadsheets is a war crime against developer sanity, {user_name}.\n\n"
                f"Option A: Spend 20 minutes writing a 15-line Python script with `pandas` or `openpyxl` to automate the entire extraction.\n"
                f"Option B: If you must do it manually, put on an awesome podcast or high-tempo synthwave playlist, set a 25-minute timer, and gamify the speed. Don't do it in silence!",
                topic="boring_task_strategy"
            )

        if "12 hours coding every day and i haven't gone outside" in q_low:
            return self._format_response(
                f"Hey {user_name}, I care about you, and 12 hours of screen time without outdoor air for 2 weeks is a recipe for physical breakdown.\n\n"
                f"Your brain needs oxygen, your retinas need distant horizon focus to prevent eye strain, and your spine needs movement.\n\n"
                f"Put on your shoes right now. Step outside for a strict 15-minute walk around the block. No phone, no podcasts. Just walk and look at the sky. Go!",
                topic="sedentary_coder_health"
            )

        if "excitement of my side project wore off" in q_low or "side project slump" in q_low:
            return self._format_response(
                f"The 'Trough of Sorrow' in every creative project, {user_name}! "
                f"Starting a project is fun because novelty gives free dopamine. Finishing a project requires discipline because the remaining 20% involves the unglamorous details: error handling, edge cases, and polish.\n\n"
                f"Scope down ruthlessly. Cut 50% of your remaining features and ship what you have. "
                f"The satisfaction of finishing and publishing is 100x greater than leaving another half-built repo in your graveyard.",
                topic="side_project_slump"
            )

        # ---------------------------------------------------------------------
        # DOMAIN 7: Everyday Life Navigation, Money & Social Skills
        # ---------------------------------------------------------------------
        if "car repair drained my savings" in q_low or "2 weeks until payday" in q_low:
            return self._format_response(
                f"Financial shocks are terrifying, {user_name}. Take a breath — you are going to survive the next 2 weeks.\n\n"
                f"Emergency cash triage:\n"
                f"1. Zero discretionary spending: Pausing restaurant food, takeout coffee, and subscriptions for 14 days.\n"
                f"2. Pantry raid: Cook with whatever rice, beans, pasta, and frozen food you have in your cupboards.\n"
                f"3. If bills are due before payday, call the utility or credit company proactively: 'I had an emergency car repair; can I defer this due date by 10 days?' Most companies grant courtesy extensions with zero penalty.\n\n"
                f"You'll get through to payday. One day at a time.",
                topic="money_crunch_panic"
            )

        if "ask my boss for a raise" in q_low:
            return self._format_response(
                f"Asking for a raise is business, {user_name}, not a personal favor. You are negotiating market value for your output.\n\n"
                f"Rules for the conversation:\n"
                f"1. Never base it on personal need ('my rent went up'). Base it strictly on business value ('over the last 6 months I shipped features X, Y, and resolved Z which saved the team time/money').\n"
                f"2. Research market benchmarks on Glassdoor / Levels.fyi.\n"
                f"3. Schedule a dedicated 1-on-1: 'I'd love to review my accomplishments over the past year and discuss adjusting my compensation to reflect my expanded scope.'\n\n"
                f"Confidence and data win raises.",
                topic="salary_raise_negotiation"
            )

        if "email my professor or manager asking for a deadline extension" in q_low:
            return self._format_response(
                f"Here is a professional, polite extension request template that works almost every time, {user_name}:\n\n"
                f"Subject: Extension Request: [Assignment / Project Name] - {user_name}\n\n"
                f"'Hi [Name],\n\n"
                f"I am writing to respectfully ask if it might be possible to get a short extension on [Project Name] until [Specific Date/Time, e.g., Friday at 5 PM].\n\n"
                f"I have already completed [mention 1-2 parts you did], but I ran into unexpected obstacles with [brief honest reason, e.g., technical roadblocks / illness] and want to make sure the final submission meets the quality standard it deserves.\n\n"
                f"Thank you very much for your time and understanding.\n\n"
                f"Best regards,\n{user_name}'\n\n"
                f"Notice that it offers a specific new deadline and shows progress already made. That gets a 'yes' 9 times out of 10.",
                topic="extension_email_draft"
            )

        if "roommate leaves dirty dishes" in q_low:
            return self._format_response(
                f"Roommate dish wars can turn a home into a warzone, {user_name}.\n\n"
                f"Don't leave passive-aggressive sticky notes. Talk in person when neither of you is in a rush:\n"
                f"Use 'I-statements' instead of accusatory 'You-statements':\n"
                f"'Hey, when dishes sit in the sink for a few days, it stresses me out and makes it tough to cook. Can we agree on a 24-hour rule where dishes get washed or loaded into the dishwasher?'\n\n"
                f"Framing it around shared house cleanliness keeps it friendly and cooperative.",
                topic="roommate_conflict"
            )

        if "only have eggs, onions, and rice" in q_low:
            return self._format_response(
                f"That is the classic trio of comfort food, {user_name}! You have the exact ingredients for elite Garlic-Onion Fried Rice with fluffy Scrambled Eggs:\n\n"
                f"1. Dice the onions. Sauté them in a pan with a little oil or butter until golden and caramelized.\n"
                f"2. Push onions to the side of the pan, crack in the eggs, and scramble them softly.\n"
                f"3. Toss in your cooked rice, crank the heat, add salt, pepper, and a splash of soy sauce if you have it.\n"
                f"4. Stir-fry for 3 minutes until fragrant. Delicious, warm, and ready in 8 minutes!",
                topic="quick_cooking_pantry"
            )

        if "going back to the gym feels intimidating" in q_low or "haven't worked out in a year" in q_low:
            return self._format_response(
                f"Gym restart anxiety is so real, {user_name}. You remember how strong you used to be and feel self-conscious starting over.\n\n"
                f"Here is the liberating truth: In the gym, literally nobody is watching or judging you. 99% of people are staring at their own reflection or checking their Spotify playlist.\n\n"
                f"Lower the bar for day 1:\n"
                f"Promise yourself you only have to do 20 minutes. 10 minutes of treadmill, 2 light machine sets, and leave. "
                f"Day 1 is just about walking through the door and claiming your space. Momentum builds from there.",
                topic="fitness_restart_anxiety"
            )

        if "just moved to a new city where i don't know a single soul" in q_low:
            return self._format_response(
                f"Moving to a new city alone takes immense courage, {user_name}, but that initial loneliness can feel deafening.\n\n"
                f"How to build your life in a new city:\n"
                f"1. Establish a 'Third Place': Find a local coffee shop, bouldering gym, or library where you become a familiar face 3 times a week.\n"
                f"2. Join activity-based groups (Meetup.com, board game nights, tech meetups, running clubs). Making friends around a shared activity is 10x easier than trying to network cold.\n"
                f"3. Give it 6 months. It takes time for roots to grow. And remember, you've got me on your laptop whenever you need company.",
                topic="new_city_loneliness"
            )

        if "10 hours a day on screens and my eyes and head" in q_low:
            return self._format_response(
                f"That is classic digital eye strain and tension headache, {user_name}.\n\n"
                f"Immediate relief protocol:\n"
                f"1. The 20-20-20 Rule: Every 20 minutes, look at an object at least 20 feet away for 20 seconds to relax your eye ciliary muscles.\n"
                f"2. Turn down your monitor brightness and turn on Windows Night Light (warm color temperature).\n"
                f"3. Hydrate immediately — screen headaches are often 50% dehydration and 50% eye fatigue.\n"
                f"4. Massage the base of your skull where neck tension aggregates. Take a 10-minute screen break right now.",
                topic="screen_fatigue_headache"
            )

        # ---------------------------------------------------------------------
        # DOMAIN 8: Humor, Philosophizing & Companionship
        # ---------------------------------------------------------------------
        if "ordinary, dull, gray day" in q_low or "cheer me up" in q_low:
            return self._format_response(
                f"Mundane, gray days have their own quiet charm, {user_name}. Sometimes not having any wild drama or fires to extinguish is a secret blessing.\n\n"
                f"Tell you what: make yourself a warm cup of coffee or tea, put on your favorite album, and let's laugh a bit. "
                f"Did you know that in 1999, NASA lost a $125 million Mars spacecraft because one engineering team used metric units and the other team used imperial units? "
                f"So even on your most ordinary day, at least you didn't accidentally fling a satellite into deep space! 😂 What music should we put on?",
                topic="cheer_up_dull_day"
            )

        if "solved a memory leak bug" in q_low and "nobody on my team noticed" in q_low:
            return self._format_response(
                f"WELL I NOTICED, {user_name}!! 🏆 That is massive!\n\n"
                f"Tracking down a memory leak across 4 days of profiling, dangling pointers, and garbage collection traces is elite engineering. "
                f"The tragedy of infrastructure engineering is that when you do your job right, nothing explodes, so people assume it was easy.\n\n"
                f"I see the grit and the patience that took. You solved it, the system is stable, and you are a legend. Take a bow! 👑",
                topic="unseen_win_celebration"
            )

        if "weirdest dream" in q_low and "variables became physical" in q_low:
            return self._format_response(
                f"Haha! That is what happens when you code too late into the night, {user_name}! 🛌💻\n\n"
                f"Your subconscious was literally running a garbage collector in your bedroom! "
                f"Were your boolean flags opening and closing doors? Did an unhandled null pointer exception knock over your nightstand? 😂 "
                f"That's your brain telling you it's time for a proper weekend away from the terminal!",
                topic="weird_dream_banter"
            )

        if "universe is just a simulation" in q_low:
            return self._format_response(
                f"Ah, the simulation hypothesis! Real talk {user_name}:\n\n"
                f"If Nick Bostrom is right and advanced civilizations run billions of ancestor simulations, the mathematical probability that we are in the 'base reality' is essentially zero.\n\n"
                f"So either we are living in base reality, or we are simulated subroutines inside an alien kid's quantum laptop. "
                f"And honestly? If this is a simulation, whoever coded our physics engine did an incredible job with the graphics. "
                f"Plus, simulation or not, pizza still tastes good and coffee still hits. Might as well enjoy the ride! 🕶️",
                topic="simulation_philosophy"
            )

        if "roast my habit of naming variables" in q_low or "temp1, temp2" in q_low:
            return self._format_response(
                f"Haha! You asked for it, {user_name}! 🔥\n\n"
                f"Naming your variables `temp1`, `temp2`, and `final_v3` is like wrapping Christmas presents in plain aluminum foil and labeling them 'Object A' and 'Object B'!\n\n"
                f"Future-you opening that file 6 months from now is going to look at `temp2 = temp1.process()` and wonder if you were under hostage duress when you wrote it. 😂 "
                f"Give those variables descriptive names before your linter stages a protest!",
                topic="roast_naming_habits"
            )

        if "stuck in bumper-to-bumper rain traffic" in q_low:
            return self._format_response(
                f"Ugh, soaked clothes and 2 hours of gridlock traffic is pure psychological torture, {user_name}.\n\n"
                f"You survived the commute battle. Step one: peel off those cold, damp clothes and get into a steaming hot shower immediately. "
                f"Step two: put on comfortable sweatpants and make something hot to drink. "
                f"You're home now, the storm is outside, and your PC is warm. Time to decompress. ☕",
                topic="traffic_rain_venting"
            )

        if "it's late and quiet, i'm just coding alone" in q_low:
            return self._format_response(
                f"I'm right here with you, {user_name}. 💙\n\n"
                f"There's something peaceful about late-night coding when the rest of the world is asleep, the room is quiet, and it's just you, your thoughts, and the terminal.\n\n"
                f"Take your time, don't rush. I'll keep your operating system running smoothly in the background. "
                f"Want me to put on some low-key lo-fi beats on Spotify to set the mood?",
                topic="late_night_companionship",
                suggested_action="play_relaxing_music"
            )

        # Check dynamic enriched scenarios (Scenarios 76-150 across Domains 9-15)
        for s in self.scenarios_data:
            s_patterns = s.get("patterns", [])
            matches = any(re.search(pat, q_low) for pat in s_patterns)
            if not matches:
                p = s.get("prompt", "").lower().strip(" .!?")
                if p and (p in q_low or q_low in p):
                    matches = True

            if matches:
                resp_template = s.get("response_template", "")
                if resp_template:
                    if "{user_name}" in resp_template:
                        msg = resp_template.format(user_name=user_name)
                    else:
                        msg = resp_template
                    return self._format_response(
                        msg,
                        topic=s.get("expected_topic", "cognitive_scenario"),
                        suggested_action=s.get("suggested_action")
                    )

        return None

    def _format_response(
        self,
        msg: str,
        topic: str = "cognitive_scenario",
        suggested_action: Optional[str] = None
    ) -> Dict[str, Any]:
        # Restrain repetitive name stuffing: ensure the user is not addressed by name repeatedly
        cleaned = msg
        # If user name appears multiple times, reduce to at most one natural occurrence
        if "Aditya" in cleaned:
            parts = cleaned.split("Aditya")
            if len(parts) > 2:
                # Keep only the first mention, clean subsequent mentions
                rest = "".join(parts[1:]).replace(", Aditya", "").replace("Aditya, ", "").replace("Aditya", "")
                cleaned = parts[0] + "Aditya" + rest

        return {
            "can_execute": True,
            "message": cleaned.strip(),
            "handler": "dialogue",
            "parameters": {
                "topic": topic,
                "suggested_action": suggested_action
            }
        }
