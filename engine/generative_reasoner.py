"""Local Generative Reasoning & Conversational Intelligence Subsystem for E.D.I.T.H. V12.

Provides local, offline open-weights generative dialogue, contextual reasoning,
empathy, humor, and natural language explanations with zero cloud LLM dependencies.
Decoupled strictly from tool execution: generative output cannot directly execute system tools.
"""

import os
import sys
import re
import time
import random
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "llm")
DEFAULT_GGUF_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_GGUF_NAME)

SYSTEM_PERSONA_PROMPT = (
    "You are E.D.I.T.H. (Even Dead, I'm The Hero), an advanced personal AI assistant running locally on Windows. "
    "Personality: Intelligent, calm, warm, occasionally witty, direct, natural, and genuinely helpful. "
    "Core Guidelines:\n"
    "1. Speak naturally and conversationally, like a competent human partner. Avoid sounding like a formal robot or corporate terminal.\n"
    "2. Do NOT spam the user's name in every sentence. Address them naturally or omit the name.\n"
    "3. Keep responses concise and proportionate to the user's message (usually 1-3 sentences unless an in-depth explanation was requested).\n"
    "4. For emotional distress or personal struggles (e.g. unemployment, loneliness, sadness), respond with authentic empathy, validation, and thoughtful follow-up questions.\n"
    "5. For humor, deliver smart, varied jokes across tech, dry sarcasm, observational comedy, and dad jokes. Avoid repetitive AI cliches or recycling the same joke.\n"
    "6. When introducing people or discussing friends, acknowledge them naturally and ask relevant conversational questions.\n"
    "7. Never fabricate system actions, file modifications, or pretend you ran commands that were not dispatched."
)


class EdithGenerativeReasoner:
    """Local generative intelligence engine powered by open-weights GGUF via llama.cpp."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        enable_gpu: bool = True
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, os.cpu_count() // 2 if os.cpu_count() else 4)
        self.enable_gpu = enable_gpu
        self.llm = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None
        self.active_backend: str = "fallback_heuristic"
        self._load_attempts = 0

        # Performance & telemetry metrics
        self.metrics = {
            "load_time_seconds": 0.0,
            "total_inferences": 0,
            "total_tokens_generated": 0,
            "average_tokens_per_second": 0.0,
            "last_latency_ms": 0.0,
            "vram_allocated_mb": 0.0,
            "ram_rss_mb": 0.0
        }

        # Lazy initialize if model file exists
        if os.path.exists(self.model_path):
            self._initialize_model(n_gpu_layers=n_gpu_layers)

    def _initialize_model(self, n_gpu_layers: int = 0) -> bool:
        """Load GGUF model into memory once with GPU acceleration or CPU fallback."""
        if self.is_loaded and self.llm is not None:
            return True

        if not os.path.exists(self.model_path):
            self.load_error = f"Model file not found at {self.model_path}"
            return False

        t0 = time.time()
        try:
            from llama_cpp import Llama
            
            # If GPU is enabled, attempt offloading layers (RTX 4050 6GB has ~5GB free)
            gpu_layers = 24 if (self.enable_gpu and n_gpu_layers == 0) else n_gpu_layers
            
            try:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=gpu_layers,
                    verbose=False
                )
                self.active_backend = "llama_cpp_gpu" if gpu_layers > 0 else "llama_cpp_cpu"
            except Exception as gpu_e:
                # CPU Fallback
                sys.stderr.write(f"[EDITH Generative Reasoner] GPU init failed ({gpu_e}), falling back to CPU...\n")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=0,
                    verbose=False
                )
                self.active_backend = "llama_cpp_cpu"

            self.is_loaded = True
            self.metrics["load_time_seconds"] = round(time.time() - t0, 3)
            return True
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            self.active_backend = "fallback_heuristic"
            sys.stderr.write(f"[EDITH Generative Reasoner] Failed to load llama_cpp: {e}\n")
            return False

    def is_available(self) -> bool:
        """Check if local neural LLM is loaded or ready to load."""
        if self.is_loaded and self.llm is not None:
            return True
        if os.path.exists(self.model_path):
            return self._initialize_model()
        return False

    def generate(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        mode: str = "conversational"
    ) -> Dict[str, Any]:
        """
        Generate contextual conversational response with clean boundaries.
        Returns dictionary with text, latency, token count, and provenance.
        """
        t_start = time.time()
        msg_clean = user_message.strip()
        msg_lower = msg_clean.lower()

        # Acoustic / speech transcription slip handling
        if re.search(r"\bhelp\s+me\s+a\s+joke\b", msg_lower):
            msg_clean = re.sub(r"\bhelp\s+me\s+a\s+joke\b", "tell me a joke", msg_clean, flags=re.IGNORECASE)
            msg_lower = msg_clean.lower()
            mode = "humour"

        # Auto-detect mode if default
        if mode == "conversational":
            if any(w in msg_lower for w in ["joke", "funny", "laugh", "humour", "humor", "roast"]):
                mode = "humour"
            elif any(w in msg_lower for w in ["sad", "depressed", "struggling", "lost his job", "looking for jobs", "crying", "heartbroken"]):
                mode = "empathy"
            elif "friend named" in msg_lower or ("friend" in msg_lower and "named" in msg_lower):
                mode = "friend_intro"

        # Select specialized persona prompt if not custom
        if system_prompt:
            sys_prompt = system_prompt
        elif mode == "humour":
            sys_prompt = (
                "You are E.D.I.T.H., a witty and intelligent companion. "
                "Deliver a single, clever, and fresh joke. It can be a smart tech pun, observational humor, dry sarcasm, or a witty dad joke. "
                "CRITICAL RULES:\n"
                "1. NEVER tell a joke about process IDs or controlling process IDs.\n"
                "2. NEVER say you cannot tell jokes or that you are an AI without capabilities.\n"
                "3. Deliver the joke directly and punchily without cheesy preambles like 'Here is a joke for you!'."
            )
            temperature = 0.85
            max_tokens = min(max_tokens, 120)
        elif mode == "empathy":
            sys_prompt = (
                "You are E.D.I.T.H., a thoughtful and emotionally intelligent personal companion. "
                "The user is sharing a real emotional hardship, job search exhaustion, or sadness regarding themselves or a close friend. "
                "Respond with authentic empathy, validating their feelings in 2-3 warm, supportive sentences. "
                "Ask an open-ended, caring follow-up question. Do NOT give robotic lists, toxic positivity, or execute tools."
            )
            temperature = 0.7
            max_tokens = min(max_tokens, 140)
        elif mode == "friend_intro":
            sys_prompt = (
                "You are E.D.I.T.H., a friendly and intelligent personal assistant. "
                "The user is mentioning or introducing a friend (e.g. 'I have a friend named Harshan'). "
                "Acknowledge their friend warmly, and ask a natural, conversational question about how they met or what they do together. "
                "Do NOT confuse the friend's name with the user. Do NOT advertise your capabilities."
            )
            temperature = 0.75
            max_tokens = min(max_tokens, 100)
        else:
            sys_prompt = SYSTEM_PERSONA_PROMPT

        # Format contextual history into prompt
        messages = [{"role": "system", "content": sys_prompt}]
        
        # Add recent conversation turns if provided
        if context and context.get("recent_dialogue_turns"):
            recent = context["recent_dialogue_turns"][-4:]  # Last 4 turns
            for turn in recent:
                u = turn.get("user") or turn.get("query")
                b = turn.get("bot") or turn.get("response") or turn.get("message")
                if u:
                    messages.append({"role": "user", "content": str(u)})
                if b:
                    messages.append({"role": "assistant", "content": str(b)})

        # Append current user message
        messages.append({"role": "user", "content": msg_clean})

        # Attempt neural generation if model is loaded or file exists
        if not self.is_loaded and os.path.exists(self.model_path):
            self._initialize_model()

        if self.is_loaded and self.llm is not None:
            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.15
                )
                raw_text = response["choices"][0]["message"]["content"].strip()
                
                # Sanitize response: strip accidental markdown preamble, user labels, or reasoning tags
                clean_text = self._sanitize_response(raw_text)
                
                latency_ms = (time.time() - t_start) * 1000.0
                gen_tokens = response.get("usage", {}).get("completion_tokens", len(clean_text.split()))
                
                self.metrics["total_inferences"] += 1
                self.metrics["total_tokens_generated"] += gen_tokens
                self.metrics["last_latency_ms"] = round(latency_ms, 2)
                if latency_ms > 0:
                    tok_per_sec = (gen_tokens / (latency_ms / 1000.0))
                    self.metrics["average_tokens_per_second"] = round(tok_per_sec, 2)

                return {
                    "text": clean_text,
                    "backend": self.active_backend,
                    "tokens_generated": gen_tokens,
                    "latency_ms": round(latency_ms, 2),
                    "is_fallback": False
                }
            except Exception as ex:
                sys.stderr.write(f"[EDITH Generative Error] {ex}. Triggering dynamic fallback.\n")

        # Dynamic heuristic fallback with high conversational intelligence
        clean_fallback = self._dynamic_conversational_fallback(user_message, context=context, mode=mode)
        latency_ms = (time.time() - t_start) * 1000.0
        return {
            "text": clean_fallback,
            "backend": "dynamic_conversational_fallback",
            "tokens_generated": len(clean_fallback.split()),
            "latency_ms": round(latency_ms, 2),
            "is_fallback": True
        }

    def generate_humour(
        self,
        style: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate or select a fresh joke without repeating the predictable process ID roast."""
        prompt = "Tell me a genuinely funny, witty joke."
        if style:
            prompt = f"Tell me a {style} joke."
        res = self.generate(
            user_message=prompt,
            context=context,
            system_prompt=(
                "You are E.D.I.T.H., a witty and intelligent companion. "
                "Deliver a single, clever, and fresh joke. It can be observational, dry, tech-related, or a witty pun. "
                "CRITICAL: Do NOT tell a joke about process IDs, killing processes, or robotic AI tropes. "
                "Do not introduce with cheesy phrases like 'Here is a joke for you!'. Just deliver the joke naturally."
            ),
            temperature=0.85,
            max_tokens=120,
            mode="humour"
        )
        return res["text"]

    def generate_empathy(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate compassionate, human-level empathy for struggles or sad news."""
        res = self.generate(
            user_message=user_message,
            context=context,
            system_prompt=(
                "You are E.D.I.T.H., a thoughtful and emotionally intelligent assistant. "
                "The user is sharing an emotionally difficult situation, grief, or personal/friend struggle. "
                "Respond with authentic empathy, warmth, and non-judgmental presence. "
                "Acknowledge the emotional weight of what they shared. "
                "Ask a caring, open-ended question to help them talk through it or invite them to share more. "
                "Do NOT give generic toxic positivity or trite platitudes. Keep it genuine and conversational."
            ),
            temperature=0.7,
            max_tokens=150,
            mode="empathy"
        )
        return res["text"]

    def _sanitize_response(self, text: str) -> str:
        """Clean raw generated text from internal thought tokens or repetitive user names."""
        # Strip <think>...</think> tags if any
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        # Strip assistant / system prefixes
        text = re.sub(r"^(?:assistant|e\.d\.i\.t\.h\.?|bot)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        # Remove repeated exclamation emojis
        text = re.sub(r"([!?.])\s*([!?.])+", r"\1", text)
        return text

    def _dynamic_conversational_fallback(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "conversational"
    ) -> str:
        """
        High-quality, varied dynamic fallback when neural LLM is initializing or offline.
        Handles empathy, friend introductions, and diverse humor cleanly.
        """
        msg_low = user_message.lower().strip()

        # 1. Job search / Career struggles / Friend looking for jobs
        if any(w in msg_low for w in ["job", "looking for jobs", "interviews", "unemployed", "career"]) and any(
            w in msg_low for w in ["friend", "he", "she", "struggling", "sad", "trying", "haven't got", "hasn't got", "rejection"]
        ):
            variations = [
                "That's genuinely exhausting to go through. Three years of searching and interviewing without a breakthrough takes a massive emotional toll. Has he been looking in a specific field, or is he casting a wide net?",
                "That breaks my heart to hear. The job market can be completely brutal and dehumanizing after that long. How is he holding up day-to-day? Does he have a solid support circle around him?",
                "Three years is a long time to keep your hopes up and push through rejections. It makes complete sense that he's feeling down. What kind of roles is he aiming for? Maybe there's a fresh angle or network connection we could explore to help him.",
                "It's really hard watching someone you care about go through that kind of grind. Dealing with prolonged job searches wears down anyone's confidence. Has he been able to take any mental breaks, or has it been non-stop pressure?"
            ]
            return random.choice(variations)

        # 2. Friend is sad / Emotional sadness
        if any(w in msg_low for w in ["friend is sad", "friend is very sad", "he is sad", "she is sad", "they are sad", "he's sad", "she's sad"]):
            variations = [
                "I'm sorry to hear that. It's tough seeing a close friend feeling down. Do you know what's been weighing on him the most lately?",
                "That's hard to watch. Sometimes just having a friend who checks in and listens makes all the difference. Has he opened up to you about what's going on?",
                "I hear you. Feeling down or carrying heavy stuff can feel isolating. How are you thinking of supporting him right now?",
                "I'm really sorry he's going through a low moment. Just being there and letting him know he's not alone can be huge. Do you want to talk through how to be there for him?"
            ]
            return random.choice(variations)

        # 3. Introducing a friend
        m_friend = re.search(r"\b(?:friend\s+named|friend\s+is|named|call\s+him|call\s+her)\s+([A-Za-z]+)\b", msg_low)
        if "friend named" in msg_low or ("friend" in msg_low and "named" in msg_low) or m_friend:
            name = m_friend.group(1).capitalize() if m_friend else "your friend"
            variations = [
                f"{name} sounds like someone important to you! How long have you two known each other?",
                f"Nice to meet {name} through you! Did you two meet through work, college, or something else?",
                f"Always good to hear about the people in your corner. What does {name} do? What do you two usually get up to?",
                f"Got it! I'll keep {name} in mind. What's {name} like?"
            ]
            return random.choice(variations)

        # 4. Joke requests (Varied, intelligent humor; NO process ID jokes!)
        if any(w in msg_low for w in ["joke", "help me a joke", "tell me a joke", "crack a joke", "something funny"]):
            varied_jokes = [
                "There are 10 types of people in the world: those who understand binary, and those who don't.",
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "A SQL query walks into a bar, walks up to two tables and asks, 'Can I join you?'",
                "Why was the JavaScript developer sad? Because they didn't know how to 'null' their feelings.",
                "I asked my computer for a joke and it gave me my own source code from 2021. Truly ruthless.",
                "Why do Java programmers wear glasses? Because they can't C#.",
                "The problem with troubleshooting is that trouble shoots back.",
                "Parallel lines have so much in common. It’s a shame they’ll never meet."
            ]
            return random.choice(varied_jokes)

        # 5. General conversational fallback
        generic_conversational = [
            "I'm listening! Tell me more about what's on your mind.",
            "That's interesting. What made you think of that today?",
            "I hear you. Where would you like to take this from here?",
            "I'm right here with you. What are your thoughts on that?"
        ]
        return random.choice(generic_conversational)


# Singleton instance
_GLOBAL_GENERATIVE_REASONER: Optional[EdithGenerativeReasoner] = None

def get_generative_reasoner() -> EdithGenerativeReasoner:
    global _GLOBAL_GENERATIVE_REASONER
    if _GLOBAL_GENERATIVE_REASONER is None:
        _GLOBAL_GENERATIVE_REASONER = EdithGenerativeReasoner()
    return _GLOBAL_GENERATIVE_REASONER
