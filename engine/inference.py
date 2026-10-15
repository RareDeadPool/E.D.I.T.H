"""Inference Engine for E.D.I.T.H. Neural Network."""

import os
import sys
import re
import json
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from edith_ann.tokenizer import EdithTokenizer
from edith_ann.model import EdithNeuralNetwork

DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_best.pt")
DEFAULT_TOKENIZER = os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer.json")


class EdithInferenceEngine:
    """High-speed neural inference engine for E.D.I.T.H."""

    def __init__(self, checkpoint_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        tokenizer_path = tokenizer_path or DEFAULT_TOKENIZER
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first via train.py")

        # Load Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.maps = checkpoint["metadata_maps"]

        self.intents = {v: k for k, v in self.maps["intents"].items()}
        self.slots = {v: k for k, v in self.maps["slots"].items()}
        self.actions = {v: k for k, v in self.maps["actions"].items()}

        # Load Tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
        self.tokenizer = EdithTokenizer.load(tokenizer_path)

        # Initialize Neural Network Architecture
        ckpt_vocab_size = checkpoint.get("vocab_size", self.tokenizer.vocab_size)
        if ckpt_vocab_size != self.tokenizer.vocab_size:
            self.tokenizer.vocab = {k: v for k, v in self.tokenizer.vocab.items() if v < ckpt_vocab_size}
            self.tokenizer.inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
            self.tokenizer.vocab_size = len(self.tokenizer.vocab)

        self.model = EdithNeuralNetwork(
            vocab_size=self.tokenizer.vocab_size,
            num_intents=len(self.intents),
            num_slots=len(self.slots),
            num_actions=len(self.actions),
            d_model=self.config["model"]["d_model"],
            n_heads=self.config["model"]["n_heads"],
            n_layers=self.config["model"]["n_layers"],
            d_ff=self.config["model"]["d_ff"],
            max_seq_len=self.config["model"]["max_seq_len"],
            dropout=0.0,  # disable dropout in inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()

        # 6. Optional ONNX Runtime Engine Acceleration (<2 ms per query)
        self.ort_session = None
        base, _ = os.path.splitext(checkpoint_path)
        onnx_path = base + ".onnx"
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            except Exception:
                self.ort_session = None

    def predict(self, command_text: str) -> Dict[str, Any]:
        """
        Run forward pass on raw user command and extract structured task directive.
        Evaluates intent classification, BIO slot extraction, action policy,
        complexity categorization, and calibrated operational confidence.
        """
        cmd_clean = command_text.strip()
        cmd_lower = cmd_clean.lower()

        max_seq_len = self.config["model"]["max_seq_len"]
        input_ids, attention_mask = self.tokenizer.encode(
            command_text, max_len=max_seq_len, add_special_tokens=True, pad_to_max=True
        )

        if self.ort_session is not None:
            import numpy as np
            ort_in = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "attention_mask": np.array([attention_mask], dtype=np.int64),
            }
            ort_outs = self.ort_session.run(None, ort_in)
            outputs = {
                "intent_logits": torch.from_numpy(ort_outs[0]),
                "slot_logits": torch.from_numpy(ort_outs[1]),
                "action_logits": torch.from_numpy(ort_outs[2]),
                "confidence": torch.from_numpy(ort_outs[3]),
                "complexity_logits": torch.from_numpy(ort_outs[4]),
            }
        else:
            t_input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            t_attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(t_input_ids, attention_mask=t_attention_mask)

        # 1. Intent Prediction & Top Candidates
        intent_probs = F.softmax(outputs["intent_logits"], dim=-1)[0]
        intent_id = int(torch.argmax(intent_probs).item())
        intent_name = self.intents.get(intent_id, "unknown")
        intent_confidence = float(intent_probs[intent_id].item())

        top_k = min(3, len(intent_probs))
        top_indices = torch.topk(intent_probs, k=top_k).indices.tolist()
        top_intents = [
            {
                "intent": self.intents.get(idx, "unknown"),
                "confidence": round(float(intent_probs[idx].item()), 4)
            }
            for idx in top_indices
        ]

        # 2. Action Policy Prediction
        action_probs = F.softmax(outputs["action_logits"], dim=-1)[0]
        action_id = int(torch.argmax(action_probs).item())
        action_name = self.actions.get(action_id, "noop")
        action_confidence = float(action_probs[action_id].item())

        # 3. Overall Operational Confidence
        operational_confidence = float(outputs["confidence"][0][0].item())

        # 4. Complexity Prediction (Single-step vs Multi-step)
        is_multi_step = False
        complexity_conf = 0.0
        if "complexity_logits" in outputs:
            comp_probs = F.softmax(outputs["complexity_logits"], dim=-1)[0]
            is_multi_step = bool(torch.argmax(comp_probs).item() == 1)
            complexity_conf = float(comp_probs[1].item())

        # 5. Slot Extraction & Entity Parsing
        slot_preds = torch.argmax(outputs["slot_logits"], dim=-1)[0].tolist()
        raw_tokens = self.tokenizer.tokenize_raw(command_text)

        # Map slots to raw tokens (skipping BOS at index 0)
        entities: Dict[str, List[str]] = {}
        current_entity_type = None
        current_entity_tokens = []

        for i, token in enumerate(raw_tokens):
            pos = i + 1  # offset for BOS
            if pos >= len(slot_preds):
                break
            tag = self.slots.get(slot_preds[pos], "O")

            if tag.startswith("B-"):
                if current_entity_type and current_entity_tokens:
                    entities.setdefault(current_entity_type, []).append(" ".join(current_entity_tokens))
                current_entity_type = tag[2:]
                current_entity_tokens = [token]
            elif tag.startswith("I-") and current_entity_type == tag[2:]:
                current_entity_tokens.append(token)
            else:
                if current_entity_type and current_entity_tokens:
                    entities.setdefault(current_entity_type, []).append(" ".join(current_entity_tokens))
                    current_entity_type = None
                    current_entity_tokens = []

        if current_entity_type and current_entity_tokens:
            entities.setdefault(current_entity_type, []).append(" ".join(current_entity_tokens))

        # Flatten entities: join multi-token lists into clean strings
        parsed_entities = {}
        for k, v in entities.items():
            if isinstance(v, list):
                parsed_entities[k] = " ".join(str(item) for item in v).strip()
            else:
                parsed_entities[k] = str(v).strip()

        if intent_name == "music_playback" or action_name == "exec_play_music":
            parsed_entities = self._refine_music_slots(command_text, parsed_entities)

        # Check for pure non-alphanumeric noise or gibberish
        has_alpha = any(c.isalnum() for c in cmd_clean)
        if not has_alpha or operational_confidence < 0.20:
            intent_name = "unsupported_ood"
            action_name = "request_clarification"
            operational_confidence = 0.0 if not has_alpha else operational_confidence
            status = "out_of_distribution"
        elif intent_name == "unsupported_ood" or operational_confidence < 0.35:
            status = "out_of_distribution" if intent_name == "unsupported_ood" else "low_confidence"
        else:
            status = "success"

        return {
            "query": command_text,
            "intent": intent_name,
            "intent_confidence": round(intent_confidence, 4),
            "top_intents": top_intents,
            "action": action_name,
            "action_confidence": round(action_confidence, 4),
            "operational_confidence": round(operational_confidence, 4),
            "is_multi_step": is_multi_step,
            "complexity_confidence": round(complexity_conf, 4),
            "entities": parsed_entities,
            "status": status,
        }

    @staticmethod
    def _refine_music_slots(command: str, raw_entities: Dict[str, str]) -> Dict[str, str]:
        """Refine music entity boundaries (SONG, MODIFIER, ARTIST, PLATFORM) from raw command."""
        cmd = command.strip()
        cmd_lower = cmd.lower()

        # Clean leading trigger phrases
        for trig in [
            'open spotify and play', 'open youtube and play', 'play on spotify and',
            'edith play', 'play song', 'stream', 'put on', 'can you play',
            'i want to hear', 'i want to listen to', 'play me', 'queue up',
            'search and play', 'search for and play', 'search for', 'play'
        ]:
            if cmd_lower.startswith(trig + ' '):
                body = cmd[len(trig)+1:].strip()
                break
        else:
            body = cmd

        # Clean trailing context/activity clauses (e.g. 'while I work', 'while studying', 'for me')
        body = re.sub(
            r"\s+\b(?:while\s+(?:i\s+)?(?:work|study|sleep|relax|read|code)|in\s+the\s+background|for\s+me)\b.*$",
            "",
            body,
            flags=re.IGNORECASE
        ).strip()

        # Platform detection
        platform = raw_entities.get('PLATFORM', '')
        for p in ['on spotify', 'on youtube', 'on apple music', 'on soundcloud']:
            m = re.search(rf'\s+{re.escape(p)}\b', body, re.IGNORECASE)
            if m:
                platform = p.split()[-1]
                body = (body[:m.start()] + body[m.end():]).strip()
                break

        # Artist detection via 'by <artist>'
        artist = raw_entities.get('ARTIST', '')
        m_by = re.search(r'\s+by\s+([A-Za-z0-9\s\.\'\-]+)$', body, re.IGNORECASE)
        if m_by:
            artist = m_by.group(1).strip()
            body = body[:m_by.start()].strip()

        known_modifiers = [
            'slowed and reverb', 'bass boosted and reverb', 'lofi and chill', 'acoustic and live',
            'live version', 'acoustic version', 'instrumental version', 'karaoke version',
            'extended version', 'deluxe edition', 'anniversary edition', 'remastered version',
            'phonk remix', 'slowed remix', 'sped up remix', 'lofi remix', 'official remix',
            'drill remix', 'trap remix', 'remix', 'acoustic', 'unplugged', 'stripped',
            'live concert', 'live performance', 'live', 'instrumental', 'no vocals',
            'slowed', 'sped up', 'nightcore', 'remastered', 'clean version', 'explicit',
            'relaxing', 'calming', 'chill', 'focus', 'ambient', 'peaceful'
        ]

        # Pattern: 'the <modifier> version of <song>' or '<modifier> of <song>'
        m_inv = re.match(r'^(?:the\s+)?([A-Za-z0-9\s]+?)\s+version\s+of\s+(.+)$', body, re.IGNORECASE)
        if not m_inv:
            m_inv = re.match(r'^(?:the\s+)?(acoustic|instrumental|live|remix|clean|explicit|orchestral|piano|guitar|unplugged)\s+of\s+(.+)$', body, re.IGNORECASE)

        modifier = raw_entities.get('MODIFIER', '')
        song = ""
        if m_inv:
            mod_candidate = m_inv.group(1).strip()
            if not mod_candidate.lower().endswith('version'):
                modifier = mod_candidate + ' version'
            else:
                modifier = mod_candidate
            song = m_inv.group(2).strip()
        else:
            # Check featuring
            m_feat = re.search(r'\b((?:feat\.|ft\.|featuring)\s+[^,]+)$', body, re.IGNORECASE)
            if m_feat:
                modifier = m_feat.group(1).strip()
                body = body[:m_feat.start()].strip()
                song = body
            else:
                song = body
                for km in sorted(known_modifiers, key=len, reverse=True):
                    m_mod = re.search(rf'\b({re.escape(km)})\b', body, re.IGNORECASE)
                    if m_mod:
                        modifier = m_mod.group(1).strip()
                        song = (body[:m_mod.start()] + body[m_mod.end():]).strip()
                        break

        # If modifier was captured in artist, extract it
        if artist:
            for km in sorted(known_modifiers, key=len, reverse=True):
                m_amod = re.search(rf'\b({re.escape(km)})\b', artist, re.IGNORECASE)
                if m_amod:
                    if not modifier:
                        modifier = m_amod.group(1).strip()
                    artist = (artist[:m_amod.start()] + artist[m_amod.end():]).strip()
                    break

        # Strip leftover 'version' from song
        song = re.sub(r'\s+version\b', '', song, flags=re.IGNORECASE).strip()

        if not song:
            song = raw_entities.get('SONG', '')
            if not song and body != cmd:
                song = body

        refined = dict(raw_entities)
        if song and song.strip():
            refined['SONG'] = song.strip()
        if modifier:
            refined['MODIFIER'] = modifier.strip()
        elif 'MODIFIER' in refined and not refined['MODIFIER']:
            refined.pop('MODIFIER', None)
        if artist:
            refined['ARTIST'] = artist.strip()
        if platform:
            refined['PLATFORM'] = platform.strip()
        return refined
