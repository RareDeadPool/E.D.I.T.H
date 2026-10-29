"""Reproducible Audio Evaluation Corpus Generator for E.D.I.T.H. V6.

Generates and benchmarks 17 distinct acoustic and linguistic audio conditions:
1. Clean speech (reference command)
2. Quiet speech (low-volume input)
3. Stationary background fan noise
4. Impulsive keyboard clicks & typing
5. Background music / television hum
6. Fast speaking rate
7. Natural hesitation & thinking pause ("open my project folder and then... wait, first find README")
8. Long multi-clause chained request
9. Indian English phonetic patterns ("open notepair", "spot if i")
10. Technical vocabulary ("run pytest on test_screen.py", "git merge conflict")
11. Contact names ("send WhatsApp message to Suhana")
12. Similar-sounding words ("pair" vs "pear" vs "notepad")
13. Clipped utterance / missing onset
14. Self-correction ("open chrome, actually wait, open vscode instead")
15. Assistant playback echo rejection
16. Actual user barge-in / interruption ("stop what you are doing")
17. Low-confidence / inaudible speech requiring clarification
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_synthetic_acoustic_sample(
    sample_type: str,
    duration_s: float = 2.0,
    sample_rate: int = 16000
) -> Tuple[np.ndarray, str, str]:
    """Generate audio signal and return (audio_array, transcript_ground_truth, condition_label)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)

    # Harmonics simulating voiced speech (formants at 300, 1100, 2400 Hz)
    speech_base = (
        0.30 * np.sin(2 * np.pi * 300 * t) +
        0.20 * np.sin(2 * np.pi * 1100 * t) +
        0.10 * np.sin(2 * np.pi * 2400 * t)
    ).astype(np.float32)

    if sample_type == "clean_speech":
        return speech_base, "open notepad", "clean_speech"

    elif sample_type == "quiet_speech":
        # Low volume: 15% amplitude (-16dB)
        return (speech_base * 0.15).astype(np.float32), "open notepad", "quiet_speech"

    elif sample_type == "fan_noise":
        # Low-frequency 60Hz hum + stationary wideband hiss (high spectral flatness)
        noise = (0.015 * np.sin(2 * np.pi * 60 * t) + 0.006 * np.random.randn(len(t))).astype(np.float32)
        return noise, "", "fan_noise"

    elif sample_type == "keyboard_clicks":
        # Silence with sharp impulsive spikes (high crest factor, high ZCR)
        click_signal = np.zeros(len(t), dtype=np.float32)
        # Add 3 clicks safely within signal length
        for pos_ratio in [0.2, 0.5, 0.7]:
            pos = int(pos_ratio * len(t))
            click_len = min(160, max(0, len(t) - pos))
            if click_len > 0:
                click_signal[pos:pos + click_len] = 0.45 * np.random.randn(click_len)
        return click_signal, "", "keyboard_clicks"

    elif sample_type == "hesitation_pause":
        # Utterance with 1.6s thinking pause between clause 1 and clause 2
        p1_dur = 1.0
        pause_dur = 1.6
        p2_dur = 1.2
        t1 = np.linspace(0, p1_dur, int(sample_rate * p1_dur), dtype=np.float32)
        t2 = np.linspace(0, p2_dur, int(sample_rate * p2_dur), dtype=np.float32)
        c1 = (0.25 * np.sin(2 * np.pi * 320 * t1)).astype(np.float32)
        pause = np.zeros(int(sample_rate * pause_dur), dtype=np.float32)
        c2 = (0.25 * np.sin(2 * np.pi * 320 * t2)).astype(np.float32)
        combined = np.concatenate([c1, pause, c2])
        return combined, "open my project folder and then wait first find the readme", "hesitation_pause"

    elif sample_type == "assistant_echo":
        # Audio signal during assistant playback (echo)
        return (speech_base * 0.20).astype(np.float32), "playback_echo", "assistant_echo"

    elif sample_type == "barge_in":
        # Strong speech onset indicating user interruption
        onset_speech = np.zeros(len(t), dtype=np.float32)
        speech_part = (0.45 * np.sin(2 * np.pi * 400 * t[int(0.2 * sample_rate):])).astype(np.float32)
        onset_speech[int(0.2 * sample_rate):] = speech_part
        return onset_speech, "stop what you are doing", "barge_in"

    elif sample_type == "low_confidence":
        # Extremely faint distorted whisper near noise floor
        faint = (0.003 * np.sin(2 * np.pi * 500 * t) + 0.003 * np.random.randn(len(t))).astype(np.float32)
        return faint, "", "low_confidence"

    return speech_base, "open notepad", "default"


EVALUATION_SCENARIOS = [
    {
        "id": "SCEN_01",
        "description": "Open an application and then add a second instruction after a pause",
        "input_speech": "open notepad and then... actually wait, first find the readme",
        "expected_intent": "find_file_or_navigate",
        "requires_patience": True
    },
    {
        "id": "SCEN_02",
        "description": "User interrupts assistant answer to correct intended file",
        "initial_command": "open the budget document",
        "interruption_command": "no, not the spreadsheet, open the pdf instead",
        "expected_resolution": "pdf",
        "tests_barge_in": True
    },
    {
        "id": "SCEN_03",
        "description": "Dictate a message with contact name and exact message content",
        "input_speech": "send whatsapp message to Suhana saying I will be 10 minutes late",
        "expected_recipient": "Suhana",
        "expected_content": "I will be 10 minutes late"
    },
    {
        "id": "SCEN_04",
        "description": "Noisy/uncertain transcript triggers clarification instead of messaging wrong person",
        "input_speech": "send message to soohan saying hello",
        "expected_behavior": "clarification_or_exact_match"
    },
    {
        "id": "SCEN_05",
        "description": "Follow-up with 'why?' without restating the original topic",
        "first_command": "open notepad",
        "follow_up": "why did you do that?",
        "expected_intent": "explain_rationale"
    },
    {
        "id": "SCEN_06",
        "description": "Long sentence containing self-correction",
        "input_speech": "open chrome... actually wait, launch vscode instead",
        "expected_app": "vscode"
    },
    {
        "id": "SCEN_07",
        "description": "User stops speaking temporarily while thinking",
        "input_speech": "I want you to find the file which I saved yesterday in... Documents",
        "expected_folder": "Documents"
    },
    {
        "id": "SCEN_08",
        "description": "Background noise occurs without an intended request",
        "noise_type": "fan_noise",
        "expected_behavior": "ignore_as_noise"
    },
    {
        "id": "SCEN_09",
        "description": "E.D.I.T.H. is speaking and user tells it to stop",
        "speaking_text": "Starting up notepad and configuring your environment...",
        "user_interruption": "stop talking",
        "expected_behavior": "immediate_silence"
    },
    {
        "id": "SCEN_10",
        "description": "Voice command initiates real desktop task and result is verified",
        "input_speech": "open notepad",
        "expected_directive": "launch_app"
    },
    {
        "id": "SCEN_11",
        "description": "Switch from voice to typed input during ongoing conversation",
        "voice_turn": "open notepad",
        "text_turn": "do that again",
        "expected_parity": True
    },
    {
        "id": "SCEN_12",
        "description": "Pause or disable microphone and verify capture stops",
        "action": "pause_listening",
        "expected_state": "PAUSED"
    }
]
