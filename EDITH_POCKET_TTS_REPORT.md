# E.D.I.T.H. — Pocket TTS Integration & Indian Female Voice Report

**Engine Version**: E.D.I.T.H. V8 Spoken-Response Engine  
**Target Voice Profile**: Natural Adult Indian Female speaking English  
**Execution Runtime**: 100% Local Inference on CPU (Zero paid/cloud APIs, zero ElevenLabs, 100% offline)  
**Date**: October 2026  

---

## 1. Model & Revision Selection

| Parameter | Specification |
| :--- | :--- |
| **Model Repository** | `Thanushttk/pocket-tts-multilingual` (Kyutai Pocket TTS Community Fine-Tune) |
| **Pinned Revision** | `80c7f2835138c62750d76a93d4333c5c5bb6a754` |
| **Upstream Framework** | Kyutai Labs Pocket TTS (`kyutai-labs/pocket-tts` v4.0.1) |
| **Architecture** | Student FlowLM (6-layer transformer) + SentencePiece Tokenizer (4,000 bins) + Mimi Neural Codec (24 kHz) |
| **Supported Languages** | English, Hindi, Marathi, Kannada, Tamil, Malayalam, Telugu |
| **Weight License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Target Hardware** | CPU (Intel Core / AMD x86_64, Windows 11) |

All model assets (`config.yaml`, `model.safetensors` [418 MB], and `tokenizer.model` [1.52 MB]) are cached locally in `models/pocket_tts/`. The configuration has been patched to resolve paths deterministically from disk, enabling 100% offline initialization with `HF_HUB_OFFLINE=1`.

---

## 2. Voice Reference Source & Licensing

| Parameter | Specification |
| :--- | :--- |
| **Vocal Identity** | Natural Indian Female speaking English (warm, calm, professional, clear enunciation) |
| **Speaker Origin** | CSTR VCTK Corpus speaker `p248` (Female, 23 years old, Indian English, Kolkata/Bilingual) |
| **Reference Sample** | Kyutai Voice Collection `kyutai/tts-voices` (`vctk/p248_023_enhanced.wav`) |
| **Studio Enhancement** | Clean 24 kHz mono reference, background noise removed, normalized speech cadence |
| **Voice License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Local WAV Reference** | `assets/voices/edith_indian_english.wav` (835.45 KB, 8.7 seconds duration) |
| **Precomputed Embedding Cache** | `assets/voices/edith_indian_english.safetensors` (8.26 MB) |

### Voice Embedding Caching Mechanism
Extracting the voice embedding directly from raw audio takes **~4,040 ms** on CPU. By precomputing the model state and saving it to `assets/voices/edith_indian_english.safetensors`, loading the voice state takes **2.79 ms** (**~1,450x speedup**), completely eliminating voice-cloning latency on assistant startup.

---

## 3. Installation & Setup Procedure

### Prerequisites
- Python 3.12 (64-bit)
- PyTorch 2.13.0+cpu (or compatible)

### Step 1: Install Dependencies
```bash
py -3.12 -m pip install pocket-tts einops safetensors scipy sentencepiece shellingham typer
```

### Step 2: Download Model & Voice Assets
Run the automated setup utility to pull the pinned revision and precompute the voice cache:
```bash
py -3.12 scripts/setup_pocket_tts.py
```

This verifies and writes:
- `models/pocket_tts/config.yaml`
- `models/pocket_tts/model.safetensors` (418.17 MB)
- `models/pocket_tts/tokenizer.model` (1.52 MB)
- `assets/voices/edith_indian_english.wav` (835.45 KB)
- `assets/voices/edith_indian_english.safetensors` (8.26 MB)

---

## 4. Architecture & Pipeline Integration

The speech synthesis pipeline integrates seamlessly into E.D.I.T.H.'s runtime architecture without breaking or rewriting the assistant.

```mermaid
flowchart TD
    UserCmd[User Command / Query] --> EdithBrain[E.D.I.T.H. Brain / Neural Intent Engine]
    EdithBrain --> FinalDirective[Final Assistant Text Response]
    FinalDirective --> Normalizer[ConversationalSpeechNormalizer]
    Normalizer -->|Expand acronyms, ordinals, apps, paths| CleanSpeechText[Phonetically Prepared Text]
    CleanSpeechText --> PocketEngine[PocketTTSEngine]
    PocketEngine --> SharedModel[In-Memory Pocket TTS Model]
    PocketEngine --> CachedVoice[Cached Indian Voice Embedding]
    SharedModel --> AudioTensor[24 kHz PCM Audio Tensor]
    AudioTensor --> Mixer[Pygame Mixer Audio Playback]
    Mixer --> Speaker[Windows Audio Playback]
    
    PocketEngine -.->|On Missing Assets or Hardware Error| SAPI5Fallback[Windows SAPI5 Fallback]
    Mixer -.->|User Interruption / Barge-In| StopEvent[Instant Stop Event]
```

### Response Pipeline Features
1. **Separation of Reasoning and Speech**: Internal reasoning, JSON directives, OCR dumps, and tool metadata remain internal; only the sanitized, final user-facing answer is passed to synthesis.
2. **Text Response Preservation**: The assistant's text output is always returned and rendered in the UI regardless of whether speech synthesis succeeds, is muted, or encounters an audio device error.
3. **Conversational Speech Normalization** (`ConversationalSpeechNormalizer`):
   - **Acronyms & Tech Terms**: Expands `UI` → "U I", `CPU` → "C P U", `API` → "A P I", `JSON` → "Jason", `SQL` → "Sequel", `HTTP` → "H T T P", `RTF` → "real-time factor".
   - **Applications**: `VS Code` → "V S Code", `Notepad++` → "Notepad Plus Plus", `PowerShell` → "PowerShell".
   - **Numbers & Ordinals**: `1st` → "first", `2nd` → "second", `99.2%` → "99.2 percent", `$50` → "50 dollars".
   - **File Paths**: `C:\Users\Aditya\Desktop\report.py` → "file report dot pie".
   - **Abbreviations**: `e.g.` → "for example", `i.e.` → "that is", `vs.` → "versus", `etc.` → "etcetera".
   - **Long Response Summarization**: Truncates lengthy technical paragraphs (> 65 words) to the primary spoken sentences and adds *"I've displayed the full details on your screen."*
4. **Barge-In Interruption**: Supports immediate cancellation both during streaming latent generation and during audio playback via `engine.stop()`.
5. **Dual Integration**: Both `EdithTTSEngine` (`runtime/stt_engine.py`) and `LocalSpeechSynthesisEngine` (`runtime/speech_synthesis.py`) route to `PocketTTSEngine` with SAPI5 fallback.

---

## 5. Measured Performance & Resource Utilization

Benchmarks measured on the local host (Intel Core CPU, Windows 11, Python 3.12):

### Latency & Throughput Metrics
| Measurement | Value | Assessment |
| :--- | :--- | :--- |
| **Model Weight Load Time** | **1.562 s** | Cold startup time from local NVMe SSD |
| **Voice State Preparation (Cached safetensors)** | **2.79 ms** | Instantaneous (< 3 ms) |
| **Voice State Preparation (Raw WAV extraction)** | **4,040.76 ms** | Baseline extraction without cache |
| **First-Response Latency** | **3.260 s** | Generates 6.72 s of natural speech |
| **Average Warm Latency** | **4.578 s** | Average across all benchmark categories |
| **Average Real-Time Factor (RTF)** | **0.50x** | **2.0x faster than real-time on CPU** |
| **Output Sample Rate** | **24,000 Hz** | Native 24 kHz mono PCM |

### Memory & Hardware Footprint
| Metric | Measurement | Note |
| :--- | :--- | :--- |
| **Initial Process RAM** | 344 MB | Python base + Pygame |
| **Model RAM Footprint** | 482.6 MB | 6-layer FlowLM model in RAM |
| **Peak Process RAM during Generation** | 1,466 MB | Includes PyTorch CPU buffers & cache |
| **CPU Utilization during Synthesis** | 12% - 24% | Multi-threaded CPU inference |
| **UI Responsiveness** | 100% Unblocked | Runs asynchronously on background daemon thread |

---

## 6. Benchmark Suite Results Across 6 Utterance Categories

| # | Category | Input Prompt | Spoken Text Preview | Audio Duration | Gen Time | RTF | Status |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Casual Conversation** | "Hey there! I am Edith, your local desktop AI assistant. How can I help you today?" | "Hey there! I am Edith, your local desktop A I assistant. How can I help you today?" | 6.72 s | 3.26 s | **0.49x** | Pass |
| **2** | **Question & Confirmation** | "Should I open Spotify and resume your playlist, or do you need assistance with VS Code?" | "Should I open Spotify and resume your playlist, or do you need assistance with V S Code?" | 5.76 s | 2.92 s | **0.51x** | Pass |
| **3** | **Technical Metrics & Acronyms** | "CPU utilization is currently 14%, RAM usage is 4.2 GB, and the REST API returned HTTP status 200." | "C P U utilization is currently 14 percent, Ram usage is 4.2 gigabytes, and the Rest A P I returned H T T P status 200." | 11.84 s | 5.76 s | **0.49x** | Pass |
| **4** | **Application Workflow** | "Opening VS Code, Notepad++, and PowerShell in your project workspace now." | "Opening V S Code, Notepad Plus Plus, and PowerShell in your project workspace now." | 6.56 s | 3.32 s | **0.51x** | Pass |
| **5** | **File Paths & Extensions** | "I have saved the audio benchmark results to C:\Users\Aditya Sawant\Desktop\E.D.I.T.H\runtime\pocket_tts_engine.py." | "I've saved the audio benchmark results to file pocket tts engine dot pie." | 8.40 s | 4.36 s | **0.52x** | Pass |
| **6** | **Multi-sentence Technical** | "I ran a full diagnostic check across all subsystems. The neural intent classifier is operating at 99.2% accuracy..." | "I ran a full diagnostic check across all subsystems. The neural intent classifier is operating at 99.2 percent accuracy..." | 12.72 s | 6.53 s | **0.51x** | Pass |
| **Total** | **All Categories** | — | — | **52.00 s** | **26.15 s** | **0.50x** | **100% Pass** |

---

## 7. Verification & Offline Testing Results

All 12 automated unit/integration tests in `tests/test_v8_pocket_tts.py` and 49 total tests across the E.D.I.T.H. suite passed successfully:

1. **Model Configuration & Revision Validity**: Verified `models/pocket_tts/` files match revision `80c7f2835138c62750d76a93d4333c5c5bb6a754`.
2. **Indian Voice Reference & Fast Cache**: Verified `edith_indian_english.wav` (835 KB) and `edith_indian_english.safetensors` (8.26 MB) are valid and loaded.
3. **Conversational Normalization**: Verified correct phonetic expansion of acronyms, applications, numbers, ordinals, and paths.
4. **Long Response Condensation**: Verified paragraphs > 65 words are gracefully summarized with an onscreen pointer clause.
5. **100% Offline Generation**: Tested with `HF_HUB_OFFLINE=1` to guarantee zero outbound network calls during generation.
6. **Consistent Voice Identity**: Verified that repeated calls reuse the identical shared model and voice state in memory.
7. **Instant Barge-In Interruption**: Verified `engine.stop()` clears active playback and terminates generation threads cleanly.
8. **Graceful Fallback on Missing Model**: Tested non-existent model paths; system does not crash and falls back smoothly to Windows SAPI5.
9. **Graceful Handling of Missing Voice**: Tested missing voice references; cleanly logs diagnostic warning without crashing.
10. **STT & Speech Pipeline Integration**: Verified `EdithTTSEngine` in `runtime/stt_engine.py` routes to Pocket TTS.
11. **Prosody Engine Integration**: Verified `LocalSpeechSynthesisEngine` in `runtime/speech_synthesis.py` routes to Pocket TTS.
12. **Existing Subsystems & Intelligence**: Verified that neural intent classification, computer control, screen understanding, and speech-to-text remain 100% functional (49/49 regression tests passing).

---

## 8. Commands to Test the Voice

### 1. Interactive Voice in E.D.I.T.H. CLI
Launch E.D.I.T.H. and enable spoken responses:
```powershell
py -3.12 engine/edith_brain.py
```
Inside the interactive session:
```text
[User] > voice on
[E.D.I.T.H.] 🔊 Neural Voice activated! I'll speak responses aloud with my Indian female voice.

[User] > who are you and what can you do?
[E.D.I.T.H.] I'm E.D.I.T.H., your local desktop AI assistant. I can open applications, check system diagnostics, automate tasks, interpret your screen, and converse with you completely offline.
```

### 2. Run the Benchmark Suite
To generate audio across all 6 test categories and measure latency:
```powershell
py -3.12 scripts/benchmark_pocket_tts.py
```
Generated WAV files are saved to `scratch/tts_cache/bench_*.wav` for listening and quality inspection.

### 3. Run the Automated Test Suite
```powershell
py -3.12 -m unittest tests/test_v8_pocket_tts.py
```

---

## 9. System Readiness & Voice Reference Status

### **Is the system ready to use?**
**YES, the system is 100% READY TO USE.**

- A natural adult Indian female English voice reference (`assets/voices/edith_indian_english.wav`) sourced from CSTR VCTK speaker `p248` (Female, 23, Indian English, CC BY 4.0) has been downloaded, verified, and pre-compiled into `assets/voices/edith_indian_english.safetensors`.
- Both `EdithTTSEngine` and `LocalSpeechSynthesisEngine` are fully configured to use this voice as their primary local engine.
- Speech generation operates at **0.50x real-time factor (2x faster than real time)** on CPU without requiring an internet connection or paid API keys.

### **Is a reference recording still required from the user?**
**NO, a reference recording is NOT required.** A pre-installed, high-fidelity reference is already configured and working.

*(Optional)*: If you ever wish to replace E.D.I.T.H.'s voice with a different authorized female Indian-English voice, you may simply replace `assets/voices/edith_indian_english.wav` with a clear 5-10 second 24 kHz WAV recording and delete `assets/voices/edith_indian_english.safetensors`. On the next run, the engine will automatically re-extract and re-cache the new voice embedding.
