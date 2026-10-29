"""Robust Local Audio Front End for E.D.I.T.H. V6.

Provides:
1. Stable microphone capture at 16kHz float32 mono with device health checks.
2. Adaptive background noise floor estimation and multi-band spectral noise suppression.
3. Transient spike filtering (keyboard clicks, mouse clicks).
4. Dynamic Automatic Gain Control (AGC) for quiet speech.
5. Multi-feature Voice Activity Detection (VAD): RMS, spectral centroid, zero-crossing rate, speech-band ratio.
6. Pre-roll ring buffer (0.5s) ensuring initial speech onset is preserved.
7. Acoustic echo suppression and playback awareness to prevent self-transcription loops.
"""

import os
import sys
import time
import math
import io
import wave
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioSignalMetrics:
    """Real-time signal analysis and health diagnostics."""

    @staticmethod
    def compute_rms(signal: np.ndarray) -> float:
        """Compute root-mean-square energy of 1D float32 audio."""
        if len(signal) == 0:
            return 0.0
        return float(np.sqrt(np.mean(signal ** 2)))

    @staticmethod
    def compute_zero_crossing_rate(signal: np.ndarray) -> float:
        """Compute normalized zero-crossing rate."""
        if len(signal) < 2:
            return 0.0
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2.0
        return float(zero_crossings / len(signal))

    @staticmethod
    def compute_spectral_features(signal: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Compute spectral centroid, voice-band energy ratio (300Hz-3400Hz), and spectral flatness."""
        if len(signal) < 32:
            return {"centroid": 0.0, "voice_band_ratio": 0.0, "spectral_flatness": 1.0}

        # Windowed FFT
        window = np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(signal * window))
        power = (spectrum ** 2) + 1e-12
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)

        spec_sum = np.sum(spectrum)
        if spec_sum < 1e-7:
            return {"centroid": 0.0, "voice_band_ratio": 0.0, "spectral_flatness": 1.0}

        # Spectral Centroid (center of mass of spectrum)
        centroid = float(np.sum(freqs * spectrum) / spec_sum)

        # Voice-band energy ratio (300 Hz - 3400 Hz)
        voice_mask = (freqs >= 300) & (freqs <= 3400)
        voice_energy = np.sum(spectrum[voice_mask])
        voice_band_ratio = float(voice_energy / spec_sum)

        # Spectral Flatness (Wiener entropy: geometric mean / arithmetic mean)
        # White noise / fan noise has high flatness (>0.05); speech has discrete harmonic peaks (<0.02)
        geo_mean = float(np.exp(np.mean(np.log(power))))
        ari_mean = float(np.mean(power))
        spectral_flatness = float(geo_mean / max(ari_mean, 1e-12))

        return {
            "centroid": centroid,
            "voice_band_ratio": voice_band_ratio,
            "spectral_flatness": spectral_flatness
        }

    @staticmethod
    def check_signal_health(signal: np.ndarray) -> Dict[str, Any]:
        """Diagnose audio health: clipping, under-amplification, DC offset."""
        if len(signal) == 0:
            return {"healthy": False, "clipping": False, "under_amplified": True, "dc_offset": 0.0}

        peak = float(np.max(np.abs(signal)))
        dc_offset = float(np.mean(signal))
        rms = AudioSignalMetrics.compute_rms(signal)

        clipping = peak >= 0.99
        under_amplified = peak < 0.005 and rms < 0.001

        return {
            "healthy": not clipping and not under_amplified,
            "clipping": clipping,
            "under_amplified": under_amplified,
            "peak": round(peak, 4),
            "rms": round(rms, 6),
            "dc_offset": round(dc_offset, 6)
        }


class SpectralNoiseSuppressor:
    """Lightweight local spectral noise suppressor and transient filter.
    
    Uses adaptive noise profile estimation and spectral subtraction with
    over-subtraction and spectral floor to eliminate fan noise, AC hum, and keyboard spikes
    without distorting speech.
    """

    def __init__(self, sample_rate: int = 16000, alpha: float = 1.5, beta: float = 0.05):
        self.sample_rate = sample_rate
        self.alpha = alpha  # Over-subtraction factor
        self.beta = beta    # Spectral floor factor
        self.noise_profile: Optional[np.ndarray] = None
        self.noise_update_count = 0
        self.lock = threading.Lock()

    def update_noise_profile(self, noise_chunk: np.ndarray):
        """Update stationary background noise spectrum from a known non-speech chunk."""
        if len(noise_chunk) < 64:
            return
        window = np.hanning(len(noise_chunk))
        mag_spec = np.abs(np.fft.rfft(noise_chunk * window))

        with self.lock:
            if self.noise_profile is None or len(self.noise_profile) != len(mag_spec):
                self.noise_profile = mag_spec
                self.noise_update_count = 1
            else:
                # Recursive exponential moving average
                decay = 0.90
                self.noise_profile = decay * self.noise_profile + (1.0 - decay) * mag_spec
                self.noise_update_count += 1

    def suppress(self, signal: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction noise gate to 1D float32 audio."""
        if self.noise_profile is None or len(signal) < 64:
            return signal

        n = len(signal)
        window = np.hanning(n)
        fft_complex = np.fft.rfft(signal * window)
        mag_spec = np.abs(fft_complex)
        phase_spec = np.angle(fft_complex)

        with self.lock:
            noise_est = self.noise_profile
            if len(noise_est) != len(mag_spec):
                # Resample noise profile to matching FFT length
                indices = np.linspace(0, len(noise_est) - 1, len(mag_spec))
                noise_est = np.interp(indices, np.arange(len(noise_est)), noise_est)

        # Spectral subtraction: S_sub = max(mag - alpha * noise, beta * noise)
        subtracted = mag_spec - (self.alpha * noise_est)
        floor = self.beta * mag_spec
        cleaned_mag = np.maximum(subtracted, floor)

        # Reconstruct complex FFT and inverse FFT
        cleaned_fft = cleaned_mag * np.exp(1j * phase_spec)
        cleaned_time = np.fft.irfft(cleaned_fft, n=n)

        # Blend with original to prevent musical noise artifacts
        output = 0.85 * cleaned_time + 0.15 * signal
        return output.astype(np.float32)


class MultiFeatureVAD:
    """Voice Activity Detector combining energy, spectral centroid, ZCR, and voice-band ratio.
    
    Distinguishes:
    - Quiet human speech vs ambient fan noise
    - High-frequency keyboard clicks vs unvoiced fricatives
    - Music / playback vs direct speech
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.ambient_rms_floor = 0.002
        self.ambient_alpha = 0.98

    def update_ambient_floor(self, rms: float):
        """Adaptively track the minimum ambient noise floor."""
        if rms > 0.0001 and rms < 0.03:
            # Slow adaptation to ambient shifts
            self.ambient_rms_floor = (self.ambient_alpha * self.ambient_rms_floor) + ((1.0 - self.ambient_alpha) * rms)

    def evaluate(self, chunk: np.ndarray, is_assistant_speaking: bool = False) -> Dict[str, Any]:
        """Evaluate if an audio chunk contains active human speech."""
        if len(chunk) == 0:
            return {"is_speech": False, "confidence": 0.0, "rms": 0.0, "is_transient": False}

        rms = AudioSignalMetrics.compute_rms(chunk)
        zcr = AudioSignalMetrics.compute_zero_crossing_rate(chunk)
        spectral = AudioSignalMetrics.compute_spectral_features(chunk, self.sample_rate)
        centroid = spectral["centroid"]
        voice_band = spectral["voice_band_ratio"]
        flatness = spectral["spectral_flatness"]

        # 1. Transient Spike Detection (keyboard click / mouse tap)
        # Short impulsive spikes have high ZCR, high centroid (>3200) and high peak-to-RMS ratio
        peak = float(np.max(np.abs(chunk)))
        crest_factor = (peak / (rms + 1e-6))
        is_transient = (crest_factor > 4.5 and (zcr > 0.15 or centroid > 3200) and voice_band < 0.48)

        # 2. Rejection of playback echo when assistant is speaking
        if is_assistant_speaking:
            speech_threshold = max(0.025, self.ambient_rms_floor * 4.0)
        else:
            speech_threshold = max(0.0035, self.ambient_rms_floor * 1.5)

        # 3. Speech decision score
        # Voiced speech has harmonic peaks (low flatness < 0.05), voice band concentration (>0.20),
        # and centroid between 400 and 3800 Hz.
        # Continuous fan noise / hiss has high flatness (>0.05) or low centroid (<300 Hz).
        is_harmonic_speech = (flatness < 0.04) and (voice_band >= 0.20) and (400 <= centroid <= 3800)
        has_adequate_energy = (rms >= speech_threshold)

        if has_adequate_energy and is_harmonic_speech and not is_transient:
            confidence = min(1.0, float(rms / (speech_threshold * 2.0)) * 0.7 + voice_band * 0.3)
            is_speech = True
        elif has_adequate_energy and voice_band >= 0.45 and not is_transient and flatness < 0.08:
            # Strong voiced phoneme
            confidence = 0.85
            is_speech = True
        else:
            confidence = 0.0
            is_speech = False
            self.update_ambient_floor(rms)

        return {
            "is_speech": is_speech,
            "confidence": round(confidence, 3),
            "rms": round(rms, 6),
            "centroid": round(centroid, 1),
            "voice_band_ratio": round(voice_band, 3),
            "spectral_flatness": round(flatness, 6),
            "zcr": round(zcr, 4),
            "is_transient": is_transient
        }


class AudioFrontEnd:
    """Comprehensive local audio front end integrating capture, noise reduction, VAD, and AGC."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.04,  # 40ms frame
        pre_roll_duration: float = 0.50
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.pre_roll_chunks = int(pre_roll_duration / chunk_duration)

        self.noise_suppressor = SpectralNoiseSuppressor(sample_rate=sample_rate)
        self.vad = MultiFeatureVAD(sample_rate=sample_rate)

        self._active_stream: Optional[Any] = None
        self._is_capturing = False
        self._assistant_is_speaking = False
        self._device_index: Optional[int] = None
        self._lock = threading.Lock()

    def set_assistant_speaking(self, speaking: bool):
        """Notify front end that assistant TTS is playing (enables echo cancellation / barge-in mode)."""
        self._assistant_is_speaking = speaking

    def is_assistant_speaking(self) -> bool:
        return self._assistant_is_speaking

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """List all available microphone input devices."""
        if sd is None:
            return []
        try:
            devices = []
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": i,
                        "name": d.get("name", f"Device {i}"),
                        "channels": d.get("max_input_channels"),
                        "default_samplerate": d.get("default_samplerate")
                    })
            return devices
        except Exception:
            return []

    def set_input_device(self, device_index: Optional[int]):
        """Select a specific input microphone device."""
        self._device_index = device_index

    def apply_agc(self, audio: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
        """Dynamic soft Automatic Gain Control: safely elevates quiet speech without clipping."""
        if len(audio) == 0:
            return audio

        current_rms = AudioSignalMetrics.compute_rms(audio)
        if current_rms < 1e-6:
            return audio

        # Compute required gain
        gain = target_rms / current_rms
        # Clamp maximum gain boost to 6.0x (+15dB) to prevent noise explosion
        gain = min(gain, 6.0)
        # Clamp minimum attenuation to 0.2x (-14dB)
        gain = max(gain, 0.2)

        boosted = audio * gain

        # Soft peak limiter to prevent digital clipping
        peak = float(np.max(np.abs(boosted)))
        if peak > 0.95:
            boosted = boosted * (0.92 / peak)

        return boosted.astype(np.float32)

    def record_utterance(
        self,
        silence_timeout: float = 1.2,
        max_duration: float = 18.0,
        init_timeout: float = 8.0,
        on_voice_start: Optional[Callable[[], None]] = None,
        on_partial_audio: Optional[Callable[[np.ndarray], None]] = None,
        partial_interval: float = 0.8
    ) -> Optional[np.ndarray]:
        """Record a single user utterance with pre-roll, dynamic VAD, AGC, and spectral noise gating.
        
        - Buffers pre-roll frames so word onset is preserved.
        - Triggers `on_voice_start` on speech onset (enabling immediate barge-in).
        - Calls `on_partial_audio` every `partial_interval` seconds during speech for streaming STT.
        - Ends turn adaptively when trailing silence reaches `silence_timeout`.
        - Returns float32 1D numpy array, or None if timed out with no speech.
        """
        if sd is None:
            return None

        pre_roll: List[np.ndarray] = []
        speech_chunks: List[np.ndarray] = []
        speech_started = False
        trailing_silence = 0.0
        start_time = time.time()
        last_partial_time = 0.0

        # Ambient calibration chunks
        calib_chunks = 0
        max_calib = 6

        try:
            device = self._device_index
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device,
                blocksize=self.chunk_samples
            ) as stream:
                while True:
                    chunk, overflowed = stream.read(self.chunk_samples)
                    chunk_flat = chunk.flatten()
                    now = time.time()
                    elapsed_total = now - start_time

                    # 1. Calibration phase
                    if calib_chunks < max_calib:
                        self.noise_suppressor.update_noise_profile(chunk_flat)
                        calib_chunks += 1
                        pre_roll.append(chunk_flat)
                        continue

                    # 2. Noise suppression & VAD evaluation
                    cleaned_chunk = self.noise_suppressor.suppress(chunk_flat)
                    vad_res = self.vad.evaluate(
                        cleaned_chunk,
                        is_assistant_speaking=self._assistant_is_speaking
                    )

                    # 3. Waiting for speech onset
                    if not speech_started:
                        pre_roll.append(chunk_flat)
                        if len(pre_roll) > self.pre_roll_chunks:
                            pre_roll.pop(0)

                        if vad_res["is_speech"]:
                            speech_started = True
                            if on_voice_start:
                                try:
                                    on_voice_start()
                                except Exception:
                                    pass
                            speech_chunks.extend(pre_roll)
                            speech_chunks.append(chunk_flat)
                            trailing_silence = 0.0
                            last_partial_time = now
                        elif elapsed_total >= init_timeout:
                            # Silence timeout before speech started
                            return None

                    # 4. Speech in progress
                    else:
                        speech_chunks.append(chunk_flat)

                        # Emit partial audio chunk for streaming STT
                        if on_partial_audio and (now - last_partial_time >= partial_interval):
                            accumulated = np.concatenate(speech_chunks)
                            try:
                                on_partial_audio(accumulated)
                            except Exception:
                                pass
                            last_partial_time = now

                        if not vad_res["is_speech"]:
                            trailing_silence += self.chunk_duration
                            if trailing_silence >= silence_timeout:
                                break
                        else:
                            trailing_silence = 0.0

                        if elapsed_total >= max_duration:
                            break

            if speech_chunks:
                full_raw = np.concatenate(speech_chunks)
                # Apply noise suppression and dynamic AGC
                cleaned_full = self.noise_suppressor.suppress(full_raw)
                normalized = self.apply_agc(cleaned_full)
                return normalized

            return None

        except Exception as e:
            sys.stderr.write(f"[AudioFrontEnd Error] {e}\n")
            return None

    @staticmethod
    def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Export float32 numpy array to standard 16-bit PCM WAV bytes."""
        clipped = np.clip(audio, -1.0, 1.0)
        int16_pcm = (clipped * 32767).astype(np.int16)
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_pcm.tobytes())
        return bio.getvalue()
