"""
Audio Module
- Microphone recording
- Speech recognition: faster-whisper (local offline) with Google STT fallback
- Text-to-speech (pyttsx3)
- Text-based emotion analysis
"""
import io
import queue
import threading
import time
import tempfile
import wave
import os
from typing import Optional, Callable

import speech_recognition as sr
import pyttsx3
from loguru import logger


def _init_whisper_model():
    """Initialize faster-whisper local model (tiny ~75MB, auto-downloaded on first run)"""
    try:
        from faster_whisper import WhisperModel
        # tiny model is fastest; use small/medium for better accuracy
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("faster-whisper tiny model loaded (local offline recognition)")
        print("[Speech] faster-whisper local model ready")
        return model
    except Exception as e:
        logger.warning("faster-whisper unavailable, falling back to Google STT: {}", e)
        print(f"[Speech] faster-whisper unavailable: {e}")
        return None


def _recognize_with_whisper(whisper_model, audio: sr.AudioData, language: str = "zh") -> str:
    """Recognize speech using faster-whisper from AudioData"""
    # Write AudioData to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        wav_data = audio.get_wav_data()
        f.write(wav_data)
    try:
        lang_code = language.split("-")[0]  # "zh-CN" -> "zh"
        segments, info = whisper_model.transcribe(
            tmp_path, language=lang_code, beam_size=3,
            vad_filter=True,          # silence filter to reduce false recognitions
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = "".join(seg.text for seg in segments).strip()
        return text
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


class SpeechRecognizer:
    """Speech recognizer: microphone -> text
    Prefers faster-whisper local offline recognition; no error if network unavailable.
    """

    def __init__(self, cfg: dict):
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.record_seconds = cfg.get("record_seconds", 5)
        self.language = cfg.get("language", "zh-CN")
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._result_queue: queue.Queue = queue.Queue()
        # Pre-load whisper model in background (non-blocking)
        self._whisper_model = None
        self._whisper_ready = threading.Event()
        threading.Thread(target=self._load_whisper, daemon=True).start()

    def _load_whisper(self):
        self._whisper_model = _init_whisper_model()
        self._whisper_ready.set()

    def start(self, on_result: Optional[Callable[[str], None]] = None):
        """Start continuous speech recognition in background thread"""
        self._running = True
        self._on_result = on_result
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Speech recognition started, language: {}", self.language)

    def _recognize(self, audio: sr.AudioData) -> str:
        """Recognize audio: try whisper first, fall back to Google STT"""
        # Wait for whisper to finish loading (max 30s)
        self._whisper_ready.wait(timeout=30)
        if self._whisper_model:
            try:
                text = _recognize_with_whisper(self._whisper_model, audio, self.language)
                if text:
                    return text
            except Exception as e:
                logger.debug("Whisper recognition error, trying Google: {}", e)
        # Fall back to Google STT
        return self.recognizer.recognize_google(audio, language=self.language)

    def _listen_loop(self):
        with sr.Microphone(sample_rate=self.sample_rate) as source:
            logger.info("Calibrating ambient noise, please wait...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Speech recognition ready, listening...")
            print("\n[Speech] Microphone ready, please speak...\n")
            while self._running:
                try:
                    print("[Speech] Listening...")
                    audio = self.recognizer.listen(
                        source, timeout=5, phrase_time_limit=self.record_seconds
                    )
                    print("[Speech] Recognizing...")
                    text = self._recognize(audio)
                    if not text:
                        print("[Speech] Could not recognize speech, please try again")
                        continue
                    print(f"\n[Speech Result] >> {text}\n")
                    logger.info("Speech recognized: {}", text)
                    self._result_queue.put(text)
                    if self._on_result:
                        self._on_result(text)
                except sr.WaitTimeoutError:
                    pass   # silent - no output when no speech detected
                except sr.UnknownValueError:
                    print("[Speech] Could not recognize speech, please try again")
                    logger.debug("Unable to recognize speech content")
                except sr.RequestError as e:
                    logger.debug("Google STT unavailable (using local recognition): {}", e)
                except Exception as e:
                    logger.error("Speech recognition error: {}", e)

    def get_latest_text(self) -> Optional[str]:
        """Non-blocking get of latest recognized text"""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Speech recognition stopped")


class TextToSpeech:
    """Text-to-speech: text -> speaker output"""

    def __init__(self, cfg: dict):
        self.rate = cfg.get("tts_rate", 150)
        self.volume = cfg.get("tts_volume", 1.0)
        self._engine: Optional[pyttsx3.Engine] = None
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error("TTS initialization failed: {}", e)
            self._engine = None

    def speak(self, text: str):
        """Play text (thread-safe)"""
        if not self._engine:
            logger.warning("TTS engine not initialized, skipping: {}", text)
            return
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.error("TTS playback error: {}", e)

    def speak_async(self, text: str):
        """Async playback (non-blocking)"""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()


class AudioEmotionAnalyzer:
    """Text-based emotion analysis from recognized speech"""

    POSITIVE = {"happy", "joy", "glad", "great", "good", "nice", "wonderful", "excellent",
                "love", "like", "satisfied", "pleased",
                # Chinese keywords
                "开心", "高兴", "快乐", "喜欢", "好", "棒", "太好了", "很好", "不错", "满意", "愉快"}
    NEGATIVE = {"sad", "unhappy", "upset", "angry", "hate", "tired", "pain", "anxious", "bad", "terrible",
                # Chinese keywords
                "难过", "悲伤", "生气", "愤怒", "讨厌", "伤心", "烦", "累", "痛苦", "焦虑", "不好"}

    def analyze(self, text: str) -> Optional[dict]:
        """Simple keyword-based text emotion analysis (can be replaced with a model)"""
        if not text:
            return None
        text_lower = text.lower()
        score = 0
        for word in self.POSITIVE:
            if word in text_lower or word in text:
                score += 1
        for word in self.NEGATIVE:
            if word in text_lower or word in text:
                score -= 1
        if score > 0:
            emotion, emotion_en, confidence = "happy", "happy", min(0.5 + score * 0.1, 0.9)
        elif score < 0:
            emotion, emotion_en, confidence = "sad", "sad", min(0.5 + abs(score) * 0.1, 0.9)
        else:
            emotion, emotion_en, confidence = "neutral", "neutral", 0.5
        return {
            "emotion": emotion,
            "emotion_en": emotion_en,
            "confidence": round(confidence, 3),
            "text": text,
        }
