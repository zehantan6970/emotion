"""
Emotion Analysis System - Main Entry
Integrates camera, speech, and multimodal fusion for real-time emotion analysis
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load environment variables from .env file (tokens, API keys, etc.)
# Tokens are stored in .env (gitignored), never hardcoded here.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed; set env vars manually or via system

import time
import threading
import sys
import io
import warnings
warnings.filterwarnings("ignore")
# Fix Windows GBK terminal encoding issues
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("gbk", "gb2312", "cp936"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger, SessionRecorder
from src.vision import CameraCapture, VisionEmotionAnalyzer
from src.audio import SpeechRecognizer, TextToSpeech, AudioEmotionAnalyzer
from src.fusion import EmotionFusion


class EmotionAnalysisSystem:
    """Multimodal Emotion Analysis System"""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        setup_logger(self.config)
        logger.info("=" * 50)
        logger.info("Emotion Analysis System Starting")
        logger.info("=" * 50)

        # Initialize modules
        cam_cfg = self.config.get("camera", {})
        vis_cfg = self.config.get("vision_emotion", {})
        aud_cfg = self.config.get("audio", {})
        fus_cfg = self.config.get("fusion", {})
        out_cfg = self.config.get("output", {})

        self.camera = CameraCapture(cam_cfg)
        self.vision_analyzer = VisionEmotionAnalyzer(vis_cfg)
        self.speech_recognizer = SpeechRecognizer(aud_cfg)
        self.tts = TextToSpeech(aud_cfg)
        self.audio_emotion = AudioEmotionAnalyzer()
        self.fusion = EmotionFusion(fus_cfg)
        self.session_recorder = SessionRecorder()  # structured session data

        self.show_window = cam_cfg.get("show_window", True)
        self.window_name = cam_cfg.get("window_name", "Emotion Analysis")
        self.analysis_interval = out_cfg.get("analysis_interval", 1.0)
        self.console_output = out_cfg.get("console_output", True)

        self._running = False
        self._latest_audio_emotion = None
        self._latest_speech_text = None
        self._audio_lock = threading.Lock()

    def _on_speech_recognized(self, text: str):
        """Speech recognition callback - triggers audio emotion analysis"""
        result = self.audio_emotion.analyze(text)
        with self._audio_lock:
            self._latest_speech_text = text
            self._latest_audio_emotion = result
        # Print recognized speech prominently
        print(f"\n{'─'*40}")
        print(f"  [Speech] You said: {text}")
        if result:
            print(f"  [Audio Emotion] {result['emotion_en']}  Confidence: {result['confidence']:.0%}")
            logger.info("[Audio Emotion] {} | {} | Confidence: {:.0%}",
                        text, result["emotion_en"], result["confidence"])
            # TTS feedback
            feedback = self._generate_feedback(result["emotion_en"])
            if feedback:
                print(f"  [System Reply] {feedback}")
                self.tts.speak_async(feedback)
        print(f"{'─'*40}\n")

    def _generate_feedback(self, emotion: str) -> Optional[str]:
        """Generate TTS feedback based on detected emotion"""
        feedbacks = {
            "happy":    "Great to see you are in a good mood!",
            "sad":      "You seem a bit down. Want to talk about it?",
            "angry":    "I notice you seem upset. Take a deep breath.",
            "fear":     "Don't worry, everything will be fine.",
            "disgust":  "Looks like something is bothering you.",
            "surprise": "Something surprised you?",
        }
        return feedbacks.get(emotion)

    def run(self):
        """Start the main system loop"""
        # Start camera
        if not self.camera.start():
            logger.error("Failed to start camera, exiting")
            return

        # Start speech recognition
        self.speech_recognizer.start(on_result=self._on_speech_recognized)

        self._running = True
        last_analysis_time = 0
        last_vision_result = None

        logger.info("System running - press ESC or Q in the video window to quit")
        if self.console_output:
            print("\n[Emotion Analysis System Started]")
            print("  To quit: focus the video window and press ESC or Q, or press Ctrl+C\n")

        try:
            while self._running:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()

                # Run vision analysis at configured interval
                if now - last_analysis_time >= self.analysis_interval:
                    last_analysis_time = now
                    vision_result = self.vision_analyzer.analyze(frame)

                    with self._audio_lock:
                        audio_result = self._latest_audio_emotion

                    # Multimodal fusion
                    fused = self.fusion.fuse(vision_result, audio_result)
                    last_vision_result = vision_result

                    # Save structured session record
                    if fused["confidence"] > 0:
                        with self._audio_lock:
                            speech_text = self._latest_speech_text
                        self.session_recorder.record(
                            fused_result=fused,
                            vision_result=vision_result,
                            audio_result=audio_result,
                            speech_text=speech_text,
                        )

                    # Console output
                    if self.console_output and fused["confidence"] > 0:
                        self._print_result(fused)

                    logger.debug("Fused emotion: {} | Confidence: {:.0%}", fused["emotion_en"], fused["confidence"])

                # Overlay results on video window
                if self.show_window:
                    display_frame = self.vision_analyzer.draw_result(frame.copy(), last_vision_result)
                    cv2.imshow(self.window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    # ESC(27) or Q to quit
                    if key == 27 or key == ord("q"):
                        logger.info("User pressed quit key, shutting down")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupt signal received, shutting down...")
        finally:
            self._shutdown()

    def _print_result(self, result: dict):
        """Print fused emotion result to console"""
        bar = "=" * 40
        print(f"\n{bar}")
        print(f"  Current Emotion: {result['emotion_en']}  Confidence: {result['confidence']:.0%}")
        print(f"  Emotion Scores:")
        for emotion, score in result["all_scores"].items():
            filled = int(score * 20)
            bar_str = "#" * filled + "-" * (20 - filled)
            print(f"    {emotion:8s}: [{bar_str}] {score:.0%}")
        if result["sources"]["audio"] and result["sources"]["audio"].get("text"):
            print(f"  Speech Text: {result['sources']['audio']['text']}")
        print(bar)

    def _shutdown(self):
        self._running = False
        self.camera.stop()
        self.speech_recognizer.stop()
        self.session_recorder.close()  # flush & save session JSON
        if self.show_window:
            cv2.destroyAllWindows()
        logger.info("System stopped")


def main():
    system = EmotionAnalysisSystem()
    system.run()


if __name__ == "__main__":
    main()
