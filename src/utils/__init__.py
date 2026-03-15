"""
Utility module: config loading, logger initialization, and session data recording
"""
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(config: dict) -> None:
    """Initialize loguru logging system"""
    log_cfg = config.get("logging", {})
    log_dir = PROJECT_ROOT / log_cfg.get("log_dir", "logs")
    log_dir.mkdir(exist_ok=True)

    logger.remove()  # Remove default handler

    # Console output
    logger.add(
        sys.stderr,
        level=log_cfg.get("level", "INFO"),
        format=log_cfg.get(
            "format",
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        ),
        colorize=True,
    )

    # File output (rotate by size, not daily — avoids missing intra-day data)
    logger.add(
        str(log_dir / "emotion_{time:YYYY-MM-DD}.log"),
        level=log_cfg.get("level", "INFO"),
        format=log_cfg.get(
            "format",
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        ),
        rotation=log_cfg.get("max_size", "10 MB"),
        retention=log_cfg.get("retention", "7 days"),
        encoding="utf-8",
    )

    logger.info("Logger initialized, log directory: {}", str(log_dir))


class SessionRecorder:
    """
    Records structured emotion session data to JSON files under data/sessions/.

    Each session = one run of main.py.
    Each record = one emotion analysis cycle (timestamp + vision + audio + fused result).

    Purpose:
    - Feed historical emotion trends to LLM therapy module (Phase 3)
    - Provide timestamped labels for EEG signal alignment (Phase 4)
    - Enable per-user longitudinal emotion tracking (multi-person, Phase 2)
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = PROJECT_ROOT / "data" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.session_file = self.session_dir / f"session_{self.session_id}.json"
        self.records = []
        self.meta = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_records": 0,
        }
        logger.info("SessionRecorder started, file: {}", str(self.session_file))

    def record(
        self,
        fused_result: dict,
        vision_result: dict = None,
        audio_result: dict = None,
        speech_text: str = None,
    ):
        """
        Append one analysis frame to the session record.

        Args:
            fused_result:  Output from EmotionFusion.fuse()
            vision_result: Raw VisionEmotionAnalyzer output (optional)
            audio_result:  Raw AudioEmotionAnalyzer output (optional)
            speech_text:   Original recognized speech text (preserves user language)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "fused": {
                "emotion": fused_result.get("emotion_en", "unknown"),
                "confidence": round(fused_result.get("confidence", 0), 4),
                "all_scores": {
                    k: round(v, 4)
                    for k, v in fused_result.get("all_scores", {}).items()
                },
            },
        }

        if vision_result:
            entry["vision"] = {
                "emotion": vision_result.get("emotion_en", "unknown"),
                "confidence": round(vision_result.get("confidence", 0), 4),
            }

        if audio_result:
            entry["audio"] = {
                "emotion": audio_result.get("emotion_en", "unknown"),
                "confidence": round(audio_result.get("confidence", 0), 4),
            }

        # Speech text preserved as-is (Chinese stays Chinese, English stays English)
        if speech_text:
            entry["speech_text"] = speech_text

        self.records.append(entry)

        # Flush to disk every 10 records to avoid data loss on crash
        if len(self.records) % 10 == 0:
            self._flush()

    def close(self):
        """Finalize and save the complete session file."""
        self.meta["end_time"] = datetime.now().isoformat()
        self.meta["total_records"] = len(self.records)
        self._flush()
        logger.info(
            "SessionRecorder closed: {} records saved to {}",
            len(self.records),
            str(self.session_file),
        )

    def _flush(self):
        """Write current state to disk (atomic via temp file)."""
        payload = {**self.meta, "records": self.records}
        tmp = self.session_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self.session_file)  # atomic rename
