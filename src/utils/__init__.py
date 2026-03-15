"""
Utility module: config loading, logger initialization, and common components
"""
import os
import sys
import yaml
from pathlib import Path
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

    # File output (rotate daily)
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
