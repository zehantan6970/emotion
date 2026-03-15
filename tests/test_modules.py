"""
测试：视觉模块单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_camera_import():
    from src.vision import CameraCapture, VisionEmotionAnalyzer
    print("[OK] 视觉模块导入成功")


def test_audio_import():
    from src.audio import SpeechRecognizer, TextToSpeech, AudioEmotionAnalyzer
    print("[OK] 音频模块导入成功")


def test_fusion_import():
    from src.fusion import EmotionFusion
    cfg = {"vision_weight": 0.6, "audio_weight": 0.4, "smoothing_window": 3}
    fusion = EmotionFusion(cfg)

    vision = {"emotion": "开心", "emotion_en": "happy", "confidence": 0.8}
    audio = {"emotion": "平静", "emotion_en": "neutral", "confidence": 0.6}
    result = fusion.fuse(vision, audio)
    assert "emotion" in result
    assert "confidence" in result
    print(f"[OK] 融合模块测试通过: {result['emotion']} ({result['confidence']:.0%})")


def test_audio_emotion():
    from src.audio import AudioEmotionAnalyzer
    analyzer = AudioEmotionAnalyzer()
    result = analyzer.analyze("今天天气真好，我很开心！")
    assert result["emotion"] == "开心"
    print(f"[OK] 文本情感分析: '{result['text']}' -> {result['emotion']}")

    result2 = analyzer.analyze("我很难过，今天很糟糕")
    assert result2["emotion"] == "悲伤"
    print(f"[OK] 文本情感分析: '{result2['text']}' -> {result2['emotion']}")


if __name__ == "__main__":
    print("=" * 40)
    print("情感分析工程模块测试")
    print("=" * 40)
    test_camera_import()
    test_audio_import()
    test_fusion_import()
    test_audio_emotion()
    print("=" * 40)
    print("所有测试通过！")
