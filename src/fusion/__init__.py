"""
Multimodal Emotion Fusion Module
- Fuses vision and audio emotion results
- Smooths output to reduce fluctuation
- Extensible plugin interface
"""
from collections import deque
from typing import Optional, Dict, Any, List

from loguru import logger


# Emotion to vector mapping (7-dimensional standard emotion space)
EMOTION_VECTOR = {
    "happy":    [0, 0, 0, 1, 0, 0, 0],
    "sad":      [0, 0, 0, 0, 1, 0, 0],
    "angry":    [1, 0, 0, 0, 0, 0, 0],
    "disgust":  [0, 1, 0, 0, 0, 0, 0],
    "fear":     [0, 0, 1, 0, 0, 0, 0],
    "surprise": [0, 0, 0, 0, 0, 1, 0],
    "neutral":  [0, 0, 0, 0, 0, 0, 1],
}
EMOTION_LABELS_EN = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionFusion:
    """
    Multimodal emotion fusion
    Supports: weighted average fusion + temporal smoothing
    Extensible: confidence-adaptive weights, plugin interface
    """

    def __init__(self, cfg: dict):
        self.vision_weight = cfg.get("vision_weight", 0.6)
        self.audio_weight = cfg.get("audio_weight", 0.4)
        self.smoothing_window = cfg.get("smoothing_window", 5)
        self._history: deque = deque(maxlen=self.smoothing_window)
        self._plugins: List = []  # extensible plugin list

    def fuse(
        self,
        vision_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Fuse vision and audio emotion results
        Returns: fused emotion result dict
        """
        # Convert results to vectors
        v_vec = self._to_vector(vision_result)
        a_vec = self._to_vector(audio_result)

        # Dynamic weights (only modalities with data contribute)
        has_vision = vision_result is not None
        has_audio = audio_result is not None

        if has_vision and has_audio:
            w_v, w_a = self.vision_weight, self.audio_weight
        elif has_vision:
            w_v, w_a = 1.0, 0.0
        elif has_audio:
            w_v, w_a = 0.0, 1.0
        else:
            return self._empty_result()

        # Weighted fusion
        fused = [w_v * v + w_a * a for v, a in zip(v_vec, a_vec)]

        # Temporal smoothing
        self._history.append(fused)
        smoothed = [
            sum(h[i] for h in self._history) / len(self._history)
            for i in range(len(fused))
        ]

        # Normalize and find top emotion
        total = sum(smoothed) or 1.0
        normalized = [s / total for s in smoothed]
        top_idx = normalized.index(max(normalized))

        result = {
            "emotion": EMOTION_LABELS_EN[top_idx],
            "emotion_en": EMOTION_LABELS_EN[top_idx],
            "confidence": round(normalized[top_idx], 3),
            "all_scores": {
                EMOTION_LABELS_EN[i]: round(normalized[i], 3)
                for i in range(len(EMOTION_LABELS_EN))
            },
            "sources": {
                "vision": vision_result,
                "audio": audio_result,
            },
        }

        # Execute plugin hooks (extension interface)
        for plugin in self._plugins:
            try:
                plugin.on_result(result)
            except Exception as e:
                logger.warning("Plugin execution error: {}", e)

        return result

    def _to_vector(self, result: Optional[Dict]) -> List[float]:
        """Convert emotion result to 7-dimensional vector"""
        if result is None:
            return [1.0 / 7] * 7  # uniform distribution when no data
        emotion_en = result.get("emotion_en", "neutral")
        confidence = result.get("confidence", 0.5)
        base = [1.0 / 7 * (1 - confidence)] * 7
        vec = EMOTION_VECTOR.get(emotion_en, EMOTION_VECTOR["neutral"])
        return [base[i] + vec[i] * confidence for i in range(7)]

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "emotion": "neutral",
            "emotion_en": "neutral",
            "confidence": 0.0,
            "all_scores": {en: 0.0 for en in EMOTION_LABELS_EN},
            "sources": {"vision": None, "audio": None},
        }

    def register_plugin(self, plugin):
        """Register extension plugin (reserved interface)"""
        self._plugins.append(plugin)
        logger.info("Plugin registered: {}", plugin.__class__.__name__)

    def reset_history(self):
        """Clear smoothing history"""
        self._history.clear()
