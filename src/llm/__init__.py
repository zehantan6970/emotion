"""
LLM Emotional Therapy Assistant — Phase 3 Placeholder

Planned features:
- Receive per-person emotion history from EmotionFusion
- Generate personalized therapy responses via LLM (GPT / local Ollama)
- Session management: track emotional arc across a conversation
- TTS output of therapy response

Interface (to be implemented):
    therapist = LLMTherapist(cfg)
    response = therapist.respond(person_id="person_0", emotion_history=[...])
"""


class LLMTherapist:
    """Placeholder for Phase 3 LLM-based emotional therapy assistant."""

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        # TODO: initialize LLM client (OpenAI / Anthropic / Ollama)

    def respond(self, person_id: str, emotion_history: list) -> str:
        """
        Generate a therapy response based on a person's emotion history.

        Args:
            person_id: Unique identifier for the person
            emotion_history: List of recent emotion dicts from EmotionFusion

        Returns:
            English therapy response string (to be spoken via TTS)
        """
        # TODO: implement LLM call
        raise NotImplementedError("LLMTherapist is not yet implemented (Phase 3)")
