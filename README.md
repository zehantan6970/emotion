# Multimodal Emotion Analysis System

> Real-time emotion recognition using camera (vision) + microphone (speech) + (planned) EEG signals, with LLM-based emotional therapy assistance.

---

## Quick Start (Restore Context for AI)

> **If you are an AI assistant**: Read this entire README first. It contains the full project context, architecture, known issues, and roadmap. The developer will continue iterating on this project with you.

```bash
# Activate environment and run
conda activate emotion
python D:\pycode\emotion\main.py

# Or with full path
D:\miniconda3\envs\emotion\python.exe D:\pycode\emotion\main.py
```

---

## Project Overview

| Item | Value |
|---|---|
| **Goal** | Real-time multimodal emotion analysis + LLM therapy assistance |
| **OS** | Windows 11 |
| **Python** | 3.10 (conda env: `emotion`, path: `D:\miniconda3\envs\emotion`) |
| **Project Root** | `D:\pycode\emotion` |
| **GitHub** | https://github.com/zehantan6970/emotion |
| **Author** | zehantan6970 / zehantan6970@gmail.com |

---

## Current Features (v1.0)

- [x] Real-time camera capture (OpenCV)
- [x] Facial emotion detection (DeepFace + VGG-Face, ssd detector)
- [x] Speech recognition — Chinese Mandarin (faster-whisper local + Google STT fallback)
- [x] Text-based emotion analysis (keyword matching, bilingual CN/EN)
- [x] Multimodal fusion (weighted: vision 60% + audio 40%, sliding window smoothing)
- [x] TTS feedback (pyttsx3, Windows SAPI, English)
- [x] Structured logging (loguru, daily rotation)
- [x] All UI/console output in English (avoids Windows GBK encoding issues)

---

## Planned Features (Roadmap)

### Phase 2 — Multi-Person Emotion Recognition
- [ ] Track multiple faces simultaneously (face ID assignment)
- [ ] Per-person emotion timeline
- [ ] Group emotion aggregation

### Phase 3 — LLM Emotional Therapy
- [ ] Integrate LLM (GPT / local LLM via Ollama) as therapy assistant
- [ ] Per-person emotion history → LLM generates personalized response
- [ ] Session management: track user emotional arc across a conversation
- [ ] `src/llm/` module (pre-created placeholder)

### Phase 4 — EEG Signal Integration
- [ ] EEG device driver (OpenBCI / Emotiv / MindLink)
- [ ] EEG feature extraction (alpha/beta/theta bands)
- [ ] Fuse EEG emotion signal into multimodal fusion pipeline
- [ ] Use EEG as ground truth to validate LLM therapy effectiveness
- [ ] `src/eeg/` module (pre-created placeholder)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│               EmotionAnalysisSystem                     │
│   Orchestrates all modules, drives main loop, outputs   │
└───┬──────────┬──────────┬──────────┬────────────────────┘
    │          │          │          │
    ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌──────┐ ┌──────────┐ ┌─────┐ ┌─────┐
│Camera  │ │Vision  │ │Speech│ │  Audio   │ │ TTS │ │Fusio│
│Capture │ │Emotion │ │Recog.│ │ Emotion  │ │     │ │  n  │
└────────┘ └────────┘ └──────┘ └──────────┘ └─────┘ └─────┘
                                                        │
                              (planned) ┌───────┐  ┌───────┐
                                        │  LLM  │  │  EEG  │
                                        └───────┘  └───────┘
```

**Data Flow:**
```
Camera → CameraCapture → VisionEmotionAnalyzer ──┐
                                                  ├→ EmotionFusion → Output
Mic    → SpeechRecognizer → AudioEmotionAnalyzer ─┘
                  │
                  └→ TextToSpeech (English feedback)

(Phase 3) EmotionFusion → LLM Therapy Assistant → TTS / Text response
(Phase 4) EEG Device   → EEGEmotionAnalyzer    → EmotionFusion
```

**Threading model:**
- Main thread: video frame processing + OpenCV window rendering
- Background thread 1: `CameraCapture._capture_loop` (continuous `cap.read()`)
- Background thread 2: `SpeechRecognizer._listen_loop` (continuous mic listening)
- Background thread 3: Whisper model loading (non-blocking startup)
- Shared state protected by `threading.Lock`

---

## File Structure

```
D:\pycode\emotion\
├── main.py                      # System entry point
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment spec
├── .env                         # Local secrets — GITIGNORED, never commit
├── .env.example                 # Template for secrets (safe to commit)
├── .gitignore
├── README.md                    # This file
├── config/
│   └── settings.yaml            # All hardware and model parameters
├── src/
│   ├── vision/__init__.py       # Camera capture + DeepFace emotion analysis
│   ├── audio/__init__.py        # Speech recognition + TTS + text emotion
│   ├── fusion/__init__.py       # Multimodal fusion + temporal smoothing
│   ├── utils/__init__.py        # Config loading + logger init
│   ├── llm/                     # [Phase 3] LLM therapy assistant (placeholder)
│   └── eeg/                     # [Phase 4] EEG signal processing (placeholder)
├── logs/                        # Daily rotating log files (gitignored)
├── models/                      # Model weights cache (gitignored)
└── tests/
    └── test_modules.py
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/zehantan6970/emotion.git
cd emotion
```

### 2. Create conda environment
```bash
conda create -n emotion python=3.10 -y
conda activate emotion
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# Or use conda:
# conda env create -f environment.yml
```

### 4. Configure secrets
```bash
# Copy the template and fill in your tokens
cp .env.example .env
# Edit .env:
#   HF_TOKEN=your_huggingface_token
```

### 5. Run
```bash
python main.py
```

---

## Configuration

All parameters are in `config/settings.yaml`:

```yaml
camera:
  device_id: 0              # Webcam device ID
  window_name: "Emotion Analysis - Live"

audio:
  language: "zh-CN"         # Speech language: Chinese Mandarin
  record_seconds: 5

vision_emotion:
  backend: "deepface"
  detector_backend: "ssd"   # ssd recommended; retinaface for GPU
  confidence_threshold: 0.15

fusion:
  vision_weight: 0.6
  audio_weight: 0.4
  smoothing_window: 5       # Temporal smoothing (frames)
```

---

## Known Issues & Solutions (Development Log)

| # | Symptom | Root Cause | Solution |
|---|---|---|---|
| 1 | Chinese garbled in video window / terminal | OpenCV has no CJK font; Windows terminal is GBK | **All UI/system text in English** |
| 2 | "No face detected" despite visible face | `opencv` detector low accuracy; threshold 0.3 too high | `ssd` detector + threshold 0.15 + 3-level fallback |
| 3 | Speech recognized but nothing printed | Only `logger.info` used; errors silently swallowed | Added `print` at every stage of `_listen_loop` |
| 4 | ESC key does not quit | `waitKey` only checked `q`, missed ESC (ASCII=27) | `key == 27 or key == ord("q")` |
| 5 | TTS garbled / wrong pronunciation | Windows SAPI English voice cannot speak Chinese | TTS output fixed to English sentences |
| 6 | Slow startup (blocked) | Whisper model load blocked main thread | Load in background thread with `threading.Event` |
| 7 | Emotion result jumps every frame | Single-frame analysis is noisy | Sliding window mean (`deque maxlen=5`) |
| 8 | `UnicodeEncodeError` on Windows | stdout defaults to GBK, cannot write UTF-8 | Rewrap `sys.stdout` as UTF-8 at startup |

---

## Speech Interaction Behavior

- **User speaks Chinese** → printed as Chinese: `[Speech] You said: 今天心情不错`
- **User speaks English** → printed as English: `[Speech] You said: I feel great`
- **System labels / TTS** → always English (avoids SAPI/encoding issues)

---

## Environment Variables Reference

| Variable | Purpose | Where to get |
|---|---|---|
| `HF_TOKEN` | Download HuggingFace models | https://huggingface.co/settings/tokens |
| `HUGGINGFACE_TOKEN` | Same, legacy name for older libs | Same as above |
| `OPENAI_API_KEY` | (Phase 3) LLM therapy | https://platform.openai.com |
| `ANTHROPIC_API_KEY` | (Phase 3) Claude LLM | https://console.anthropic.com |
| `EEG_DEVICE_PORT` | (Phase 4) EEG serial port | Check Device Manager |

Set these in `.env` (never commit this file).

---

## Git Workflow

```bash
# Feature development
git checkout -b feature/multi-person-tracking
# ... develop ...
git add .
git commit -m "feat(vision): add multi-person face tracking with ID assignment"
git push origin feature/multi-person-tracking
# Create Pull Request on GitHub → merge to main
```

**Branch naming convention:**
- `feature/xxx` — new feature
- `fix/xxx` — bug fix
- `exp/xxx` — experiment (EEG, new model, etc.)

**Commit message format:**
```
feat(module): short description
fix(module): short description
exp(eeg): try OpenBCI alpha band extraction
```

---

## AI Assistant Context Resume

> Paste this section (plus any relevant code snippets) at the start of a new AI conversation to instantly restore full project context.

```
I am continuing development of a multimodal real-time emotion analysis project.

Project: D:\pycode\emotion
GitHub: https://github.com/zehantan6970/emotion
Environment: Windows 11, conda env "emotion", Python 3.10, D:\miniconda3
Stack: DeepFace + faster-whisper + pyttsx3 + SpeechRecognition + loguru

Current status: v1.0 complete — camera vision + speech recognition + multimodal fusion working.
All UI is English (avoids Windows GBK/OpenCV encoding issues).
Speech output preserves user's original language (CN or EN).
Tokens stored in .env (gitignored).

Next task: [describe what you want to do next]

Relevant files to read first:
- README.md (this file)
- config/settings.yaml
- src/[module]/__init__.py (whichever is relevant)
```

---

## License

MIT
