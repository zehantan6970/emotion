# Multimodal Real-Time Emotion Analysis System

> Real-time emotion recognition using **Camera (Vision)** + **Microphone (Speech)** with multimodal fusion.
> Roadmap: Multi-person tracking → LLM emotional therapy → EEG signal validation.

---

## Quick Start

```bash
# Activate environment
conda activate emotion

# Run
python D:\pycode\emotion\main.py
# or full path:
D:\miniconda3\envs\emotion\python.exe D:\pycode\emotion\main.py
```

**Press `ESC` or `Q` in the video window to quit.**

---

## Project Info

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.10 — conda env `emotion` at `D:\miniconda3\envs\emotion` |
| Project Root | `D:\pycode\emotion` |
| GitHub | https://github.com/zehantan6970/emotion |
| Author | zehantan6970 / zehantan6970@gmail.com |

---

## File Structure

```
D:\pycode\emotion\
│
├── main.py                        # System entry — EmotionAnalysisSystem
├── requirements.txt               # pip dependencies
├── .env                           # Local secrets (GITIGNORED — never commit)
├── .env.example                   # Secret template (safe to commit)
├── .gitignore
├── README.md                      # This file
│
├── config/
│   └── settings.yaml              # All tunable parameters (camera/audio/vision/fusion)
│
├── src/
│   ├── utils/__init__.py          # Config loader + Logger init + SessionRecorder
│   ├── vision/__init__.py         # CameraCapture + VisionEmotionAnalyzer + draw_result
│   ├── audio/__init__.py          # SpeechRecognizer + TextToSpeech + AudioEmotionAnalyzer
│   ├── fusion/__init__.py         # EmotionFusion (weighted average + temporal smoothing)
│   ├── llm/__init__.py            # [Phase 3 placeholder] LLMTherapist
│   └── eeg/__init__.py            # [Phase 4 placeholder] EEGEmotionAnalyzer
│
├── data/
│   └── sessions/                  # Structured emotion session JSON files (gitignored)
│       └── session_YYYYMMDD_HHMMSS.json
│
├── logs/                          # Runtime log files — daily rotation (gitignored)
│   └── emotion_YYYY-MM-DD.log
│
└── models/                        # Downloaded model weights cache (gitignored)
```

---

## System Architecture

### Module Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           main.py                               │
│                    EmotionAnalysisSystem                        │
│   Orchestrates all modules · drives main loop · outputs results │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│Camera│ │Vision  │ │Speech  │ │  TTS   │ │  Audio   │
│Captur│ │Emotion │ │Recogniz│ │        │ │ Emotion  │
│  e   │ │Analyzer│ │  -er   │ │        │ │ Analyzer │
└──┬───┘ └───┬────┘ └───┬────┘ └────────┘ └────┬─────┘
   │         │           │                       │
   │  frames │           │ text callback         │ emotion dict
   └────┬────┘           └──────────┬────────────┘
        │                           │
        ▼                           ▼
   ┌─────────────────────────────────┐
   │          EmotionFusion          │
   │  weighted average + smoothing   │
   └────────────────┬────────────────┘
                    │ fused emotion dict
                    ▼
         ┌──────────────────┐
         │  SessionRecorder │   → data/sessions/session_*.json
         │  (utils module)  │
         └──────────────────┘
                    │
                    ▼
         Console print + logs/emotion_YYYY-MM-DD.log
         OpenCV video window OSD

[Phase 3 — planned]
EmotionFusion ──→ LLMTherapist ──→ TTS (spoken therapy response)

[Phase 4 — planned]
EEG Device ──→ EEGEmotionAnalyzer ──→ EmotionFusion (3rd modality)
```

### Data Flow

```
Camera
  └─► CameraCapture._capture_loop()   [background thread]
        └─► get_frame()               [main thread, non-blocking]
              └─► VisionEmotionAnalyzer.analyze(frame)
                    └─► DeepFace (VGG-Face, ssd detector)
                          └─► {"emotion_en": "happy", "confidence": 0.87, ...}

Microphone
  └─► SpeechRecognizer._listen_loop() [background thread]
        └─► faster-whisper (local)  OR  Google STT (fallback)
              └─► on_result(text) callback
                    └─► AudioEmotionAnalyzer.analyze(text)
                          └─► keyword matching → {"emotion_en": "happy", ...}
                    └─► TextToSpeech.speak_async(english_feedback)

EmotionFusion.fuse(vision_result, audio_result)
  └─► weighted vector sum (vision 60% + audio 40%, dynamic)
        └─► deque(maxlen=5) temporal smoothing
              └─► {"emotion_en": "happy", "confidence": 0.73, "all_scores": {...}}

SessionRecorder.record(fused, vision, audio, speech_text)
  └─► data/sessions/session_YYYYMMDD_HHMMSS.json  (flush every 10 records)
```

### Threading Model

| Thread | Role | Sync |
|---|---|---|
| Main thread | Video frame processing + OpenCV window rendering | — |
| `CameraCapture._capture_loop` | Continuous `cap.read()` loop | `threading.Lock` on `_frame` |
| `SpeechRecognizer._listen_loop` | Continuous microphone listening | `threading.Lock` on `_latest_audio_emotion` |
| Whisper loader | Load model at startup (non-blocking) | `threading.Event` (`_whisper_ready`) |
| TTS worker | Async TTS playback | `threading.Lock` on pyttsx3 engine |

---

## Module Reference

### `main.py` — System Orchestrator

**Class:** `EmotionAnalysisSystem`

| Item | Detail |
|---|---|
| **Purpose** | Initialize and coordinate all modules; drive the main event loop; aggregate and output results |
| **Input** | `config/settings.yaml` · camera frames · speech callback events |
| **Output** | OpenCV video window (OSD overlay) · console print · `logs/*.log` · `data/sessions/*.json` |
| **Key Methods** | `run()` — main loop · `_on_speech_recognized(text)` — speech callback · `_shutdown()` — graceful teardown |

---

### `src/utils/__init__.py` — Foundation Layer

#### `load_config(config_path) → dict`

| Item | Detail |
|---|---|
| **Purpose** | Parse `settings.yaml` into a Python dict for all modules |
| **Input** | YAML file path (default: `config/settings.yaml`) |
| **Output** | `dict` with keys: `camera`, `audio`, `vision_emotion`, `fusion`, `logging`, `output` |

#### `setup_logger(config)`

| Item | Detail |
|---|---|
| **Purpose** | Initialize loguru with console + rotating file handlers |
| **Input** | Config dict |
| **Output** | Loguru logger active; log files in `logs/emotion_YYYY-MM-DD.log` (10 MB rotation, 7-day retention, UTF-8) |

#### `SessionRecorder`

| Item | Detail |
|---|---|
| **Purpose** | Persist structured emotion data per session for downstream analysis (LLM / EEG / longitudinal) |
| **Input** | `fused_result` dict + optional `vision_result`, `audio_result`, `speech_text` per cycle |
| **Output** | `data/sessions/session_YYYYMMDD_HHMMSS.json` — flushed every 10 records, atomic write |
| **Record Format** | `{"timestamp", "fused": {emotion, confidence, all_scores}, "vision": {...}, "audio": {...}, "speech_text": "..."}` |
| **Why important** | Phase 3 LLM reads emotion history; Phase 4 EEG aligns by timestamp; speech_text preserves user language |

---

### `src/vision/__init__.py` — Vision Emotion Analysis

#### `CameraCapture`

| Item | Detail |
|---|---|
| **Purpose** | Continuously capture frames from webcam in a background thread |
| **Input** | `device_id`, `width`, `height`, `fps` from config |
| **Output** | `numpy.ndarray` BGR frame via `get_frame()` (thread-safe, always latest) |
| **Key design** | Background thread + `threading.Lock`; `get_frame()` non-blocking copy |

#### `VisionEmotionAnalyzer`

| Item | Detail |
|---|---|
| **Purpose** | Detect faces and classify emotions from a single video frame |
| **Input** | BGR `numpy.ndarray` frame |
| **Output** | `{"emotion_en": str, "confidence": float, "all_scores": dict, "box": dict, "detector": str}` or `None` |
| **Backend** | DeepFace (VGG-Face model, ssd detector) — lazy-loaded on first call |
| **Fallback chain** | `ssd` → `opencv` → `skip` → `haar+skip` (ROI crop on Haar detection) → `None` |
| **Confidence threshold** | 0.15 (relaxed); `skip` mode uses 0.08 |

#### `draw_result(frame, result) → frame`

| Item | Detail |
|---|---|
| **Purpose** | Overlay face box, emotion label, score bars on video frame |
| **Input** | BGR frame + vision result dict |
| **Output** | BGR frame with English-only OSD annotations |

---

### `src/audio/__init__.py` — Audio Processing

#### `SpeechRecognizer`

| Item | Detail |
|---|---|
| **Purpose** | Continuously listen to microphone and convert speech to text |
| **Input** | Microphone PCM audio (16 kHz, mono) |
| **Output** | Recognized text string via `on_result(text)` callback |
| **Primary engine** | `faster-whisper tiny` (local, offline, ~75 MB, supports Chinese) |
| **Fallback engine** | Google Speech-to-Text API (requires internet) |
| **Key design** | Whisper model loaded in background thread (`threading.Event`); every stage prints progress |

#### `TextToSpeech`

| Item | Detail |
|---|---|
| **Purpose** | Convert English text to speech playback |
| **Input** | English string only (Windows SAPI cannot pronounce Chinese correctly) |
| **Output** | Speaker audio (no return value) |
| **Engine** | pyttsx3 → Windows SAPI (local, no internet) |
| **API** | `speak(text)` blocking · `speak_async(text)` non-blocking |

#### `AudioEmotionAnalyzer`

| Item | Detail |
|---|---|
| **Purpose** | Infer emotion from recognized speech text using keyword matching |
| **Input** | Text string (Chinese or English) |
| **Output** | `{"emotion_en": str, "confidence": float, "text": str}` |
| **Algorithm** | Bilingual keyword scoring: positive words +1, negative words −1; score → happy / sad / neutral |
| **Upgrade path** | Replace with HuggingFace sentiment model (Token already configured in `.env`) |

---

### `src/fusion/__init__.py` — Multimodal Fusion

#### `EmotionFusion`

| Item | Detail |
|---|---|
| **Purpose** | Combine vision + audio emotion signals; smooth temporal fluctuation |
| **Input** | `vision_result` dict + `audio_result` dict (either may be `None`) |
| **Output** | `{"emotion_en": str, "confidence": float, "all_scores": dict, "sources": {vision, audio}}` |
| **Step 1** | Map each result to a 7-dim emotion vector (confidence-weighted) |
| **Step 2** | Dynamic weighted sum: both → vision 60% + audio 40%; one only → 100%; neither → empty |
| **Step 3** | `deque(maxlen=5)` sliding window mean → stable output across frames |
| **Extension** | `register_plugin(plugin)` hook for adding observers (e.g., LLM trigger) |

---

### `src/llm/__init__.py` — LLM Therapy Assistant *(Phase 3, Placeholder)*

| Item | Detail |
|---|---|
| **Purpose** | Generate personalized emotional therapy responses from a user's emotion history |
| **Input** | `person_id: str` + `emotion_history: list[dict]` (from SessionRecorder) |
| **Output** | English therapy response string (for TTS playback) |
| **Status** | Interface defined; implementation pending. Planned: OpenAI / Anthropic / Ollama |
| **Env vars needed** | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (add to `.env`) |

---

### `src/eeg/__init__.py` — EEG Signal Analysis *(Phase 4, Placeholder)*

| Item | Detail |
|---|---|
| **Purpose** | Stream EEG signals, extract emotional features, feed as 3rd modality into EmotionFusion |
| **Input** | EEG device serial stream (OpenBCI / Emotiv / MindLink) |
| **Output** | `{"emotion": str, "confidence": float, "bands": {"alpha", "beta", "theta"}}` |
| **Secondary use** | Ground-truth validation of LLM therapy effectiveness (compare EEG state before/after) |
| **Status** | Interface defined; implementation pending. Env var: `EEG_DEVICE_PORT` |

---

## Configuration Reference (`config/settings.yaml`)

```yaml
camera:
  device_id: 0              # Webcam ID (0 = default)
  width: 640
  height: 480
  fps: 30
  window_name: "Emotion Analysis - Live"

audio:
  language: "zh-CN"         # Speech recognition language
  record_seconds: 5         # Max speech segment length

vision_emotion:
  backend: "deepface"
  model_name: "VGG-Face"
  detector_backend: "ssd"   # MUST be ssd (opencv too inaccurate)
  confidence_threshold: 0.15

fusion:
  vision_weight: 0.6
  audio_weight: 0.4
  smoothing_window: 5

logging:
  level: "INFO"
  log_dir: "logs"
  max_size: "10 MB"
  retention: "7 days"
```

---

## Environment Variables (`.env`)

```
HF_TOKEN=hf_...                  # HuggingFace model download
HUGGINGFACE_TOKEN=hf_...         # Same — legacy name for older libs
OPENAI_API_KEY=sk-...            # Phase 3: OpenAI LLM
ANTHROPIC_API_KEY=sk-ant-...     # Phase 3: Claude LLM (alternative)
EEG_DEVICE_PORT=COM3             # Phase 4: EEG serial port
```

Copy `.env.example` to `.env` and fill in your values. **Never commit `.env`.**

---

## Installation

```bash
git clone https://github.com/zehantan6970/emotion.git
cd emotion
conda create -n emotion python=3.10 -y
conda activate emotion
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your tokens
python main.py
```

---

## Known Issues & Solutions

| # | Symptom | Root Cause | Solution Applied |
|---|---|---|---|
| 1 | Chinese garbled in video window or terminal | OpenCV has no CJK font; Windows terminal is GBK | **All UI/system text forced to English** |
| 2 | "No face detected" with visible face | `opencv` detector inaccurate; threshold 0.3 too high | `ssd` detector + threshold `0.15` + 4-level fallback |
| 3 | Speech recognized but nothing printed | Only `logger.info`; errors silently swallowed | `print` added at every stage of `_listen_loop` |
| 4 | ESC key does not quit | `waitKey` only checked `q` (missed ASCII 27) | `key == 27 or key == ord("q")` |
| 5 | TTS garbled / wrong pronunciation | Windows SAPI English voice cannot speak Chinese | TTS input fixed to English sentences only |
| 6 | Slow startup (UI blocked) | Whisper model load on main thread | Background thread + `threading.Event` ready signal |
| 7 | Emotion flickers every frame | Single-frame analysis is noisy | `deque(maxlen=5)` sliding window mean |
| 8 | `UnicodeEncodeError` on Windows | stdout defaults to GBK | Rewrap `sys.stdout` as UTF-8 at startup |

---

## Speech Interaction Behavior

```
────────────────────────────────────────
  [Speech] You said: 今天心情不错           ← user's original language preserved
  [Audio Emotion] happy  Confidence: 60%   ← always English label
  [System Reply] Great to see you are in a good mood!   ← always English
────────────────────────────────────────
```

- `You said:` → **preserves user language** (Chinese in → Chinese shown; English in → English shown)
- All other labels, system messages, TTS output → **fixed English**

---

## Data Outputs

### 1. Runtime Logs — `logs/emotion_YYYY-MM-DD.log`

- Format: `YYYY-MM-DD HH:mm:ss | LEVEL | module:function:line | message`
- Encoding: UTF-8
- Rotation: 10 MB per file; retained 7 days
- Content: module init, detection events, errors, session open/close
- **Git status:** excluded (local only)

### 2. Session Records — `data/sessions/session_YYYYMMDD_HHMMSS.json`

- One file per run of `main.py`
- Flushed every 10 records; atomic write (`.tmp` rename)
- **Git status:** excluded (can be large); directory tracked via `.gitkeep`
- **Used by:** Phase 3 LLM (emotion history), Phase 4 EEG (timestamp alignment)

**Example record:**
```json
{
  "session_id": "20260315_173200",
  "start_time": "2026-03-15T17:32:00.123",
  "end_time": "2026-03-15T17:45:12.456",
  "total_records": 720,
  "records": [
    {
      "timestamp": "2026-03-15T17:32:10.789",
      "fused":  { "emotion": "happy", "confidence": 0.73, "all_scores": { "happy": 0.73, "neutral": 0.18, ... } },
      "vision": { "emotion": "happy", "confidence": 0.87 },
      "audio":  { "emotion": "neutral", "confidence": 0.50 },
      "speech_text": "今天天气不错"
    }
  ]
}
```

---

## Development Roadmap

| Phase | Feature | Status |
|---|---|---|
| **v1.0** | Camera + Speech + Multimodal Fusion + TTS | ✅ Complete |
| **Phase 2** | Multi-person face tracking (face ID per person) | ⏳ Next |
| **Phase 3** | LLM emotional therapy assistant (`src/llm/`) | ⏳ Planned |
| **Phase 4** | EEG signal integration + therapy validation (`src/eeg/`) | ⏳ Planned |

---

## Git Workflow

```bash
# New feature
git checkout -b feature/multi-person-tracking
git add .
git commit -m "feat(vision): add multi-person face tracking with ID assignment"
git push origin feature/multi-person-tracking
# → open Pull Request on GitHub → merge to main
```

**Branch naming:** `feature/xxx` · `fix/xxx` · `exp/xxx` (experiments)

**Commit format:** `feat(module): description` · `fix(module): description` · `exp(eeg): description`

---

## AI Assistant — Context Resume

> Paste this block at the start of a new AI chat to instantly restore project context.

```
I am continuing development of a multimodal real-time emotion analysis project.

Project path : D:\pycode\emotion
GitHub       : https://github.com/zehantan6970/emotion
Environment  : Windows 11, conda env "emotion", Python 3.10, D:\miniconda3
Stack        : DeepFace(VGG-Face/ssd) + faster-whisper + pyttsx3 + SpeechRecognition + loguru
Version      : v1.0 complete — camera vision + speech + multimodal fusion working

Key constraints:
- ALL UI / logger / print must be English (OpenCV no CJK; Windows terminal GBK)
- Speech recognized text printed AS-IS (Chinese stays Chinese, English stays English)
- TTS input must be English only (Windows SAPI cannot pronounce Chinese)
- Face detector: ssd, confidence_threshold: 0.15, 4-level fallback
- Tokens in .env (gitignored) — HF_TOKEN already configured

Session data saved to: data/sessions/session_*.json (each run)
Runtime logs in      : logs/emotion_YYYY-MM-DD.log

Next task: [describe what you want to do]

Please read README.md and relevant src/[module]/__init__.py before writing any code.
```

---

## License

MIT
