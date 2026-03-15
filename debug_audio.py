# -*- coding: utf-8 -*-
"""
语音识别调试脚本 - 第二步：只调试麦克风 + faster-whisper 本地识别
无需网络，全程离线运行
"""
import os
import sys
import io
import tempfile
import time

# 解决 Windows GBK 终端
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  语音识别调试脚本（faster-whisper 本地离线）")
print("  说话后等待识别结果，按 Ctrl+C 退出")
print("=" * 60)

# ── 1. 检查 SpeechRecognition ──────────────────────────────────
try:
    import speech_recognition as sr
    print("[OK] SpeechRecognition 可用")
except ImportError:
    print("[错误] speech_recognition 未安装: pip install SpeechRecognition")
    sys.exit(1)

# ── 2. 检查并加载 faster-whisper ──────────────────────────────
print("\n[加载] 正在加载 faster-whisper tiny 模型（首次需下载约75MB）...")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("[OK] faster-whisper 模型加载成功")
except Exception as e:
    print(f"[错误] faster-whisper 加载失败: {e}")
    sys.exit(1)

# ── 3. 检查麦克风 ──────────────────────────────────────────────
print("\n[检查] 可用麦克风列表:")
mics = sr.Microphone.list_microphone_names()
for i, name in enumerate(mics):
    print(f"  [{i}] {name}")

# ── 4. 初始化识别器 ────────────────────────────────────────────
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

print("\n[校准] 正在校准环境噪音，请保持安静...")
try:
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print(f"[OK] 噪音校准完成，能量阈值: {recognizer.energy_threshold:.0f}")
        print(f"\n[就绪] 开始监听，请用中文说话（每次说完后稍作停顿）\n")

        count = 0
        while True:
            try:
                print(f"[第{count+1}次] 等待说话...")
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
                print(f"[第{count+1}次] 识别中...")
                t0 = time.time()

                # 写临时 WAV 文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp = f.name
                    f.write(audio.get_wav_data())

                # faster-whisper 识别
                segments, info = model.transcribe(
                    tmp, language="zh", beam_size=3,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                )
                text = "".join(seg.text for seg in segments).strip()
                elapsed = time.time() - t0
                os.unlink(tmp)

                count += 1
                if text:
                    print(f"\n{'='*50}")
                    print(f"  识别结果: {text}")
                    print(f"  耗时: {elapsed:.2f}s")
                    print(f"{'='*50}\n")
                else:
                    print(f"[第{count}次] 未识别到内容（可能是静音或噪音）\n")

            except sr.WaitTimeoutError:
                print("[超时] 未检测到说话，继续等待...\n")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[异常] {e}\n")

except KeyboardInterrupt:
    print("\n[退出] 语音调试结束")
