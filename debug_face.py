# -*- coding: utf-8 -*-
"""
人脸检测调试脚本 - 第一步：只调试摄像头 + 人脸检测
逐步尝试多个检测后端，显示哪个能成功检测到人脸
按 D 键切换检测方式
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # 抑制 oneDNN 日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # 抑制 TF 日志

import sys
import io
# 解决 Windows GBK 终端输出 emoji 报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import time

print("=" * 60)
print("  人脸检测调试脚本")
print("  按 ESC 或 Q 退出，按 D 切换检测方式")
print("=" * 60)

# ── 1. 打开摄像头 ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[错误] 无法打开摄像头，请检查摄像头连接")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"[OK] 摄像头已打开，分辨率: {int(cap.get(3))}x{int(cap.get(4))}")

# ── 2. 检查可用的检测后端 ──────────────────────────────────────
print("\n[检查] 正在检查可用的人脸检测方案...")

# 方案A：OpenCV Haar Cascade（内置，最稳定）
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(cascade_path)
print(f"[OK] OpenCV Haar Cascade 已加载")

# 方案B：DeepFace
deepface_available = False
try:
    import warnings
    warnings.filterwarnings("ignore")
    from deepface import DeepFace
    deepface_available = True
    print("[OK] DeepFace 可用")
except Exception as e:
    print(f"[跳过] DeepFace 不可用: {str(e)[:60]}")

# 方案C：FER
fer_available = False
try:
    from fer import FER
    fer_detector = FER(mtcnn=False)
    fer_available = True
    print("[OK] FER 可用")
except ImportError:
    print("[跳过] FER 未安装")

print()

# ── 3. DeepFace 预热 ───────────────────────────────────────────
if deepface_available:
    print("[预热] DeepFace 首次加载模型，请稍候（约10-30秒）...")
    try:
        import numpy as np
        import contextlib
        ret, frame = cap.read()
        if ret:
            with contextlib.redirect_stderr(open(os.devnull, "w")):
                DeepFace.analyze(frame, actions=["emotion"],
                                 detector_backend="skip",
                                 enforce_detection=False, silent=True)
        print("[OK] DeepFace 模型预热完成")
    except Exception as e:
        print(f"[警告] DeepFace 预热异常: {str(e)[:80]}")

# ── 4. 检测方式列表（按优先级） ────────────────────────────────
modes = []
modes.append("OpenCV-Haar")
if deepface_available:
    modes += ["DeepFace/ssd", "DeepFace/opencv", "DeepFace/skip"]
if fer_available:
    modes.append("FER")

current_mode = 0
print(f"\n[模式] 当前检测方式: {modes[current_mode]}  (按 D 切换)")
print("[运行] 实时检测中...\n")

# ── 5. 主循环 ──────────────────────────────────────────────────
last_detect_time = 0
detect_interval = 0.4   # 每0.4秒做一次分析
last_result_text = "等待检测..."
last_face_box = None    # 保留上一帧的人脸框
face_count = 0
total_detect = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    now = time.time()

    # ── 按间隔执行检测 ──────────────────────────────────────────
    if now - last_detect_time >= detect_interval:
        last_detect_time = now
        total_detect += 1
        mode = modes[current_mode]

        try:
            # ── Haar Cascade ──────────────────────────────────
            if mode == "OpenCV-Haar":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)   # 直方图均衡，提升暗光检测
                faces = haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(50, 50),
                )
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    last_face_box = ("rect", x, y, w, h)
                    last_result_text = f"Haar 检测到人脸 ({w}x{h}px)"
                    face_count += 1
                    print(f"[Haar] 检测到人脸  坐标:({x},{y}) 大小:{w}x{h}  总计:{face_count}次")
                else:
                    last_result_text = "Haar: 未检测到人脸"
                    last_face_box = None

            # ── DeepFace ──────────────────────────────────────
            elif mode.startswith("DeepFace") and deepface_available:
                backend = mode.split("/")[1]
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend=backend,
                    enforce_detection=False,
                    silent=True,
                )
                r = results[0] if isinstance(results, list) else results
                dominant = r.get("dominant_emotion", "")
                emotions = r.get("emotion", {})
                region = r.get("region", {})
                rw = region.get("w", 0)
                rh = region.get("h", 0)
                conf = emotions.get(dominant, 0) / 100.0 if dominant else 0

                if dominant and rw > 20 and rh > 20:
                    rx, ry = region.get("x", 0), region.get("y", 0)
                    last_face_box = ("rect", rx, ry, rw, rh)
                    last_result_text = f"{mode}: {dominant} ({conf:.0%})"
                    face_count += 1
                    print(f"[{mode}] 检测到人脸: {dominant} {conf:.0%}  区域:({rx},{ry},{rw},{rh})  总计:{face_count}次")
                elif dominant:
                    last_result_text = f"{mode}: {dominant} ({conf:.0%}) [无定位框]"
                    face_count += 1
                    last_face_box = None
                    print(f"[{mode}] 情感(无框): {dominant} {conf:.0%}  总计:{face_count}次")
                else:
                    last_result_text = f"{mode}: 未检测到人脸"
                    last_face_box = None

            # ── FER ───────────────────────────────────────────
            elif mode == "FER" and fer_available:
                result = fer_detector.detect_emotions(frame)
                if result:
                    emotions = result[0]["emotions"]
                    top = max(emotions, key=emotions.get)
                    conf = emotions[top]
                    box = result[0].get("box", [0, 0, 0, 0])
                    x, y, w, h = box
                    last_face_box = ("rect", x, y, w, h)
                    last_result_text = f"FER: {top} ({conf:.0%})"
                    face_count += 1
                    print(f"[FER] 检测到人脸: {top} {conf:.0%}  总计:{face_count}次")
                else:
                    last_result_text = "FER: 未检测到人脸"
                    last_face_box = None

        except Exception as e:
            last_result_text = f"异常: {str(e)[:50]}"
            print(f"[异常] {mode}: {e}")

    # ── 绘制人脸框 ──────────────────────────────────────────────
    if last_face_box:
        kind, x, y, w, h = last_face_box
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ── 在画面上显示信息 ────────────────────────────────────────
    fh = display.shape[0]
    # 顶部黑底信息栏
    cv2.rectangle(display, (0, 0), (640, 55), (0, 0, 0), -1)
    cv2.putText(display, f"Mode [{current_mode+1}/{len(modes)}]: {modes[current_mode]}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    is_detected = ("检测到" in last_result_text or
                   ("%" in last_result_text and "未" not in last_result_text and "无" not in last_result_text))
    color = (0, 255, 0) if is_detected else (0, 80, 255)
    cv2.putText(display, last_result_text, (8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # 底部操作提示
    cv2.rectangle(display, (0, fh-28), (640, fh), (0, 0, 0), -1)
    cv2.putText(display,
                f"D:切换方式  ESC/Q:退出  检测:{face_count}/{total_detect}",
                (8, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Face Detection Debug", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        print("\n[退出] 用户退出")
        break
    elif key == ord("d") or key == ord("D"):
        current_mode = (current_mode + 1) % len(modes)
        last_result_text = "切换中..."
        last_face_box = None
        print(f"\n[切换] -> {modes[current_mode]}")

cap.release()
cv2.destroyAllWindows()
print(f"\n[统计] 共检测 {total_detect} 次，成功 {face_count} 次")
print("[完成] 调试结束")
