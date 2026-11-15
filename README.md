# Drawing Monster Defense – Hand Tracking & Air Drawing System

ระบบตรวจจับมือและวาดภาพกลางอากาศ (Air Drawing) แบบเรียลไทม์ พร้อมสถาปัตยกรรม Python Package ที่เป็นมาตรฐาน รองรับ Adaptive Hand Tracking, Automatic Camera Detection & Fallback, Dual-Zone Air Drawing, และโครงสร้างพร้อมสำหรับ Continuous Local Learning

---

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **Automatic Camera Detection & Fallback:**
   - ค้นหาและทดสอบ Local Webcam อัตโนมัติ (เปิดได้จริง + ตรวจสอบเฟรมไม่ว่างเปล่า)
   - สลับไปใช้ MJPEG Network IP Camera อัตโนมัติทันทีหากไม่มีกล้อง Local หรือกล้องดรอป
   - รองรับ Hot-swapping และ Auto-reconnect พร้อม Standby Frame เมื่อกล้องหลุด ไม่แครช

2. **Dual-Zone Air Drawing System:**
   - แบ่ง Canvas แนวตั้งกึ่งกลาง 50/50
   - **มือซ้าย (Left Hand):** วาดเฉพาะครึ่งซ้าย ($X < 50\%$) เท่านั้น
   - **มือขวา (Right Hand):** วาดเฉพาะครึ่งขวา ($X \ge 50\%$) เท่านั้น
   - ระบบ Boundary Enforcement และตัด Stroke ทันทีเมื่อมือออกนอกโซน

3. **3D Index Finger Recognition:**
   - ตรวจจับท่าชี้นิ้วชี้แบบ 3 มิติ (Collinear 3D chord-to-segment ratio & PIP angle)
   - จำแนกทิศทางการชี้ 3D อัตโนมัติ: `FORWARD` (ชี้เข้าหากล้องตามแกน Z), `UP`, `DOWN`, `SIDEWAYS`, `DIAGONAL`
   - ปฏิเสธท่า Open Palm และท่ากำมือ (Fist) อย่างแม่นยำ
   - ทนทานต่อการงอนิ้วตามธรรมชาติ (Relaxed finger curvature)

4. **Two-Hand Join Gesture (Toggle Mode):**
   - ประสานปลายนิ้วสองมือค้างไว้ $\approx 0.7$ วินาที เพื่อสลับเปิด/ปิดโหมดวาดรูป (`IDLE` $\leftrightarrow$ `DRAWING_ACTIVE`)
   - ระบบ Debouncing & Release Detection ป้องกันการสลับสถานะซ้ำซ้อน

5. **Independent Hand Control (Fist Pause):**
   - กำมือข้างใดข้างหนึ่งเพื่อหยุดวาดชั่วคราวสำหรับมือนั้น โดยที่มืออีกข้างยังคงวาดต่อได้อย่างอิสระ
   - คลายมือกำเพื่อกลับมาวาดต่อได้อย่างต่อเนื่อง

6. **Adaptive Image Enhancement & Quality Assessment (IQA):**
   - ตรวจสอบคุณภาพภาพแบบเบาเครื่อง (Lightweight IQA: Brightness, Contrast, Blur, Noise)
   - ปรับแต่งภาพเฉพาะเมื่อจำเป็น (Gamma, CLAHE, Denoise) เพื่อไม่กินทรัพยากร CPU

7. **High Stability & Low Latency:**
   - **ThreadedCameraStream:** สตรีมภาพแบบ Background Thread Ingestion (Latency < 0.2 ms, Zero Backlog)
   - **Adaptive One Euro Filter:** กรอง Jitter ปลายนิ้วขณะอยู่นิ่ง และลด Latency เหลือต่ำกว่า 8 px ขณะสะบัดมือวาดเร็ว
   - **Gap Bridging Recovery:** เชื่อมต่อเส้นวาดต่อเนื่องกรณีเฟรมดรอปชั่วขณะ ($\le 0.22$s) และตัด Stroke ทันทีเมื่อ Tracking หลุดนาน

---

## 🚀 การติดตั้งและใช้งาน (Getting Started)

### ความต้องการของระบบ (Requirements)
- Python 3.10 – 3.12
- เว็บแคมโน้ตบุ๊ก หรือ Camera Server สตรีมผ่าน MJPEG Wi-Fi

### การติดตั้ง (Installation)
```bash
# ติดตั้ง dependencies
   pip install -r requirements.txt
```

### การตั้งค่ากล้อง (Configuration)
แก้ไขไฟล์ `src/config/camera_config.py` (หรือ `camera_config.py`):
- `CAMERA_URL`: URL สตรีมกล้อง IP Camera (เช่น `http://192.168.1.172:5000/video`)
- `SHOW_DEBUG_HUD`: ตั้งค่า `True` หรือ `False` สำหรับแสดงผลข้อมูล Debug HUD บนหน้าจอ

### การเริ่มรันโปรแกรม (Run)
สามารถรันได้ทั้งสองแบบ (รองรับทั้ง Modern Package และ Root Backward-Compatibility):
```bash
# รันผ่าน Entry Point หลัก
python main.py

# หรือรันแบบ Python Module
python -m src.main

# หรือรันผ่าน Backward-Compatibility Shim
python hand_tracking.py
```

### คีย์ลัดบนคีย์บอร์ด (Keyboard Controls)
- `[Q]` / `[ESC]`: ออกจากโปรแกรม
- `[C]`: ล้างเส้นวาดบน Canvas
- `[D]`: เปิด/ปิด (Toggle) แถบข้อมูล Debug HUD
- `[S]`: เปิด/ปิด ตัวกรอง One Euro Filter Smoothing
- `[H]`: สลับ Handedness Inversion (ซ้าย/ขวา)
- `[F]`: สลับการกลับภาพแนวนอน (Horizontal Mirroring)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
DrawingMonsterDefense/
├── src/                               # ซอร์สโค้ดหลักของโปรเจกต์ (Modular Python Package)
│   ├── __init__.py
│   ├── main.py                        # Application Core Entry Point
│   ├── camera/                        # ระบบเชื่อมต่อและสตรีมกล้อง (Webcam & IP)
│   │   ├── __init__.py
│   │   └── threaded_camera.py
│   ├── tracking/                      # Hand Tracking & MediaPipe Pipeline
│   │   ├── __init__.py
│   │   └── hand_tracker.py
│   ├── drawing/                       # Dual-Zone Air Drawing & Stroke Management
│   │   ├── __init__.py
│   │   └── drawing_manager.py
│   ├── gestures/                      # Gesture State Machine (Two-Hand Join, Fist)
│   │   ├── __init__.py
│   │   └── gesture_controller.py
│   ├── vision/                        # Image Quality Assessor & Adaptive Enhancer
│   │   ├── __init__.py
│   │   ├── image_quality_assessor.py
│   │   └── adaptive_enhancer.py
│   ├── filters/                       # Adaptive Smoothing Filters (One Euro Filter)
│   │   ├── __init__.py
│   │   └── one_euro_filter.py
│   ├── config/                        # การตั้งค่าระบบ กล้อง และพารามิเตอร์
│   │   ├── __init__.py
│   │   └── camera_config.py
│   └── learning/                      # โครงสร้างเตรียมพร้อมสำหรับ Continuous Local Learning
│       └── __init__.py
├── tests/                             # ชุดทดสอบครอบคลุมทุกระบบ (Unit & Acceptance Tests)
│   ├── camera/                        # ทดสอบ Camera Detection & Robustness
│   │   ├── test_camera_detection.py
│   │   ├── test_camera_robustness.py
│   │   └── network_camera_test.py
│   ├── tracking/                      # ทดสอบ Hand Tracker & 3D Pointing
│   │   ├── test_hand_tracker.py
│   │   └── test_hand_tracking_improvement.py
│   ├── drawing/                       # ทดสอบ Dual-Zone 50/50 Canvas
│   │   └── test_dual_zone_drawing.py
│   ├── gestures/                      # ทดสอบ Gesture State Machine
│   │   └── test_gesture_toggle.py
│   └── stability/                     # ทดสอบ One Euro Filter & Latency
│       └── test_stability_and_drawing.py
├── benchmarks/                        # สคริปต์วัดประสิทธิภาพและทรัพยากร (FPS, CPU, RAM)
│   ├── benchmark_system_baseline.py
│   └── benchmark_after.py
├── diagnostics/                       # สคริปต์ตรวจสอบการทำงานสตรีมกล้องและวิเคราะห์ Jitter
│   ├── diagnose_performance.py
│   └── verify_live_detection.py
├── data/                              # พื้นที่จัดเก็บชุดข้อมูลและโมเดลสำหรับ Local Learning
│   ├── learning_dataset/
│   ├── models/
│   └── profiles/
├── assets/                            # เอาต์พุตภาพจากการทดสอบและไฟล์สื่อ
│   └── test_outputs/
├── backups/                           # พื้นที่สำรองข้อมูลโปรเจกต์
├── main.py                            # Root Entry Point
├── camera_config.py                   # Root Shim (Backward-Compatibility)
├── threaded_camera.py                 # Root Shim (Backward-Compatibility)
├── drawing_manager.py                 # Root Shim (Backward-Compatibility)
├── gesture_controller.py              # Root Shim (Backward-Compatibility)
├── hand_tracking.py                   # Root Shim (Backward-Compatibility)
├── image_quality_assessor.py          # Root Shim (Backward-Compatibility)
├── adaptive_enhancer.py               # Root Shim (Backward-Compatibility)
├── one_euro_filter.py                 # Root Shim (Backward-Compatibility)
├── requirements.txt                   # รายการไลบรารีที่จำเป็น
├── .gitignore
└── README.md
```

---

## 🧪 การรันชุดทดสอบ (Testing & Verification)

### รันการทดสอบทั้งหมดผ่าน unittest:
```bash
python -m unittest discover -s tests -t .
```

### หรือรันการทดสอบแยกตามโมดูล:
```bash
# 1. กล้อง: Automatic Detection & Robustness (18 tests)
python tests/camera/test_camera_detection.py
python tests/camera/test_camera_robustness.py

# 2. Tracking: 3D Pointing & Hand Tracking (13 tests)
python tests/tracking/test_hand_tracker.py
python tests/tracking/test_hand_tracking_improvement.py

# 3. Drawing: Dual-Zone 50/50 Canvas & Boundary (8 tests)
python tests/drawing/test_dual_zone_drawing.py

# 4. Gestures: Two-Hand Join Toggle & Fist (9 tests)
python tests/gestures/test_gesture_toggle.py

# 5. Stability: One Euro Filter & Gap Bridging (3 tests)
python tests/stability/test_stability_and_drawing.py
```

---

## 📊 การรัน Benchmark และ Diagnostics

```bash
# วัดประสิทธิภาพภาพรวม ระบบประมวลผล, FPS, CPU, และ RAM
python benchmarks/benchmark_system_baseline.py

# ตรวจสอบวิเคราะห์ความหน่วงและ Jitter ของสตรีมกล้อง
python diagnostics/diagnose_performance.py
```
