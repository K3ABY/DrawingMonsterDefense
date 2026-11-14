# Drawing Monster Defense – Phase 1: Hand Tracking & Air Drawing

ระบบตรวจจับมือและวาดภาพกลางอากาศ (Air Drawing) แบบเรียลไทม์ พัฒนาด้วย Python, OpenCV และ MediaPipe Hands สำหรับเกม **Drawing Monster Defense**

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **Dual-Zone Air Drawing System:**
   - แบ่ง Canvas แนวตั้งกึ่งกลาง 50/50
   - **มือซ้าย (Left Hand):** วาดเฉพาะครึ่งซ้าย ($X < 50\%$) เท่านั้น
   - **มือขวา (Right Hand):** วาดเฉพาะครึ่งขวา ($X \ge 50\%$) เท่านั้น
   - ระบบป้องกันการวาดข้ามโซน (Boundary Enforcement) และตัด Stroke ทันทีเมื่อมือออกนอกโซน

2. **3D Index Finger Recognition:**
   - ตรวจจับท่าชี้นิ้วชี้แบบ 3 มิติ (Collinear 3D chord-to-segment ratio & PIP angle)
   - จำแนกทิศทางการชี้ 3D อัตโนมัติ: `FORWARD` (ชี้เข้าหากล้องตามแกน Z), `UP`, `DOWN`, `SIDEWAYS`, `DIAGONAL`
   - ปฏิเสธท่า Open Palm และท่ากำมือ (Fist) อย่างแม่นยำ
   - ทนทานต่อการงอนิ้วตามธรรมชาติ (Relaxed finger curvature)

3. **Two-Hand Join Gesture (Toggle Mode):**
   - ประสานปลายนิ้วสองมือค้างไว้ $\approx 0.7$ วินาที เพื่อสลับเปิด/ปิดโหมดวาดรูป (`IDLE` $\leftrightarrow$ `DRAWING_ACTIVE`)
   - ระบบ Debouncing & Release Detection ป้องกันการสลับสถานะซ้ำซ้อน

4. **Independent Hand Control (Fist Pause):**
   - กำมือข้างใดข้างหนึ่งเพื่อหยุดวาดชั่วคราวสำหรับมือนั้น โดยที่มืออีกข้างยังคงวาดต่อได้อย่างอิสระ
   - คลายมือกำเพื่อกลับมาวาดต่อได้อย่างต่อเนื่อง

5. **High Stability & Low Latency:**
   - **ThreadedCameraStream:** สตรีมภาพแบบ Background Thread เพื่อ Ingest เฉพาะเฟรมล่าสุด ป้องกัน Frame Buffer Backlog (Latency < 0.2 ms)
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
แก้ไขไฟล์ `camera_config.py` ตามความเหมาะสม:
- `CAMERA_URL`: URL สตรีมกล้อง (เช่น `http://192.168.1.172:5000/video` หรือ `0` สำหรับเว็บแคมเครื่องโดยตรง)
- `SHOW_DEBUG_HUD`: ตั้งค่า `True` หรือ `False` สำหรับแสดงผลข้อมูล Debug HUD บนหน้าจอ

### การเริ่มรันโปรแกรม (Run)
```bash
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

## 🧪 การรันชุดทดสอบ (Testing & Verification)

โปรเจกต์มีชุด Acceptance Tests อัตโนมัติที่ผ่านการรับรอง 100%:

```bash
# รันชุดทดสอบหลัก 12 สถานการณ์ (3D Pointing, Latency, Loss Recovery, Boundary, Gesture)
python test_hand_tracking_improvement.py

# รันชุดทดสอบ Dual-Zone
python test_dual_zone_drawing.py

# รันชุดทดสอบ Two-Hand Join Toggle Gesture
python test_gesture_toggle.py

# รันชุดทดสอบ One Euro Filter & Stability
python test_stability_and_drawing.py
```

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)
```text
DrawingMonsterDefense/
├── camera_config.py                   # ค่าคอนฟิกกล้อง, MediaPipe และเกณฑ์การตรวจจับ
├── drawing_manager.py                 # ระบบจัดการ Dual-Zone Drawing และ 3D Pose Recognition
├── hand_tracking.py                   # ตัวประมวลผลหลัก MediaPipe Hand Tracking & Visual HUD
├── gesture_controller.py              # Two-Hand Join Gesture State Machine (Toggle Mode)
├── one_euro_filter.py                 # Adaptive One Euro Filter สำหรับลด Jitter
├── threaded_camera.py                 # Zero-Latency Frame Ingestion Streamer
├── requirements.txt                   # รายการไลบรารีที่จำเป็น
├── test_hand_tracking_improvement.py  # 12 Acceptance Test Scenarios
├── test_dual_zone_drawing.py          # Dual-Zone Acceptance Tests
├── test_gesture_toggle.py             # Gesture State Machine Tests
└── test_stability_and_drawing.py      # Jitter & Continuity Tests
```
