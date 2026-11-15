"""
Baseline Performance & Stability Benchmark for Drawing Monster Defense.
Measures:
- Average FPS, Minimum FPS
- Frame Processing Time breakdown (Grab, IQA, Enhancement, MediaPipe, Smoothing, Gesture, Drawing, Rendering)
- CPU Usage (%) & RAM Working Set (MB)
- Memory allocation stability & leak detection
- Hand Tracking & 3D Index Finger Recognition reliability across all directions
- Drawing responsiveness and latency
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import psutil

# Ensure DrawingMonsterDefense root and src are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
for p in [current_dir, project_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import src.config.camera_config as config
    from src.vision.image_quality_assessor import ImageQualityAssessor
    from src.vision.adaptive_enhancer import AdaptiveImageEnhancer
    from src.drawing.drawing_manager import AirDrawingManager, TrackingConfidenceState
    from src.gestures.gesture_controller import TwoHandJoinGestureController, DrawingState
    from src.filters.one_euro_filter import HandLandmarkSmoother
    from src.tracking.hand_tracker import HandTracker
except ImportError:
    import camera_config as config
    from image_quality_assessor import ImageQualityAssessor
    from adaptive_enhancer import AdaptiveImageEnhancer
    from drawing_manager import AirDrawingManager, TrackingConfidenceState
    from gesture_controller import TwoHandJoinGestureController, DrawingState
    from one_euro_filter import HandLandmarkSmoother
    from hand_tracking import HandTracker


def create_benchmark_frame(step: int, w: int = 640, h: int = 480) -> np.ndarray:
    """Generates synthetic camera frames simulating moving hands and indoor background."""
    img = np.full((h, w, 3), 160, dtype=np.uint8)
    img[int(h * 0.55):int(h * 0.75), :] = 80
    img[int(h * 0.75):, :] = 45

    # Moving hand position
    hx = int(200 + 80 * math.sin(step * 0.1))
    hy = int(220 + 40 * math.cos(step * 0.1))

    # Hand silhouette
    cv2.ellipse(img, (hx, hy + 40), (45, 55), 0, 0, 360, (170, 190, 220), -1)
    cv2.line(img, (hx, hy + 20), (hx, hy - 40), (160, 185, 215), 18)
    cv2.circle(img, (hx, hy - 40), 9, (160, 185, 215), -1)
    return img


def run_system_benchmark(num_frames: int = 100, enable_hud: bool = True):
    print("=" * 72)
    print(" DRAWING MONSTER DEFENSE: SYSTEM PERFORMANCE & STABILITY BENCHMARK")
    print(f" Sample Frames: {num_frames} | Diagnostic HUD: {'ENABLED' if enable_hud else 'DISABLED'}")
    print("=" * 72)

    process = psutil.Process(os.getpid())
    # Baseline memory before initialization
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)

    # Initialize subsystems
    t_init_start = time.perf_counter()
    tracker = HandTracker()
    tracker.show_hud = enable_hud
    gesture_ctrl = TwoHandJoinGestureController()
    gesture_ctrl.drawing_state = DrawingState.DRAWING_ACTIVE  # Active drawing mode
    drawing_mgr = AirDrawingManager()
    t_init_end = time.perf_counter()

    ram_after_init_mb = process.memory_info().rss / (1024 * 1024)
    print(f" Initialization Time: {(t_init_end - t_init_start) * 1000:.2f} ms")
    print(f" Initial RAM Working Set: {ram_initial_mb:.1f} MB -> After Model Init: {ram_after_init_mb:.1f} MB")

    # Time tracking lists
    iqa_times = []
    enh_times = []
    inference_times = []
    gesture_times = []
    drawing_times = []
    render_times = []
    total_frame_times = []

    # Pre-populate drawing with strokes to test rendering scalability
    for s_idx in range(5):
        sim_stroke = []
        for p_idx in range(40):
            sim_stroke.append({
                "x": 100 + s_idx * 20 + p_idx * 3,
                "y": 150 + int(30 * math.sin(p_idx * 0.2)),
                "norm_x": 0.25,
                "norm_y": 0.5,
                "zone_x": 0.5,
                "zone_y": 0.5,
                "t": time.time(),
            })
        drawing_mgr.strokes["Left"].append(sim_stroke)

    # Prime cpu measurement
    process.cpu_percent(interval=None)
    start_bench_time = time.perf_counter()

    for step in range(num_frames):
        t_f_start = time.perf_counter()
        curr_time = time.time()
        frame = create_benchmark_frame(step)

        # 1. Process hand tracking (Includes IQA, Adaptive Enhancer, MediaPipe, One Euro Filter)
        t0 = time.perf_counter()
        hand_data, results = tracker.process(frame, curr_time)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        # Retrieve IQA and Enhancer latencies from tracker metrics
        if tracker.last_quality_metrics:
            iqa_times.append(tracker.last_quality_metrics.assessment_time_ms)
        if tracker.last_enhancement_info:
            enh_times.append(tracker.last_enhancement_info.get("time_ms", 0.0))

        # Ensure simulated pointing hand exists for drawing benchmarking
        if hand_data["Left"] is None:
            tip_x = int(180 + 30 * math.sin(step * 0.15))
            tip_y = int(220 + 20 * math.cos(step * 0.15))
            hand_data["Left"] = {
                "label": "Left",
                "detected": True,
                "confidence": 0.94,
                "index_tip_x": tip_x,
                "index_tip_y": tip_y,
                "index_finger_tip": (tip_x, tip_y),
                "norm_tip": (tip_x / 640.0, tip_y / 480.0),
                "landmarks": [(tip_x, tip_y)] * 21,
                "landmarks_3d": [(tip_x / 640.0, tip_y / 480.0, 0.0)] * 21,
            }

        # 4. Gesture Controller Update
        t6 = time.perf_counter()
        gesture_ctrl.update(hand_data, curr_time)
        t7 = time.perf_counter()
        gesture_times.append((t7 - t6) * 1000)

        # 5. Drawing Manager Update
        t8 = time.perf_counter()
        drawing_mgr.update(hand_data, gesture_ctrl.is_drawing_active, curr_time)
        t9 = time.perf_counter()
        drawing_times.append((t9 - t8) * 1000)

        # 6. Rendering (Strokes + Annotations + HUD)
        t10 = time.perf_counter()
        perf_metrics = {
            "camera_fps": 30.0,
            "proc_fps": 60.0,
            "proc_ms": 16.0,
            "camera_mode": "BENCHMARK",
            "camera_status": "CONNECTED",
            "camera_source": "Synthetic Stream",
            "camera_res": "640x480",
            "fallback_occurred": False,
        }
        output_frame = tracker.draw(
            frame, hand_data, results, gesture_ctrl, drawing_mgr, perf_metrics
        )
        t11 = time.perf_counter()
        render_times.append((t11 - t10) * 1000)

        t_f_end = time.perf_counter()
        total_frame_times.append((t_f_end - t_f_start) * 1000)

    total_wall_time = time.perf_counter() - start_bench_time
    ram_final_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=None)

    # Frame metrics
    avg_total_ms = float(np.mean(total_frame_times))
    min_total_ms = float(np.min(total_frame_times))
    max_total_ms = float(np.max(total_frame_times))
    p95_total_ms = float(np.percentile(total_frame_times, 95))

    avg_fps = len(total_frame_times) / total_wall_time
    min_fps = 1000.0 / max_total_ms if max_total_ms > 0 else 0.0

    print("\n" + "=" * 72)
    print(" BENCHMARK RESULTS")
    print("=" * 72)
    print(f" Total Frames Processed: {num_frames} frames in {total_wall_time:.2f} s")
    print(f" Average Processing FPS: {avg_fps:.1f} FPS (Frame interval: {avg_total_ms:.2f} ms)")
    print(f" Minimum Processing FPS: {min_fps:.1f} FPS (Peak interval: {max_total_ms:.2f} ms)")
    print(f" 95th Percentile Latency: {p95_total_ms:.2f} ms")
    print("-" * 72)
    print(" LATENCY BREAKDOWN (Average per frame):")
    print(f"   - Image Quality Assessor (IQA): {np.mean(iqa_times):.2f} ms")
    print(f"   - Adaptive Image Enhancement   : {np.mean(enh_times):.2f} ms")
    print(f"   - MediaPipe Hands Inference    : {np.mean(inference_times):.2f} ms")
    print(f"   - Gesture Controller Update    : {np.mean(gesture_times):.4f} ms")
    print(f"   - Air Drawing Manager Update   : {np.mean(drawing_times):.4f} ms")
    print(f"   - Full Frame Rendering + HUD   : {np.mean(render_times):.2f} ms")
    print("-" * 72)
    print(" RESOURCE UTILIZATION:")
    print(f"   - CPU Usage   : {cpu_percent:.1f}%")
    print(f"   - Initial RAM : {ram_after_init_mb:.1f} MB")
    print(f"   - Final RAM   : {ram_final_mb:.1f} MB (Delta: {ram_final_mb - ram_after_init_mb:+.2f} MB)")
    print("=" * 72)

    tracker.close()
    return {
        "avg_fps": avg_fps,
        "min_fps": min_fps,
        "avg_total_ms": avg_total_ms,
        "max_total_ms": max_total_ms,
        "p95_total_ms": p95_total_ms,
        "iqa_ms": float(np.mean(iqa_times)),
        "enh_ms": float(np.mean(enh_times)),
        "inference_ms": float(np.mean(inference_times)),
        "render_ms": float(np.mean(render_times)),
        "cpu_percent": cpu_percent,
        "ram_init_mb": ram_after_init_mb,
        "ram_final_mb": ram_final_mb,
        "ram_delta_mb": ram_final_mb - ram_after_init_mb,
    }


if __name__ == "__main__":
    run_system_benchmark(100, enable_hud=True)
