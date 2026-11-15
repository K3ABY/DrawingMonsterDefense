"""
Performance Diagnostic Tool for Drawing Monster Defense.
Measures stream resolution, camera FPS, processing FPS, inference time,
network grab delay, tracking loss, and landmark jitter on the live stream.
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_config as config
from hand_tracking import HandTracker


def run_benchmark(num_frames: int = 100):
    print("=" * 65)
    print(" RUNNING PERFORMANCE DIAGNOSTIC BENCHMARK")
    print(f" Target Stream: {config.CAMERA_URL}")
    print(f" Sample Frames: {num_frames}")
    print("=" * 65)

    cap = cv2.VideoCapture(config.CAMERA_URL)
    if not cap.isOpened():
        print(f"[ERROR] Cannot connect to camera stream at {config.CAMERA_URL}")
        return None

    # Step 1: Benchmark Camera Stream Raw Ingestion Rate (Pure Grab/Read)
    print("\n[Phase 1] Benchmarking Raw Camera Ingestion Rate (30 frames)...")
    raw_times = []
    grab_times = []
    w, h = 0, 0

    for i in range(30):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret or frame is None:
            continue
        h, w, _ = frame.shape
        raw_times.append(t1 - t0)

    avg_raw_grab_ms = np.mean(raw_times) * 1000 if raw_times else 0
    raw_fps = 1.0 / np.mean(raw_times) if raw_times and np.mean(raw_times) > 0 else 0

    print(f"  Stream Resolution     : {w} x {h}")
    print(f"  Avg Frame Grab Time   : {avg_raw_grab_ms:.2f} ms")
    print(f"  Estimated Camera FPS  : {raw_fps:.1f} FPS")

    # Step 2: Full Pipeline Processing Benchmark
    print(f"\n[Phase 2] Benchmarking Full Processing Pipeline ({num_frames} frames)...")
    tracker = HandTracker(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    grab_durations = []
    convert_durations = []
    inference_durations = []
    post_durations = []
    render_durations = []
    total_frame_durations = []

    detection_counts = {"Left": 0, "Right": 0, "Both": 0, "None": 0}
    tracking_loss_events = {"Left": 0, "Right": 0}
    prev_detected = {"Left": False, "Right": False}

    left_tip_positions = []
    right_tip_positions = []

    start_bench_time = time.perf_counter()

    for frame_idx in range(num_frames):
        t_start = time.perf_counter()

        # Measure Frame Grab
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret or frame is None:
            print(f"  [WARN] Dropped frame at index {frame_idx}")
            continue
        grab_durations.append(t1 - t0)

        # Measure Color Convert
        t2 = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        t3 = time.perf_counter()
        convert_durations.append(t3 - t2)

        # Measure MediaPipe Inference
        t4 = time.perf_counter()
        results = tracker.hands.process(rgb_frame)
        t5 = time.perf_counter()
        inference_durations.append(t5 - t4)

        # Measure Post-processing
        t6 = time.perf_counter()
        hand_data, _ = tracker.process(frame)  # standard call
        t7 = time.perf_counter()
        post_durations.append(t7 - t6)

        # Measure Rendering
        t8 = time.perf_counter()
        _ = tracker.draw(frame, hand_data, results)
        t9 = time.perf_counter()
        render_durations.append(t9 - t8)

        total_frame_durations.append(time.perf_counter() - t_start)

        # Detection statistics
        has_left = hand_data["Left"] is not None
        has_right = hand_data["Right"] is not None

        if has_left and has_right:
            detection_counts["Both"] += 1
        elif has_left:
            detection_counts["Left"] += 1
        elif has_right:
            detection_counts["Right"] += 1
        else:
            detection_counts["None"] += 1

        # Tracking loss events (was detected, now lost)
        if prev_detected["Left"] and not has_left:
            tracking_loss_events["Left"] += 1
        if prev_detected["Right"] and not has_right:
            tracking_loss_events["Right"] += 1

        prev_detected["Left"] = has_left
        prev_detected["Right"] = has_right

        if has_left:
            left_tip_positions.append(hand_data["Left"]["index_finger_tip"])
        if has_right:
            right_tip_positions.append(hand_data["Right"]["index_finger_tip"])

    total_wall_time = time.perf_counter() - start_bench_time
    cap.release()
    tracker.close()

    actual_fps = len(total_frame_durations) / total_wall_time if total_wall_time > 0 else 0
    avg_total_ms = np.mean(total_frame_durations) * 1000
    avg_grab_ms = np.mean(grab_durations) * 1000
    avg_inf_ms = np.mean(inference_durations) * 1000
    avg_post_ms = np.mean(post_durations) * 1000
    avg_render_ms = np.mean(render_durations) * 1000

    # Jitter calculation (std dev of frame-to-frame delta)
    def calc_jitter(positions):
        if len(positions) < 4:
            return 0.0
        diffs = [np.hypot(positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
                 for i in range(1, len(positions))]
        return float(np.std(diffs))

    left_jitter = calc_jitter(left_tip_positions)
    right_jitter = calc_jitter(right_tip_positions)

    # Print Report
    print("\n" + "=" * 65)
    print(" PERFORMANCE DIAGNOSIS REPORT")
    print("=" * 65)
    print(f" Resolution             : {w} x {h}")
    print(f" Camera Stream FPS      : {raw_fps:.1f} FPS (Frame interval ~{avg_raw_grab_ms:.1f} ms)")
    print(f" Processing Pipeline FPS: {actual_fps:.1f} FPS (Total frame time ~{avg_total_ms:.1f} ms)")
    print("-" * 65)
    print(" Latency & Timing Breakdown (Average per frame):")
    print(f"   - Video Capture / Grab : {avg_grab_ms:.2f} ms ({avg_grab_ms/avg_total_ms*100:.1f}%)")
    print(f"   - MediaPipe Inference  : {avg_inf_ms:.2f} ms ({avg_inf_ms/avg_total_ms*100:.1f}%)")
    print(f"   - Landmark Extraction  : {avg_post_ms:.2f} ms ({avg_post_ms/avg_total_ms*100:.1f}%)")
    print(f"   - Canvas & HUD Render  : {avg_render_ms:.2f} ms ({avg_render_ms/avg_total_ms*100:.1f}%)")
    print("-" * 65)
    print(" Tracking & Detection Stats:")
    print(f"   - Total Frames Processed: {len(total_frame_durations)}")
    print(f"   - Left Hand Detected    : {detection_counts['Left'] + detection_counts['Both']} frames")
    print(f"   - Right Hand Detected   : {detection_counts['Right'] + detection_counts['Both']} frames")
    print(f"   - Tracking Loss Events  : Left={tracking_loss_events['Left']}, Right={tracking_loss_events['Right']}")
    print(f"   - Landmark Jitter (std) : Left={left_jitter:.2f} px, Right={right_jitter:.2f} px")
    print("=" * 65)

    return {
        "width": w,
        "height": h,
        "camera_fps": raw_fps,
        "processing_fps": actual_fps,
        "avg_total_ms": avg_total_ms,
        "avg_grab_ms": avg_grab_ms,
        "avg_inf_ms": avg_inf_ms,
        "avg_render_ms": avg_render_ms,
        "left_jitter": left_jitter,
        "right_jitter": right_jitter,
    }


if __name__ == "__main__":
    run_benchmark(num_frames=60)

