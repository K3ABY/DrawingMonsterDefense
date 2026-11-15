import os
import sys
import cv2

# Set path
sys.path.insert(0, r"C:\Users\Minec\OneDrive\Documents\test-handtracking\DrawingMonsterDefense")
import camera_config as config
from hand_tracking import HandTracker, open_camera_stream

def main():
    cap = open_camera_stream(config.CAMERA_URL)
    if cap is None:
        print("Failed to open camera")
        return

    tracker = HandTracker()
    saved = False
    detections = 0

    print("Listening to camera for up to 50 frames...")
    for i in range(50):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        data, res = tracker.process(frame)
        left = data.get("Left")
        right = data.get("Right")

        if left is not None or right is not None:
            detections += 1
            left_info = f"Left: Tip=({left['index_tip_x']}, {left['index_tip_y']}) Conf={left['confidence']:.2f}" if left else "Left: None"
            right_info = f"Right: Tip=({right['index_tip_x']}, {right['index_tip_y']}) Conf={right['confidence']:.2f}" if right else "Right: None"
            print(f"Frame {i:02d}: {left_info} | {right_info}")

            if not saved:
                rendered = tracker.draw(frame, data, res)
                out_path = r"C:\Users\Minec\OneDrive\Documents\test-handtracking\DrawingMonsterDefense\live_hand_detected.jpg"
                cv2.imwrite(out_path, rendered)
                print(f"--> Saved annotated detection frame to {out_path}")
                saved = True

    cap.release()
    tracker.close()
    print(f"\nDone! Detected hands in {detections}/50 frames.")

if __name__ == "__main__":
    main()

