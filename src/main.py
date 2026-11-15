"""
Drawing Monster Defense - Main Application Entry Point
Orchestrates Threaded Camera Ingestion, MediaPipe Hand Tracking,
and Dual-Zone Responsive Air Drawing.
"""

import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tracking.hand_tracker import main

if __name__ == "__main__":
    main()
