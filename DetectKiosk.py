import argparse
import json
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model_setup import configure_detect_model_env

configure_detect_model_env()

import Detect
from detect_split_state import load_cross_state, save_cross_state


def main():
    parser = argparse.ArgumentParser(description="Run Detect.py Kiosk logic only.")
    parser.add_argument("--video", required=True, help="Path to the kiosk video.")
    parser.add_argument("--output-dir", required=True, help="Directory for kiosk logs and outputs.")
    parser.add_argument(
        "--gallery-state",
        required=True,
        help="Pickle file that stores the shared persistent gallery between Entry and Kiosk.",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    output_dir = os.path.abspath(args.output_dir)
    gallery_state_path = os.path.abspath(args.gallery_state)

    os.makedirs(output_dir, exist_ok=True)
    reid_fashion_views_dir = os.path.join(os.path.dirname(output_dir), "reid_fashion_views")
    os.makedirs(reid_fashion_views_dir, exist_ok=True)
    Detect.REID_FASHION_DEBUG_DIR = reid_fashion_views_dir

    cross_state = load_cross_state(gallery_state_path)
    Detect.ensure_integrated_fashionclip_model()
    events = Detect.process_kiosk_video(video_path, output_dir, cross_state)
    save_cross_state(gallery_state_path, cross_state)

    summary_path = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(video_path))[0]}_kiosk_runner_summary.json",
    )
    with open(summary_path, "w") as f:
        json.dump(
            {
                "video": video_path,
                "output_dir": output_dir,
                "gallery_state": gallery_state_path,
                "event_count": len(events or []),
                "persistent_gallery_ids": sorted(
                    int(gid) for gid in cross_state.get("persistent_gallery", {}).keys()
                ),
            },
            f,
            indent=2,
        )

    print(f"[DetectKiosk] done video={video_path} output={output_dir}")


if __name__ == "__main__":
    main()
