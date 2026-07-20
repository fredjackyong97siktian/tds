import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model_setup import configure_detect_model_env

configure_detect_model_env()

def _load_detect_module():
    detect_override = os.environ.get("DETECT_MODULE_PATH")
    candidate_paths = [
        detect_override,
        os.path.join(THIS_DIR, "Detect.py"),
        os.path.join(REPO_ROOT, "Detect.py"),
    ]
    detect_path = next((path for path in candidate_paths if path and os.path.isfile(path)), None)
    if detect_path is not None:
        spec = importlib.util.spec_from_file_location("Detect", detect_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Could not load Detect.py from {detect_path}.")
        detect_module = importlib.util.module_from_spec(spec)
        sys.modules["Detect"] = detect_module
        spec.loader.exec_module(detect_module)
        return detect_module

    try:
        import Detect as detect_module

        return detect_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Detect.py was not found in any expected location: {candidate_paths}. Make sure the file is mounted into the container."
        ) from exc


Detect = _load_detect_module()
from detect_split_state import load_cross_state, save_cross_state


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] [DetectEntry] {message}")


def main():
    runner_started = time.time()
    parser = argparse.ArgumentParser(description="Run Detect.py Entry logic only.")
    parser.add_argument("--video", required=True, help="Path to the entry/exit video.")
    parser.add_argument("--output-dir", required=True, help="Directory for entry logs and outputs.")
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
    reid_views_dir = os.path.join(os.path.dirname(output_dir), "reid_views")
    reid_fashion_views_dir = os.path.join(os.path.dirname(output_dir), "reid_fashion_views")
    os.makedirs(reid_views_dir, exist_ok=True)
    os.makedirs(reid_fashion_views_dir, exist_ok=True)

    Detect.IntegratedEntry.OUTPUT_BASE = output_dir
    Detect.IntegratedEntry.CROPS_DIR = None
    Detect.IntegratedEntry.REID_DEBUG_DIR = reid_views_dir
    Detect.IntegratedEntry.REID_FASHION_DEBUG_DIR = reid_fashion_views_dir
    Detect.REID_FASHION_DEBUG_DIR = reid_fashion_views_dir

    _log(f"start video={video_path}")

    stage_started = time.time()
    cross_state = load_cross_state(gallery_state_path)
    _log(f"load_cross_state done elapsed={time.time() - stage_started:.2f}s")

    stage_started = time.time()
    Detect.ensure_integrated_fashionclip_model()
    _log(f"ensure_integrated_fashionclip_model done elapsed={time.time() - stage_started:.2f}s")

    stage_started = time.time()
    events = Detect.IntegratedEntry.process_video(video_path, output_dir, cross_state)
    _log(f"process_video done elapsed={time.time() - stage_started:.2f}s event_count={len(events or [])}")

    stage_started = time.time()
    save_cross_state(gallery_state_path, cross_state)
    _log(f"save_cross_state done elapsed={time.time() - stage_started:.2f}s")

    summary_path = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(video_path))[0]}_entry_runner_summary.json",
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

    _log(
        f"done video={video_path} output={output_dir} total_elapsed={time.time() - runner_started:.2f}s"
    )


if __name__ == "__main__":
    main()
