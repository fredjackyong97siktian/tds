import argparse
import cv2
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
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


def _tracking_summary_path(video_path: str, output_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{stem}_tracking_summary.json")


def _reid_views_summary_path(video_path: str, output_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{stem}_reid_views_summary.json")


def _tensor_like_to_float_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return None


def _combine_fashion_embedding(upper, lower):
    upper_list = _tensor_like_to_float_list(upper)
    lower_list = _tensor_like_to_float_list(lower)
    if upper_list and lower_list:
        return upper_list + lower_list
    return upper_list or lower_list


def _build_reid_views_summary(video_path: str, output_dir: str) -> str:
    reid_views_dir = os.path.join(os.path.dirname(output_dir), "reid_views")
    summary = {
        "video": os.path.splitext(os.path.basename(video_path))[0],
        "customers": [],
    }
    customers = []

    if not os.path.isdir(reid_views_dir):
        summary_path = _reid_views_summary_path(video_path, output_dir)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path

    for name in sorted(os.listdir(reid_views_dir)):
        if not name.startswith("ID"):
            continue
        person_id_raw = name[2:]
        if not person_id_raw.isdigit():
            continue
        person_id = int(person_id_raw)
        gid_dir = os.path.join(reid_views_dir, name)
        if not os.path.isdir(gid_dir):
            continue

        view_rows = []
        for image_name in sorted(os.listdir(gid_dir)):
            if not image_name.lower().endswith(".jpg"):
                continue
            image_path = os.path.join(gid_dir, image_name)
            img = cv2.imread(image_path)
            if img is None or img.size == 0:
                continue
            h, w = img.shape[:2]
            box = [0, 0, w, h]
            embedding_osnet = _tensor_like_to_float_list(
                Detect.IntegratedEntry.extract_embedding(img, box)
            )
            fashion_upper, fashion_lower = Detect.IntegratedEntry.extract_fashion_pair(img, box)
            embedding_fashion = _combine_fashion_embedding(fashion_upper, fashion_lower)
            if embedding_osnet is None and embedding_fashion is None:
                continue
            view_rows.append(
                {
                    "image_url": image_path,
                    "image_kind": "reid_view",
                    "embedding_osnet": embedding_osnet,
                    "embedding_fashion": embedding_fashion,
                }
            )

        customers.append(
            {
                "person_id": person_id,
                "views": view_rows,
            }
        )

    summary["customers"] = customers
    summary_path = _reid_views_summary_path(video_path, output_dir)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def _event_person_id(event: object) -> int | None:
    if not isinstance(event, dict):
        return None
    for key in ("person_id", "gid", "id"):
        value = event.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _event_type(event: object) -> str:
    if not isinstance(event, dict):
        return ""
    value = event.get("type") or event.get("event") or event.get("label")
    return str(value or "").strip().lower()


def _build_fallback_tracking_summary(video_path: str, events: object, cross_state: dict) -> dict:
    customers: dict[int, dict] = {}
    first_frames: dict[int, int | None] = {}
    last_frames: dict[int, int | None] = {}
    current_sources: dict[int, str] = {}
    group_ids: defaultdict[int, int] = defaultdict(int)

    if isinstance(events, list):
        for event in events:
            person_id = _event_person_id(event)
            if person_id is None:
                continue
            event_type = _event_type(event)
            frame_value = event.get("frame") if isinstance(event, dict) else None
            frame_no: int | None = None
            try:
                frame_no = int(frame_value) if frame_value is not None else None
            except (TypeError, ValueError):
                frame_no = None

            if person_id not in customers:
                customers[person_id] = {
                    "person_id": person_id,
                    "source": "fallback_events",
                    "entered": False,
                    "exited": False,
                    "entry_frame": None,
                    "exit_frame": None,
                    "last_seen_frame": frame_no,
                    "group_id": person_id,
                }
            customer = customers[person_id]

            if frame_no is not None:
                if first_frames[person_id] is None if person_id in first_frames else True:
                    first_frames[person_id] = frame_no
                else:
                    first_frames[person_id] = min(first_frames[person_id], frame_no)  # type: ignore[arg-type]
                if last_frames[person_id] is None if person_id in last_frames else True:
                    last_frames[person_id] = frame_no
                else:
                    last_frames[person_id] = max(last_frames[person_id], frame_no)  # type: ignore[arg-type]
                customer["last_seen_frame"] = last_frames[person_id]

            if event_type in {"entry", "enter"} or "enter" in event_type or "entry" in event_type:
                customer["entered"] = True
                if customer["entry_frame"] is None:
                    customer["entry_frame"] = frame_no
            if "exit" in event_type or "quit" in event_type:
                customer["exited"] = True
                customer["exit_frame"] = frame_no

            source_value = event.get("source") if isinstance(event, dict) else None
            if source_value:
                current_sources[person_id] = str(source_value)
                customer["source"] = current_sources[person_id]

            group_value = event.get("group_id") if isinstance(event, dict) else None
            try:
                if group_value is not None:
                    group_ids[person_id] = int(group_value)
                    customer["group_id"] = group_ids[person_id]
            except (TypeError, ValueError):
                pass

    persistent_gallery = cross_state.get("persistent_gallery", {})
    for raw_person_id in persistent_gallery.keys():
        try:
            person_id = int(raw_person_id)
        except (TypeError, ValueError):
            continue
        customers.setdefault(
            person_id,
            {
                "person_id": person_id,
                "source": "fallback_gallery",
                "entered": True,
                "exited": False,
                "entry_frame": None,
                "exit_frame": None,
                "last_seen_frame": last_frames.get(person_id),
                "group_id": person_id,
            },
        )

    for customer in customers.values():
        if not customer["entered"] and not customer["exited"]:
            customer["entered"] = True

    return {
        "video": os.path.splitext(os.path.basename(video_path))[0],
        "persistent_gallery_ids": sorted(
            int(gid)
            for gid in persistent_gallery.keys()
            if str(gid).strip()
        ),
        "customers": sorted(customers.values(), key=lambda row: int(row["person_id"])),
    }


def _ensure_tracking_summary(video_path: str, output_dir: str, events: object, cross_state: dict) -> str:
    summary_path = _tracking_summary_path(video_path, output_dir)
    if os.path.exists(summary_path):
        return summary_path

    fallback_summary = _build_fallback_tracking_summary(video_path, events, cross_state)
    with open(summary_path, "w") as f:
        json.dump(fallback_summary, f, indent=2)
    _log(f"fallback tracking summary created path={summary_path}")
    return summary_path


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
    tracking_summary_path = _ensure_tracking_summary(video_path, output_dir, events, cross_state)
    _log(f"ensure_tracking_summary done elapsed={time.time() - stage_started:.2f}s path={tracking_summary_path}")

    stage_started = time.time()
    reid_views_summary_path = _build_reid_views_summary(video_path, output_dir)
    _log(f"build_reid_views_summary done elapsed={time.time() - stage_started:.2f}s path={reid_views_summary_path}")

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
                "reid_views_summary": reid_views_summary_path,
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
