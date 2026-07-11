import os
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
REID_DIR = MODELS_DIR / "reid"


def _set_if_exists(env_name: str, path: Path):
    if path.is_file():
        os.environ.setdefault(env_name, str(path))


def configure_detect_model_env():
    # Entry-side env names used by the integrated Entry logic in Detect.py
    _set_if_exists("MODEL_PATH", MODELS_DIR / "yolo26s.pt")
    _set_if_exists("TRACKER_PATH", MODELS_DIR / "custom_tracker.yaml")
    _set_if_exists("REID_MASK_MODEL_PATH", MODELS_DIR / "yoloe-11l-seg.pt")
    _set_if_exists("REID_WEIGHTS_PATH", REID_DIR / "osnet_x1_0_msmt17.pt")

    # Kiosk-side env names used by Detect.py
    _set_if_exists("KIOSK_PERSON_MODEL_PATH", MODELS_DIR / "yolo26s.pt")
    _set_if_exists("KIOSK_PRODUCT_MODEL_PATH", MODELS_DIR / "yolo26s.pt")
    _set_if_exists("KIOSK_POSE_MODEL_PATH", MODELS_DIR / "yolo26l-pose.pt")
    _set_if_exists("KIOSK_PERSON_REID_WEIGHTS_PATH", REID_DIR / "osnet_x1_0_msmt17.pt")

    # Optional MobileCLIP local checkpoint if later added
    _set_if_exists("MOBILECLIP_PT_PATH", MODELS_DIR / "mobileclip_blt.pt")
