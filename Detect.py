import glob
import json
import os
import shutil
import time
import traceback
import base64
import mimetypes
import re
import urllib.request
import urllib.error
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = None
    CLIPProcessor = None
class _IntegratedModuleProxy:
    def __init__(self, namespace):
        object.__setattr__(self, "_ns", namespace)

    def __getattr__(self, name):
        try:
            return self._ns[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self._ns[name] = value


def enable_mobileclip_pt_text_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.environ.get("MOBILECLIP_PT_PATH"),
        os.path.join(base_dir, "models", "text", "mobileclip_blt.pt"),
        os.path.join(base_dir, "models", "mobileclip_blt.pt"),
        os.path.join(base_dir, "mobileclip_blt.pt"),
    ]
    pt_path = next((path for path in candidate_paths if path and os.path.isfile(path)), None)
    if pt_path is None:
        print(f"Warning: MobileCLIP PT not found. Checked: {candidate_paths}")
        return False
    try:
        import mobileclip
        import ultralytics.nn.text_model as text_model_mod
    except Exception as e:
        print(f"Warning: cannot enable MobileCLIP PT text model: {e}")
        return False

    original_build_text_model = text_model_mod.build_text_model
    if getattr(original_build_text_model, "_codex_mobileclip_pt", False):
        return True

    class MobileCLIPPT(text_model_mod.TextModel):
        config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

        def __init__(self, size: str, device: torch.device, weight_path: str) -> None:
            super().__init__()
            config = self.config_size_map.get(size, "b")
            self.model = mobileclip.create_model_and_transforms(
                f"mobileclip_{config}",
                pretrained=weight_path,
                device=device,
            )[0]
            self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
            self.to(device)
            self.device = device
            self.eval()

        def tokenize(self, texts: list[str]) -> torch.Tensor:
            return self.tokenizer(texts).to(self.device)

        @text_model_mod.smart_inference_mode()
        def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            text_features = self.model.encode_text(texts).to(dtype)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features

    def build_text_model_patched(variant: str, device: torch.device = None):
        base, size = variant.split(":")
        if base == "mobileclip" and size == "blt":
            return MobileCLIPPT(size=size, device=device, weight_path=pt_path)
        return original_build_text_model(variant, device=device)

    build_text_model_patched._codex_mobileclip_pt = True
    text_model_mod.build_text_model = build_text_model_patched
    print(f"MobileCLIP PT text model enabled: {pt_path}")
    return True


def _first_existing_path(*paths):
    return next((path for path in paths if path and os.path.exists(path)), None)


def _env_or_model_path(env_name, *fallback_paths):
    env_path = os.environ.get(env_name)
    if env_path:
        return env_path
    return _first_existing_path(*fallback_paths) or fallback_paths[-1]


enable_mobileclip_pt_text_model()


# Integrated copy of IntegratedEntry.py so Detect.py can run as a single file.
_ENTRY_SOURCE = 'import cv2\nimport numpy as np\nfrom ultralytics import YOLO\nimport json, os, time, shutil, glob\nimport torch\nimport torch.nn.functional as F\nimport torchvision.transforms as T\nimport torchreid\nfrom PIL import Image\ntry:\n    import timm\nexcept ImportError:\n    timm = None\ntry:\n    from fashion_clip.fashion_clip import FashionCLIP\nexcept ImportError:\n    FashionCLIP = None\n##ISSUE NOW - video \n# [F366] EVENT gid=4 type=Exit image=/Users/fredjackyong/Documents/kebunapp/theft_detection/Output/crops/ID4/5_F000365.jpg shouldn\'t be Exit\n# ==============================\n# CONFIG\n# ==============================\nBASE_DIR = os.path.dirname(os.path.abspath(__file__))\n\nVIDEO_FOLDER = os.environ.get("VIDEO_FOLDER", os.path.join(BASE_DIR, "new_video"))\nOUTPUT_BASE  = os.environ.get("OUTPUT_BASE", os.path.join(BASE_DIR, "Output"))\nCROPS_DIR    = os.environ.get("CROPS_DIR", os.path.join(OUTPUT_BASE, "crops"))\nREID_DEBUG_DIR = os.environ.get("REID_DEBUG_DIR", os.path.join(OUTPUT_BASE, "reid_views"))\nMODEL_PATH   = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "yolo26s.pt"))\nTRACKER_PATH = os.environ.get("TRACKER_PATH", os.path.join(BASE_DIR, "custom_tracker.yaml"))\nREID_MASK_MODEL_PATH = os.environ.get("REID_MASK_MODEL_PATH", os.path.join(BASE_DIR, "yoloe-11l-seg.pt"))\nREID_USE_SEG_MASK = os.environ.get("REID_USE_SEG_MASK", "1") == "1"\nREID_SEG_STRICT = os.environ.get("REID_SEG_STRICT", "1") == "1"\nMAX_FRAMES_PER_VIDEO = int(os.environ.get("MAX_FRAMES_PER_VIDEO", "0"))\n\nFRAME_SKIP   = 1\nRESIZE_WIDTH = 960\nMIN_BOX_HEIGHT = 150\nMIN_BOX_AREA   = 2200\nMIN_VISIBLE_RATIO = float(os.environ.get("MIN_VISIBLE_RATIO", "0.70"))\n\nZONE_LEFT        = 580\nZONE_RIGHT       = 750\nZONE_TOP_OFFSET  = 450\nZONE_BOTTOM_OFFSET = 100\n\n# ===== REID =====\nREID_SIM_THRESHOLD = 0.40   # within-video match if ANY gallery view hits this\nREID_MAX_GAP       = 300    # frames in gallery (~13s at 22fps)\nSPATIAL_GATE       = 200    # pixels\nGALLERY_EMA_ALPHA  = 0.15   # slow update — preserve original views\nMIN_SEEN_FOR_GALLERY = 8    # clean frames before saving to gallery\n\n# Multi-view gallery settings\nMAX_VIEWS_PER_PERSON = 6    # max distinct embeddings stored per person\nVIEW_DIVERSITY_THRESH = 0.75 # only add a new view if it\'s dissimilar enough\n                              # from existing views (lower = more views stored)\n\nOCCLUSION_IOU_THRESHOLD = 0.15\nEVENT_DISPLAY_TTL = 20\nDISAPPEAR_EDGE_MARGIN = int(os.environ.get("DISAPPEAR_EDGE_MARGIN", "35"))\nEXIT_LEFT_MARGIN = int(os.environ.get("EXIT_LEFT_MARGIN", "20"))\nEVENT_BOX_AREA_RATIO_MAX = float(os.environ.get("EVENT_BOX_AREA_RATIO_MAX", "1.80"))\nEVENT_BOX_AREA_RATIO_MIN = float(os.environ.get("EVENT_BOX_AREA_RATIO_MIN", "0.55"))\nEVENT_MAX_FOOT_JUMP = int(os.environ.get("EVENT_MAX_FOOT_JUMP", "90"))\nEVENT_HISTORY_LEN = int(os.environ.get("EVENT_HISTORY_LEN", "8"))\nZONE_OVERLAP_MIN_RATIO = float(os.environ.get("ZONE_OVERLAP_MIN_RATIO", "0.15"))\nEVENT_DEBUG = os.environ.get("EVENT_DEBUG", "0") == "1"\nMIN_EVENT_DET_CONF = float(os.environ.get("MIN_EVENT_DET_CONF", "0.5"))\nMIN_EXIT_ZONE_SAMPLES = int(os.environ.get("MIN_EXIT_ZONE_SAMPLES", "4"))\nMIN_EXIT_VISIBLE_RATIO = float(os.environ.get("MIN_EXIT_VISIBLE_RATIO", "0.8"))\nMIN_EXIT_BOX_WIDTH = int(os.environ.get("MIN_EXIT_BOX_WIDTH", "55"))\nMIN_EXIT_BOX_HEIGHT = int(os.environ.get("MIN_EXIT_BOX_HEIGHT", "110"))\nEXIT_CENTER_MARGIN_X = int(os.environ.get("EXIT_CENTER_MARGIN_X", "20"))\nEXIT_CENTER_MARGIN_Y = int(os.environ.get("EXIT_CENTER_MARGIN_Y", "12"))\nMIN_CARRYOVER_SEEN_FRAMES = int(os.environ.get("MIN_CARRYOVER_SEEN_FRAMES", "4"))\nREID_SCORE_DEBUG = os.environ.get("REID_SCORE_DEBUG", "1") == "1"\nREID_WITHIN_VIDEO_DEBUG = os.environ.get("REID_WITHIN_VIDEO_DEBUG", "0") == "1"\nWITHIN_VIDEO_VISIBILITY_LOG = os.environ.get("WITHIN_VIDEO_VISIBILITY_LOG", "1") == "1"\nREID_UPDATE_VIEW_LOG = os.environ.get("REID_UPDATE_VIEW_LOG", "1") == "1"\nREID_UPDATE_VIEW_MIN_SIM = float(os.environ.get("REID_UPDATE_VIEW_MIN_SIM", "0.70"))\nREID_UPDATE_VIEW_MIN_OS = float(os.environ.get("REID_UPDATE_VIEW_MIN_OS", "0.70"))\nREID_UPDATE_VIEW_MIN_FC = 0.60\nLOST_RELINK_MIN_STREAK = int(os.environ.get("LOST_RELINK_MIN_STREAK", "4"))\nSAVE_REID_DEBUG_CROPS = os.environ.get("SAVE_REID_DEBUG_CROPS", "1") == "1"\nID_BOOK_RELEASE_MISSING = int(os.environ.get("ID_BOOK_RELEASE_MISSING", "12"))\n\n# Cross-video Re-ID. Same person across multiple videos should keep the same\n# global ID. Stricter than within-video threshold and uses an ambiguity gap.\nCROSS_VIDEO_REID = os.environ.get("CROSS_VIDEO_REID", "1") == "1"\nCROSS_VIDEO_REID_SIM_THRESHOLD = float(os.environ.get("CROSS_VIDEO_REID_SIM_THRESHOLD", "0.1"))\nCROSS_VIDEO_MIN_VISIBLE_RATIO = float(os.environ.get("CROSS_VIDEO_MIN_VISIBLE_RATIO", "0.70"))\nCROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1 = os.environ.get("CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1", "1") == "0"\nCROSS_VIDEO_DELAY_ON_AMBIGUOUS = os.environ.get("CROSS_VIDEO_DELAY_ON_AMBIGUOUS", "1") == "1"\nCROSS_VIDEO_MIN_STREAK = int(os.environ.get("CROSS_VIDEO_MIN_STREAK", "4")) #ISSUE that cause the inconsistency of the ID2\nNEW_ID_AGG_FRAMES = int(os.environ.get("NEW_ID_AGG_FRAMES", "1"))\nNEW_ID_PERSIST_SIM_THRESHOLD = float(\n    os.environ.get("NEW_ID_PERSIST_SIM_THRESHOLD", str(CROSS_VIDEO_REID_SIM_THRESHOLD + 0.10))\n)\n\n# Re-ID model setup. Default to local OSNet checkpoint.\nREID_MODEL_NAME = os.environ.get("REID_MODEL_NAME", "osnet_x1_0")\nDEFAULT_REID_WEIGHTS_PATH = (\n    os.path.join(BASE_DIR, "models", "osnet_x1_0_msmt17.pt")\n    if REID_MODEL_NAME.startswith("osnet")\n    else ""\n)\nREID_WEIGHTS_PATH = os.environ.get(\n    "REID_WEIGHTS_PATH",\n    DEFAULT_REID_WEIGHTS_PATH,\n)\nREID_INPUT_HEIGHT = int(os.environ.get("REID_INPUT_HEIGHT", "256"))\nREID_INPUT_WIDTH = int(os.environ.get("REID_INPUT_WIDTH", "128"))\nREID_TIMM_PRETRAINED = os.environ.get("REID_TIMM_PRETRAINED", "0") == "1"\nFASHIONCLIP_ENABLE = os.environ.get("FASHIONCLIP_ENABLE", "1") == "1"\nFASHIONCLIP_MODEL_NAME = os.environ.get("FASHIONCLIP_MODEL_NAME", "fashion-clip")\nFASHIONCLIP_UPPER_WEIGHT = float(os.environ.get("FASHIONCLIP_UPPER_WEIGHT", "0.8"))\nFASHIONCLIP_LOWER_WEIGHT = float(os.environ.get("FASHIONCLIP_LOWER_WEIGHT", "0.2"))\nFASHIONCLIP_WITHIN_ALPHA = float(os.environ.get("FASHIONCLIP_WITHIN_ALPHA", "0.20"))\nFASHIONCLIP_CROSS_ALPHA = float(os.environ.get("FASHIONCLIP_CROSS_ALPHA", "0.25"))\n\n# ==============================\n# DEVICE + MODEL\n# ==============================\nif torch.backends.mps.is_available():\n    DEVICE = torch.device("mps")\n    print("Using Apple MPS")\nelif torch.cuda.is_available():\n    DEVICE = torch.device("cuda")\n    print("Using CUDA")\nelse:\n    DEVICE = torch.device("cpu")\n    print("Using CPU")\n\ndef _extract_state_dict(checkpoint):\n    if isinstance(checkpoint, dict):\n        for key in ("state_dict", "model", "model_state_dict", "net"):\n            if key in checkpoint and isinstance(checkpoint[key], dict):\n                checkpoint = checkpoint[key]\n                break\n    if not isinstance(checkpoint, dict):\n        raise TypeError("Unsupported checkpoint format")\n\n    cleaned = {}\n    for key, value in checkpoint.items():\n        if key.startswith("module."):\n            key = key[len("module."):]\n        cleaned[key] = value\n    return cleaned\n\n\ndef _filter_incompatible_keys(model, state_dict):\n    model_state = model.state_dict()\n    filtered = {}\n    skipped = []\n    for key, value in state_dict.items():\n        if key not in model_state:\n            skipped.append(key)\n            continue\n        if tuple(value.shape) != tuple(model_state[key].shape):\n            skipped.append(key)\n            continue\n        filtered[key] = value\n    return filtered, skipped\n\n\nclass TimmEmbeddingModel(torch.nn.Module):\n    def __init__(self, model_name, pretrained=False):\n        super().__init__()\n        self.backbone = timm.create_model(\n            model_name,\n            pretrained=pretrained,\n            num_classes=0,\n        )\n        self.feature_dim = getattr(self.backbone, "num_features", None)\n\n    def forward(self, x):\n        return self.backbone(x)\n\n\ndef load_reid_model():\n    print(f"Loading Re-ID model: {REID_MODEL_NAME}")\n    using_timm = timm is not None and REID_MODEL_NAME in timm.list_models()\n\n    if using_timm:\n        model = TimmEmbeddingModel(\n            REID_MODEL_NAME,\n            pretrained=REID_TIMM_PRETRAINED and not REID_WEIGHTS_PATH,\n        )\n    else:\n        model = torchreid.models.build_model(\n            name=REID_MODEL_NAME,\n            num_classes=1000,\n            pretrained=False,\n        )\n\n    if REID_WEIGHTS_PATH and os.path.exists(REID_WEIGHTS_PATH):\n        try:\n            checkpoint = torch.load(REID_WEIGHTS_PATH, map_location="cpu", weights_only=False)\n        except UnicodeDecodeError:\n            # Legacy torchreid checkpoints (e.g. resnet50_msmt17) were pickled\n            # under Python 2 and need explicit latin1 encoding to unpickle.\n            checkpoint = torch.load(REID_WEIGHTS_PATH, map_location="cpu",\n                                    weights_only=False, encoding="latin1")\n        state_dict = _extract_state_dict(checkpoint)\n        state_dict, skipped = _filter_incompatible_keys(model, state_dict)\n        missing, unexpected = model.load_state_dict(state_dict, strict=False)\n        print(f"Loaded Re-ID weights from {REID_WEIGHTS_PATH}")\n        if skipped:\n            print(f"  Skipped incompatible keys: {len(skipped)}")\n        if missing:\n            print(f"  Missing keys while loading: {len(missing)}")\n        if unexpected:\n            print(f"  Unexpected keys while loading: {len(unexpected)}")\n    else:\n        print("WARNING: No local Re-ID checkpoint found.")\n        if using_timm and REID_TIMM_PRETRAINED:\n            print("         timm pretrained weights were requested, but could not be loaded locally.")\n        print("         Using randomly initialized weights will make re-identification unreliable.")\n        print(f"         Expected checkpoint path: {REID_WEIGHTS_PATH}")\n\n    model = model.to(DEVICE)\n    model.eval()\n    print(f"Re-ID ready (feature_dim={getattr(model, \'feature_dim\', \'unknown\')})")\n    return model\n\n\nreid_model = load_reid_model()\n\nREID_TRANSFORM = T.Compose([\n    T.ToPILImage(),\n    T.Resize((REID_INPUT_HEIGHT, REID_INPUT_WIDTH)),\n    T.ToTensor(),\n    T.Normalize(mean=[0.485, 0.456, 0.406],\n                std=[0.229, 0.224, 0.225]),\n])\n\nos.makedirs(OUTPUT_BASE, exist_ok=True)\nyolo_model = YOLO(MODEL_PATH)\nreid_mask_model = None\nif REID_USE_SEG_MASK:\n    if os.path.exists(REID_MASK_MODEL_PATH):\n        try:\n            reid_mask_model = YOLO(REID_MASK_MODEL_PATH)\n            try:\n                reid_mask_model.set_classes(["person"])\n                print("ReID segmentation classes set: [\'person\']")\n            except Exception as e:\n                print(f"Warning: cannot set segmentation classes to person: {e}")\n            print(f"ReID segmentation ready: {REID_MASK_MODEL_PATH}")\n        except Exception as e:\n            print(f"Warning: cannot load ReID segmentation model {REID_MASK_MODEL_PATH}: {e}")\n    else:\n        print(f"Warning: ReID segmentation model not found: {REID_MASK_MODEL_PATH}")\nfashionclip_model = None\nif FASHIONCLIP_ENABLE:\n    if FashionCLIP is None:\n        print("Warning: fashion_clip is not installed. FashionCLIP fusion disabled.")\n    else:\n        try:\n            fashionclip_model = FashionCLIP(FASHIONCLIP_MODEL_NAME)\n            print(f"FashionCLIP ready: {FASHIONCLIP_MODEL_NAME}")\n        except Exception as e:\n            print(f"Warning: cannot load FashionCLIP model {FASHIONCLIP_MODEL_NAME}: {e}")\n\n\n# ==============================\n# EMBEDDING HELPERS\n# ==============================\nSEG_CROP_CACHE = {}\n\ndef _crop_with_person_segmentation(frame, box, return_used=False, cache_key=None):\n    if cache_key is not None and cache_key in SEG_CROP_CACHE:\n        crop_cached, used_cached, reason_cached = SEG_CROP_CACHE[cache_key]\n        if return_used:\n            return crop_cached, used_cached, reason_cached\n        return crop_cached\n\n    x1, y1, x2, y2 = map(int, box)\n    x1, y1 = max(0, x1), max(0, y1)\n    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)\n    if y2 <= y1 or x2 <= x1:\n        return (None, False, "invalid_box") if return_used else None\n    crop = frame[y1:y2, x1:x2]\n    if crop.size == 0:\n        return (None, False, "empty_crop") if return_used else None\n    if reid_mask_model is None:\n        return (crop, False, "model_unavailable") if return_used else crop\n\n    try:\n        # For YOLOE/open-vocabulary segmentation, fixed COCO class filters\n        # (e.g. classes=[0]) can return empty masks. Run unfiltered and keep\n        # the largest instance inside this already person-localized crop.\n        seg = reid_mask_model.predict(crop, conf=0.10, verbose=False)\n        if not seg or seg[0].masks is None or seg[0].boxes is None:\n            return (crop, False, "no_mask") if return_used else crop\n        masks = seg[0].masks.data\n        boxes = seg[0].boxes.xyxy\n        if masks is None or len(masks) == 0 or boxes is None or len(boxes) == 0:\n            return (crop, False, "no_mask") if return_used else crop\n        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])\n        best_idx = int(torch.argmax(areas).item())\n        mask = masks[best_idx].detach().cpu().numpy()\n        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)\n        mask_bin = (mask > 0.5).astype(np.uint8)\n        if int(mask_bin.sum()) <= 0:\n            return (crop, False, "mask_empty") if return_used else crop\n        masked = crop.copy()\n        masked[mask_bin == 0] = 0\n        result = (masked, True, "ok")\n        if cache_key is not None:\n            SEG_CROP_CACHE[cache_key] = result\n        if return_used:\n            return result\n        return masked\n    except Exception:\n        result = (crop, False, "predict_error")\n        if cache_key is not None:\n            SEG_CROP_CACHE[cache_key] = result\n        if return_used:\n            return result\n        return crop\n\n@torch.no_grad()\ndef extract_embedding(frame, box, return_seg_used=False, cache_key=None):\n    """Returns an L2-normalised embedding tensor, or None if crop too small."""\n    x1, y1, x2, y2 = map(int, box)\n    x1, y1 = max(0, x1), max(0, y1)\n    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)\n    if (y2 - y1) < 32 or (x2 - x1) < 16:\n        return (None, False, "small_box") if return_seg_used else None\n    crop_bgr, seg_used, seg_reason = _crop_with_person_segmentation(\n        frame, box, return_used=True, cache_key=cache_key\n    )\n    if crop_bgr is None or crop_bgr.size == 0:\n        return (None, seg_used, seg_reason) if return_seg_used else None\n    crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)\n    t = REID_TRANSFORM(crop).unsqueeze(0).to(DEVICE)\n    feat = reid_model(t)\n    emb = F.normalize(feat, p=2, dim=1).cpu().squeeze(0)\n    if return_seg_used:\n        return emb, seg_used, seg_reason\n    return emb\n\n\ndef cosine_sim(a, b):\n    return float(torch.dot(a, b).clamp(0, 1))\n\n\ndef best_sim_against_views(query_emb, views):\n    """\n    Match query against a list of view embeddings.\n    Returns the highest cosine similarity found.\n    """\n    if not views:\n        return 0.0\n    return max(cosine_sim(query_emb, v) for v in views)\n\n\ndef avg_sim_against_views(query_emb, views):\n    """\n    Match query against a list of view embeddings.\n    Returns the average cosine similarity across all views.\n    """\n    if not views:\n        return 0.0\n    sims = [cosine_sim(query_emb, v) for v in views]\n    return float(sum(sims) / max(1, len(sims)))\n\n\ndef maybe_add_view(views, new_emb):\n    """\n    Add new_emb to the view list only if it\'s sufficiently different\n    from all existing views (ensures diversity, avoids redundant copies).\n    Caps at MAX_VIEWS_PER_PERSON.\n    """\n    if new_emb is None:\n        return views\n    if len(views) >= MAX_VIEWS_PER_PERSON:\n        return views\n    # Check diversity — only add if dissimilar enough from all stored views\n    for v in views:\n        if cosine_sim(new_emb, v) > VIEW_DIVERSITY_THRESH:\n            return views   # too similar to an existing view, skip\n    return views + [new_emb]\n\n\n@torch.no_grad()\ndef extract_fashionclip_embedding(crop_bgr):\n    if fashionclip_model is None or crop_bgr is None or crop_bgr.size == 0:\n        return None\n    try:\n        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)\n        pil = Image.fromarray(rgb)\n        vec = fashionclip_model.encode_images([pil], batch_size=1)[0]\n        if isinstance(vec, np.ndarray):\n            t = torch.from_numpy(vec).float()\n        else:\n            t = torch.tensor(vec, dtype=torch.float32)\n        if t.ndim != 1:\n            t = t.view(-1)\n        return F.normalize(t.unsqueeze(0), p=2, dim=1).squeeze(0)\n    except Exception:\n        return None\n\n\ndef extract_fashion_pair(frame, box, return_seg_used=False, cache_key=None):\n    crop, seg_used, seg_reason = _crop_with_person_segmentation(\n        frame, box, return_used=True, cache_key=cache_key\n    )\n    if crop is None or crop.size == 0:\n        return (None, None, seg_used, seg_reason) if return_seg_used else (None, None)\n    h = crop.shape[0]\n    if h < 24:\n        return (None, None, seg_used, seg_reason) if return_seg_used else (None, None)\n    split = int(h * 0.55)\n    upper = crop[:split, :]\n    lower = crop[split:, :]\n    upper_emb = extract_fashionclip_embedding(upper)\n    lower_emb = extract_fashionclip_embedding(lower)\n    if return_seg_used:\n        return upper_emb, lower_emb, seg_used, seg_reason\n    return upper_emb, lower_emb\n\n\ndef fashion_pair_similarity(curr_u, curr_l, ref_u, ref_l):\n    weighted_num = 0.0\n    weighted_den = 0.0\n    if curr_u is not None and ref_u is not None:\n        weighted_num += FASHIONCLIP_UPPER_WEIGHT * cosine_sim(curr_u, ref_u)\n        weighted_den += FASHIONCLIP_UPPER_WEIGHT\n    if curr_l is not None and ref_l is not None:\n        weighted_num += FASHIONCLIP_LOWER_WEIGHT * cosine_sim(curr_l, ref_l)\n        weighted_den += FASHIONCLIP_LOWER_WEIGHT\n    if weighted_den <= 0.0:\n        return -1.0\n    return weighted_num / weighted_den\n\n\ndef ema_update_views(views, new_emb):\n    """\n    Update the closest existing view with EMA, or add as new view.\n    This keeps views fresh without destroying diversity.\n    """\n    if new_emb is None or not views:\n        return views\n    sims = [cosine_sim(new_emb, v) for v in views]\n    best_idx = int(np.argmax(sims))\n    if sims[best_idx] > VIEW_DIVERSITY_THRESH:\n        # Update the closest view with EMA\n        updated = (1 - GALLERY_EMA_ALPHA) * views[best_idx] + GALLERY_EMA_ALPHA * new_emb\n        updated = F.normalize(updated.unsqueeze(0), p=2, dim=1).squeeze(0)\n        new_views = views.copy()\n        new_views[best_idx] = updated\n        return new_views\n    else:\n        # New distinct view — add it\n        return maybe_add_view(views, new_emb)\n\n\n# ==============================\n# GEOMETRY HELPERS\n# ==============================\n\ndef get_centroid(box):\n    x1, y1, x2, y2 = box\n    return int((x1+x2)/2), int((y1+y2)/2)\n\ndef get_foot(box):\n    x1, y1, x2, y2 = box\n    return int((x1+x2)/2), int(y2)\n\ndef get_box_area(box):\n    x1, y1, x2, y2 = box\n    return max(0.0, x2 - x1) * max(0.0, y2 - y1)\n\ndef get_box_size(box):\n    x1, y1, x2, y2 = box\n    return max(0.0, x2 - x1), max(0.0, y2 - y1)\n\ndef get_visible_ratio_in_frame(box, frame_w, frame_h):\n    x1, y1, x2, y2 = box\n    full_area = max(1.0, (x2 - x1) * (y2 - y1))\n    cx1, cy1 = max(0.0, x1), max(0.0, y1)\n    cx2, cy2 = min(float(frame_w), x2), min(float(frame_h), y2)\n    if cx2 <= cx1 or cy2 <= cy1:\n        return 0.0\n    clipped_area = (cx2 - cx1) * (cy2 - cy1)\n    return float(clipped_area / full_area)\n\n\ndef smooth_box(prev, new):\n    return new if prev is None else prev * 0.7 + new * 0.3\n\ndef is_event_box_stable(mem, box):\n    last_box = mem.get("last_event_box")\n    if last_box is None:\n        return True\n\n    last_area = max(1.0, get_box_area(last_box))\n    area_ratio = get_box_area(box) / last_area\n    if area_ratio > EVENT_BOX_AREA_RATIO_MAX or area_ratio < EVENT_BOX_AREA_RATIO_MIN:\n        return False\n\n    fx, _ = get_foot(box)\n    last_fx = mem.get("last_event_foot_x", fx)\n    if abs(fx - last_fx) > EVENT_MAX_FOOT_JUMP:\n        return False\n\n    return True\n\ndef append_event_history(mem, box):\n    cx, _ = get_centroid(box)\n    bw, bh = get_box_size(box)\n    mem.setdefault("zone_center_history", []).append(cx)\n    mem.setdefault("zone_width_history", []).append(bw)\n    mem.setdefault("zone_height_history", []).append(bh)\n    for key in ("zone_center_history", "zone_width_history", "zone_height_history"):\n        if len(mem[key]) > EVENT_HISTORY_LEN:\n            mem[key] = mem[key][-EVENT_HISTORY_LEN:]\n\ndef box_iou(b1, b2):\n    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])\n    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])\n    inter = max(0, ix2-ix1) * max(0, iy2-iy1)\n    if inter == 0: return 0.0\n    return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) +\n                    (b2[2]-b2[0])*(b2[3]-b2[1]) - inter + 1e-6)\n\ndef foot_zone_status(box, frame_h):\n    fx, fy = get_foot(box)\n    if not (frame_h - ZONE_TOP_OFFSET <= fy <= frame_h - ZONE_BOTTOM_OFFSET):\n        return "OUT_Y"\n    if fx < ZONE_LEFT:\n        return "LEFT"\n    if fx > ZONE_RIGHT:\n        return "RIGHT"\n    return "INSIDE"\n\ndef get_zone_bounds(frame_h):\n    return (\n        ZONE_LEFT,\n        frame_h - ZONE_TOP_OFFSET,\n        ZONE_RIGHT,\n        frame_h - ZONE_BOTTOM_OFFSET,\n    )\n\ndef is_in_door_zone_center(box, frame_h):\n    x1, y1, x2, y2 = box\n    zx1, zy1, zx2, zy2 = get_zone_bounds(frame_h)\n    ox1, oy1 = max(x1, zx1), max(y1, zy1)\n    ox2, oy2 = min(x2, zx2), min(y2, zy2)\n    if ox2 <= ox1 or oy2 <= oy1:\n        return False\n    overlap_area = (ox2 - ox1) * (oy2 - oy1)\n    box_area = max(1.0, get_box_area(box))\n    overlap_ratio = overlap_area / box_area\n    return overlap_ratio >= ZONE_OVERLAP_MIN_RATIO\n\ndef is_at_disappear_boundary(box, frame_w, frame_h):\n    x1, y1, x2, y2 = box\n    at_frame_edge = (\n        x1 <= DISAPPEAR_EDGE_MARGIN or\n        y1 <= DISAPPEAR_EDGE_MARGIN or\n        x2 >= frame_w - DISAPPEAR_EDGE_MARGIN or\n        y2 >= frame_h - DISAPPEAR_EDGE_MARGIN\n    )\n    # The configured door zone\'s right edge is also treated as an exit boundary.\n    fx, fy = get_foot(box)\n    at_door_right_boundary = (\n        fx >= ZONE_RIGHT - DISAPPEAR_EDGE_MARGIN and\n        frame_h - ZONE_TOP_OFFSET <= fy <= frame_h - ZONE_BOTTOM_OFFSET\n    )\n    return at_frame_edge or at_door_right_boundary\n\ndef save_crop(frame, box, gid, vname, frame_count, prefix=""):\n    """Save a person crop to the global per-ID folder. Returns the path\n    written, or None if the crop was empty."""\n    out_dir = os.path.join(CROPS_DIR, f"ID{gid}")\n    os.makedirs(out_dir, exist_ok=True)\n    x1, y1, x2, y2 = map(int, box)\n    x1, y1 = max(0, x1), max(0, y1)\n    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)\n    if y2 <= y1 or x2 <= x1:\n        return None\n    fname = f"{prefix}{vname}_F{frame_count:06d}.jpg"\n    path = os.path.join(out_dir, fname)\n    cv2.imwrite(path, frame[y1:y2, x1:x2])\n    return path\n\n\ndef save_reid_view_crop(frame, box, gid, vname, frame_count, tag):\n    """Save crops used to build/update ReID gallery views, grouped by ID."""\n    out_dir = os.path.join(REID_DEBUG_DIR, f"ID{gid}")\n    os.makedirs(out_dir, exist_ok=True)\n    x1, y1, x2, y2 = map(int, box)\n    x1, y1 = max(0, x1), max(0, y1)\n    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)\n    if y2 <= y1 or x2 <= x1:\n        return None\n    fname = f"{vname}_F{frame_count:06d}_{tag}.jpg"\n    path = os.path.join(out_dir, fname)\n    cv2.imwrite(path, frame[y1:y2, x1:x2])\n    return path\n\n\n# ==============================\n# PER-VIDEO PROCESSOR\n# ==============================\n\ndef process_video(video_path, output_dir, cross_state=None):\n    if cross_state is None:\n        cross_state = {"next_gid": 1, "persistent_gallery": {}}\n    persistent_gallery = cross_state["persistent_gallery"]\n\n    vname = os.path.splitext(os.path.basename(video_path))[0]\n    out_vid   = os.path.join(output_dir, f"{vname}_output.mp4")\n    out_json  = os.path.join(output_dir, f"{vname}_events.json")\n    out_log   = os.path.join(output_dir, f"{vname}_log.txt")\n    tmp_vid   = os.path.join(output_dir, f"{vname}_temp.mp4")\n\n    def vlog(msg):\n        msg_str = str(msg)\n        prefix = f"[{vname}] "\n        log_msg = "\\n".join(prefix + line if line else line for line in msg_str.splitlines())\n        print(log_msg)\n        with open(out_log, "a") as lf:\n            lf.write(log_msg + "\\n")\n\n    # Start a fresh per-video log file each run.\n    with open(out_log, "w") as lf:\n        lf.write("")\n\n    vlog(f"\\n{\'=\'*60}\\nProcessing: {vname}\\n{\'=\'*60}")\n    if persistent_gallery:\n        vlog(f"  Persistent gallery has {len(persistent_gallery)} known ID(s): "\n             f"{sorted(persistent_gallery.keys())}")\n\n    # ── Per-video state ───────────────────────────────────────────────\n    track_memory  = {}\n    lost_gallery  = {}   # gid → {center, frame, views: [tensor, ...]}\n    pending_new_trackers = {}  # tracker_id -> {"embs": [tensor]}\n    pending_lost_relinks = {}  # tracker_id -> {"gid", "streak", "last_frame", "last_sim"}\n    pending_cross_video_relinks = {}  # tracker_id -> {"gid", "streak", "last_frame", "last_sim"}\n    pending_tracker_state = {}  # tracker_id -> pre-ID zone/event state\n    id_map        = {}   # yolo tracker_id → global_id\n    events        = []\n    event_display = {}\n    next_gid      = cross_state["next_gid"]\n\n    # ------------------------------------------------------------------\n    def emit_event(gid, event_name, frame_count, image_path):\n        last_event = track_memory[gid].get("last_event")\n        if last_event == event_name:\n            return False\n        # Exit is valid only if this ID has entered before.\n        if event_name == "Exit" and not track_memory[gid].get("has_entry_history", False):\n            return False\n        if event_name == "Exit":\n            if track_memory[gid].get("last_det_conf", 0.0) < MIN_EVENT_DET_CONF:\n                return False\n            if (track_memory[gid].get("zone_samples", 0) < MIN_EXIT_ZONE_SAMPLES\n                    and not track_memory[gid].get("allow_carryover_exit", False)):\n                return False\n            # TEMP: anti-false-exit quality gates disabled\n            # event_box = track_memory[gid].get("last_event_box")\n            # if event_box is None:\n            #     return False\n            # vis_ratio = get_visible_ratio_in_frame(event_box, new_w, new_h)\n            # bw, bh = get_box_size(event_box)\n            # cx, cy = get_centroid(event_box)\n            # if vis_ratio < MIN_EXIT_VISIBLE_RATIO:\n            #     if EVENT_DEBUG:\n            #         vlog(f"  [F{frame_count}] EXIT blocked gid={gid}: vis={vis_ratio:.2f} < {MIN_EXIT_VISIBLE_RATIO:.2f}")\n            #     return False\n            # if bw < MIN_EXIT_BOX_WIDTH or bh < MIN_EXIT_BOX_HEIGHT:\n            #     if EVENT_DEBUG:\n            #         vlog(\n            #             f"  [F{frame_count}] EXIT blocked gid={gid}: "\n            #             f"box=({bw:.1f},{bh:.1f}) < ({MIN_EXIT_BOX_WIDTH},{MIN_EXIT_BOX_HEIGHT})"\n            #         )\n            #     return False\n            # if (cx < EXIT_CENTER_MARGIN_X or cx > (new_w - EXIT_CENTER_MARGIN_X)\n            #         or cy < EXIT_CENTER_MARGIN_Y or cy > (new_h - EXIT_CENTER_MARGIN_Y)):\n            #     if EVENT_DEBUG:\n            #         vlog(\n            #             f"  [F{frame_count}] EXIT blocked gid={gid}: "\n            #             f"center=({cx:.1f},{cy:.1f}) near edge margins"\n            #         )\n            #     return False\n        events.append({\n            "person_id": int(gid),\n            "event": event_name,\n            "frame": int(frame_count),\n            "image": str(image_path),\n        })\n        # Keep status for overlay, but allow future opposite events.\n        track_memory[gid]["state"] = event_name.lower()\n        track_memory[gid]["last_event"] = event_name\n        if event_name == "Entry":\n            track_memory[gid]["has_entry_history"] = True\n        if event_name == "Exit":\n            track_memory[gid]["allow_carryover_exit"] = False\n        # Re-arm zone tracking so next event requires entering zone again.\n        track_memory[gid]["was_in_door_zone"] = False\n        track_memory[gid]["last_in_door_zone"] = False\n        track_memory[gid]["zone_entry_center_x"] = None\n        track_memory[gid]["last_zone_center_x"] = None\n        track_memory[gid]["zone_entry_foot_x"] = None\n        track_memory[gid]["last_zone_foot_x"] = None\n        track_memory[gid]["zone_center_history"] = []\n        track_memory[gid]["zone_width_history"] = []\n        track_memory[gid]["zone_height_history"] = []\n\n        # Cross-video identity should only keep people currently "inside".\n        # On Exit, remove them immediately from the persistent gallery.\n        if event_name == "Exit":\n            persistent_gallery.pop(gid, None)\n\n        vlog(f"  [F{frame_count}] EVENT gid={gid} type={event_name} image={image_path}")\n\n        event_display[gid] = {\n            "event": event_name.upper(),\n            "ttl": EVENT_DISPLAY_TTL,\n        }\n        return True\n\n    def classify_door_event(mem, box, frame_h, disappeared=False):\n        cx, _ = get_centroid(box)\n        zone_start_x = mem.get("zone_entry_center_x")\n        last_zone_x = mem.get("last_zone_center_x")\n        if zone_start_x is None:\n            zone_start_x = mem.get("first_cx", cx)\n        if last_zone_x is None:\n            last_zone_x = mem.get("last_cx", cx)\n\n        zone_width = max(1.0, float(ZONE_RIGHT - ZONE_LEFT))\n        right_half_threshold = ZONE_LEFT + zone_width * 0.55\n        moved_right = last_zone_x >= zone_start_x - 2\n\n        # Main rule: if person was in zone and then disappears, classify by\n        # last in-zone position (robust when they vanish before crossing line).\n        if disappeared:\n            if last_zone_x >= right_half_threshold and moved_right:\n                return "Entry"\n            return "Exit"\n\n        # Visible leave fallback (no disappearance yet).\n        if not is_in_door_zone_center(box, frame_h):\n            if last_zone_x >= right_half_threshold and cx > ZONE_RIGHT:\n                return "Entry"\n            if cx < ZONE_LEFT + EXIT_LEFT_MARGIN:\n                return "Exit"\n            # Left the zone but still near center: treat as exit by default.\n            if last_zone_x < right_half_threshold:\n                return "Exit"\n\n        return None\n\n    def update_pending_tracker_state(tracker_id, box, frame_h):\n        st = pending_tracker_state.get(tracker_id)\n        if st is None:\n            st = {\n                "was_in_door_zone": False,\n                "last_in_door_zone": False,\n                "zone_entry_center_x": None,\n                "last_zone_center_x": None,\n                "zone_entry_foot_x": None,\n                "last_zone_foot_x": None,\n                "last_zone_box": None,\n                "last_foot_side": None,\n                "zone_center_history": [],\n                "zone_width_history": [],\n                "zone_height_history": [],\n                "samples": 0,\n            }\n            pending_tracker_state[tracker_id] = st\n\n        in_door = is_in_door_zone_center(box, frame_h)\n        st["last_in_door_zone"] = in_door\n        cx, _ = get_centroid(box)\n        fx, _ = get_foot(box)\n        if in_door:\n            st["was_in_door_zone"] = True\n            if st["zone_entry_center_x"] is None:\n                st["zone_entry_center_x"] = cx\n            st["last_zone_center_x"] = cx\n            if st["zone_entry_foot_x"] is None:\n                st["zone_entry_foot_x"] = fx\n            st["last_zone_foot_x"] = fx\n            st["last_zone_box"] = box.copy()\n            st["last_foot_side"] = "INSIDE"\n            st["samples"] += 1\n            bw, bh = get_box_size(box)\n            st["zone_center_history"].append(cx)\n            st["zone_width_history"].append(bw)\n            st["zone_height_history"].append(bh)\n            for k in ("zone_center_history", "zone_width_history", "zone_height_history"):\n                if len(st[k]) > EVENT_HISTORY_LEN:\n                    st[k] = st[k][-EVENT_HISTORY_LEN:]\n        else:\n            st["last_foot_side"] = foot_zone_status(box, frame_h)\n\n    def pop_pending_tracker_state(tracker_id):\n        return pending_tracker_state.pop(tracker_id, None)\n\n    def is_gid_booked(gid, requester_tid=None):\n        """\n        Prevent another tracker from taking a gid that is still owned by a\n        recently-seen tracker. Release only after enough missing frames.\n        """\n        # TEMP: lock disabled\n        # return False\n        if ID_BOOK_RELEASE_MISSING <= 0:\n            return False\n        for tid, mapped_gid in id_map.items():\n            if mapped_gid != gid:\n                continue\n            if requester_tid is not None and tid == requester_tid:\n                return False\n            mem = track_memory.get(gid)\n            if mem is None:\n                continue\n            if mem.get("missing", 0) < ID_BOOK_RELEASE_MISSING:\n                return True\n        return False\n\n    def match_persistent_gallery(\n        emb, box, cx, cy, tracker_id, claimed_gids,\n        sim_floor=None\n    ):\n        # Last-resort match against IDs known from previous videos. Bootstraps\n        # a fresh track_memory entry seeded with the persistent gallery views,\n        # so the same person carries the same global ID across videos.\n        if not CROSS_VIDEO_REID or emb is None or not persistent_gallery:\n            return None\n        vis_ratio = get_visible_ratio_in_frame(box, new_w, new_h)\n        if vis_ratio < CROSS_VIDEO_MIN_VISIBLE_RATIO:\n            vlog(\n                f"  [F{frame_count}] Cross-video MISS (low visibility): "\n                f"tracker {tracker_id}, vis={vis_ratio*100:.1f}% "\n                f"< {CROSS_VIDEO_MIN_VISIBLE_RATIO*100:.1f}%"\n            )\n            return None\n\n        slot = pending_new_trackers.setdefault(\n            tracker_id, {"embs": [], "ambiguous_hold": False, "ambiguous_since": None}\n        )\n\n        curr_fu, curr_fl = extract_fashion_pair(frame, box, cache_key=tracker_id)\n        candidates = []\n        for gid, entry in persistent_gallery.items():\n            if gid in claimed_gids or gid in track_memory: \n                continue\n            # TEMP: lock disabled\n            if is_gid_booked(gid, tracker_id):\n                continue\n            views = entry.get("views") or []\n            if not views:\n                continue\n            sim_os = avg_sim_against_views(emb, views)\n            sim_fc = fashion_pair_similarity(\n                curr_fu, curr_fl,\n                entry.get("fashion_upper_init"),\n                entry.get("fashion_lower_init"),\n            )\n            sim = sim_os\n            if sim_fc >= 0.0:\n                a = max(0.0, min(1.0, FASHIONCLIP_CROSS_ALPHA))\n                sim = (1.0 - a) * sim_os + a * sim_fc\n            candidates.append((sim, gid, sim_os, sim_fc))\n\n        if not candidates:\n            return None\n\n        candidates.sort(reverse=True, key=lambda x: x[0])\n        best_sim, best_gid, _best_os, _best_fc = candidates[0]\n        if sim_floor is None:\n            sim_floor = CROSS_VIDEO_REID_SIM_THRESHOLD\n\n        # DEBUG: log every cross-video match attempt with all candidate sims\n        cand_str = ", ".join(\n            f"gid{g}=fused:{s:.3f}/os:{so:.3f}/fc:{sf:.3f}"\n            for s, g, so, sf in candidates[:5]\n        )\n        if best_sim < sim_floor:\n            slot["ambiguous_hold"] = False\n            pending_cross_video_relinks.pop(tracker_id, None)\n            vlog(f"  [F{frame_count}] Cross-video MISS (below floor): "\n                 f"tracker {tracker_id}, best={best_sim:.3f} < {sim_floor} "\n                 f"| vis={vis_ratio*100:.1f}% | candidates: [{cand_str}]")\n            return None\n        if (not CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1\n                and len(candidates) > 1\n                and best_sim - candidates[1][0] < 0.10):\n            if CROSS_VIDEO_DELAY_ON_AMBIGUOUS:\n                slot["ambiguous_hold"] = True\n                if slot.get("ambiguous_since") is None:\n                    slot["ambiguous_since"] = frame_count\n            else:\n                slot["ambiguous_hold"] = False\n            pending_cross_video_relinks.pop(tracker_id, None)\n            vlog(f"  [F{frame_count}] Cross-video MISS (ambiguous): "\n                 f"tracker {tracker_id}, best=gid{best_gid}@{best_sim:.3f} "\n                 f"vs runner-up gid{candidates[1][1]}@{candidates[1][0]:.3f} "\n                 f"(gap {best_sim - candidates[1][0]:.3f} < 0.05) "\n                 f"| vis={vis_ratio*100:.1f}% | candidates: [{cand_str}]")\n            return None\n\n        slot["ambiguous_hold"] = False\n\n        # Require a consecutive-frame streak before confirming cross-video ID.\n        cv_slot = pending_cross_video_relinks.get(tracker_id)\n        if (cv_slot\n                and cv_slot.get("gid") == best_gid\n                and frame_count - cv_slot.get("last_frame", -1) == 1):\n            cv_slot["streak"] += 1\n        else:\n            cv_slot = {"gid": best_gid, "streak": 1}\n        cv_slot["last_frame"] = frame_count\n        cv_slot["last_sim"] = best_sim\n        pending_cross_video_relinks[tracker_id] = cv_slot\n\n        if cv_slot["streak"] < max(1, CROSS_VIDEO_MIN_STREAK):\n            return None\n\n        bw, bh = box[2] - box[0], box[3] - box[1]\n        fx, _ = get_foot(box)\n        seeded_views = ema_update_views(\n            persistent_gallery[best_gid]["views"].copy(), emb)\n        track_memory[best_gid] = {\n            "first_cx": int(cx), "last_cx": int(cx), "last_cy": int(cy),\n            "first_foot_x": int(fx), "last_foot_x": int(fx),\n            "zone_entry_foot_x": None, "last_zone_foot_x": None,\n            "zone_entry_center_x": None, "last_zone_center_x": None,\n            "last_zone_box": None, "last_foot_side": None,\n            "last_event_box": box.copy(), "last_event_foot_x": int(fx),\n            "zone_center_history": [], "zone_width_history": [], "zone_height_history": [],\n            "smooth_box": box, "missing": 0, "seen_frames": 1,\n            "was_in_door_zone": False, "last_in_door_zone": False,\n            "state": None, "last_event": None, "last_w": bw, "last_h": bh,\n            "has_entry_history": True,\n            "zone_samples": 0,\n            "last_det_conf": 1.0,\n            "allow_carryover_exit": True,\n            "last_crop_path": None,\n            "views": seeded_views,\n            "fashion_upper_init": persistent_gallery[best_gid].get("fashion_upper_init"),\n            "fashion_lower_init": persistent_gallery[best_gid].get("fashion_lower_init"),\n            "pre_occ_views": None,\n            "was_occluding": False,\n            # Cross-video carry-over: this ID came from persistent gallery\n            # (previously inside). If now seen outside zone, mark Exit.\n            "pending_carryover_exit_check": True,\n        }\n        id_map[tracker_id] = best_gid\n        pending_cross_video_relinks.pop(tracker_id, None)\n        claimed_gids.add(best_gid)\n        vlog(f"  [F{frame_count}] Cross-video match: tracker {tracker_id} "\n             f"→ gid {best_gid} (sim {best_sim:.2f}, vis {vis_ratio*100:.1f}%)")\n        return best_gid\n\n    def get_pending_avg_embedding(tracker_id, emb):\n        if emb is None:\n            return None, 0\n        slot = pending_new_trackers.setdefault(tracker_id, {"embs": []})\n        slot["embs"].append(emb)\n        if len(slot["embs"]) > NEW_ID_AGG_FRAMES:\n            slot["embs"] = slot["embs"][-NEW_ID_AGG_FRAMES:]\n        count = len(slot["embs"])\n        stacked = torch.stack(slot["embs"], dim=0)\n        avg = stacked.mean(dim=0, keepdim=True)\n        avg = F.normalize(avg, p=2, dim=1).squeeze(0)\n        return avg, count\n\n    def assign_or_relink(frame, box, cx, cy, tracker_id, claimed_gids):\n        nonlocal next_gid\n        vis_ratio = get_visible_ratio_in_frame(box, new_w, new_h)\n\n        # Step 1 — existing mapping\n        if tracker_id in id_map:\n            gid = id_map[tracker_id]\n            if gid in claimed_gids:\n                # This global ID is already used by another detection\n                # in the current frame, so this tracker mapping is stale.\n                del id_map[tracker_id]\n            elif gid in track_memory:\n                mem = track_memory[gid]\n                if mem["missing"] > 0:\n                    # Was absent — verify with multi-view match\n                    emb = extract_embedding(frame, box, cache_key=tracker_id)\n                    if emb is not None:\n                        sim_os = avg_sim_against_views(emb, mem["views"])\n                        curr_fu, curr_fl = extract_fashion_pair(frame, box, cache_key=tracker_id)\n                        sim_fc = fashion_pair_similarity(\n                            curr_fu, curr_fl,\n                            mem.get("fashion_upper_init"),\n                            mem.get("fashion_lower_init"),\n                        )\n                        sim = sim_os\n                        if sim_fc >= 0.0:\n                            a = max(0.0, min(1.0, FASHIONCLIP_WITHIN_ALPHA))\n                            sim = (1.0 - a) * sim_os + a * sim_fc\n                        if WITHIN_VIDEO_VISIBILITY_LOG:\n                            vlog(\n                                f"  [F{frame_count}] ReID verify-missing(vis): "\n                                f"tracker={tracker_id} gid={gid} sim={sim:.3f} "\n                                f"vis={vis_ratio*100:.1f}% th={REID_SIM_THRESHOLD:.3f}"\n                            )\n                        if REID_WITHIN_VIDEO_DEBUG:\n                            vlog(\n                                f"  [F{frame_count}] ReID verify-missing: "\n                                f"tracker={tracker_id} gid={gid} sim={sim:.3f} "\n                                f"th={REID_SIM_THRESHOLD:.3f}"\n                            )\n                        if sim >= REID_SIM_THRESHOLD:\n                            # Confirmed — also add this as a new view\n                            mem["views"] = ema_update_views(mem["views"], emb)\n                            pending_lost_relinks.pop(tracker_id, None)\n                            claimed_gids.add(gid)\n                            return gid\n                    # Failed verify — tracker recycled this id\n                    del id_map[tracker_id]\n                else:\n                    pending_new_trackers.pop(tracker_id, None)\n                    pending_lost_relinks.pop(tracker_id, None)\n                    pending_cross_video_relinks.pop(tracker_id, None)\n                    claimed_gids.add(gid)\n                    return gid   # continuous, trust it\n            else:\n                del id_map[tracker_id]\n\n        # Step 2 — ReID against lost gallery (multi-view)\n        emb = extract_embedding(frame, box, cache_key=tracker_id)\n        best_gid, best_score = None, REID_SIM_THRESHOLD\n\n        if emb is not None:\n            curr_fu, curr_fl = extract_fashion_pair(frame, box, cache_key=tracker_id)\n            local_candidates = []\n            for gid, lost in lost_gallery.items():\n                if gid in claimed_gids:\n                    continue\n                dist = np.linalg.norm(\n                    np.array(lost["center"]) - np.array([cx, cy]))\n                if dist > SPATIAL_GATE:\n                    continue\n                sim_os = avg_sim_against_views(emb, lost["views"])\n                snap = lost.get("track_snapshot", {})\n                sim_fc = fashion_pair_similarity(\n                    curr_fu, curr_fl,\n                    snap.get("fashion_upper_init"),\n                    snap.get("fashion_lower_init"),\n                )\n                sim = sim_os\n                if sim_fc >= 0.0:\n                    a = max(0.0, min(1.0, FASHIONCLIP_WITHIN_ALPHA))\n                    sim = (1.0 - a) * sim_os + a * sim_fc\n                local_candidates.append((sim, gid, dist))\n                if sim > best_score:\n                    best_score, best_gid = sim, gid\n            if REID_WITHIN_VIDEO_DEBUG and local_candidates:\n                local_candidates.sort(reverse=True, key=lambda x: x[0])\n                top = local_candidates[:3]\n                cand_str = ", ".join(\n                    f"gid{g}:sim={s:.3f},dist={d:.1f}" for s, g, d in top\n                )\n                vlog(\n                    f"  [F{frame_count}] ReID lost-gallery candidates: "\n                    f"tracker={tracker_id} [{cand_str}] "\n                    f"(th={REID_SIM_THRESHOLD:.3f})"\n                )\n\n        if best_gid is not None:\n            slot = pending_lost_relinks.get(tracker_id)\n            if slot and slot.get("gid") == best_gid and frame_count - slot.get("last_frame", -1) == 1:\n                slot["streak"] += 1\n            else:\n                slot = {"gid": best_gid, "streak": 1}\n            slot["last_frame"] = frame_count\n            slot["last_sim"] = best_score\n            pending_lost_relinks[tracker_id] = slot\n\n            if REID_WITHIN_VIDEO_DEBUG:\n                vlog(\n                    f"  [F{frame_count}] ReID relink-candidate(lost-gallery): "\n                    f"tracker={tracker_id} gid={best_gid} sim={best_score:.3f} "\n                    f"streak={slot[\'streak\']}/{LOST_RELINK_MIN_STREAK}"\n                )\n\n            if slot["streak"] >= max(1, LOST_RELINK_MIN_STREAK):\n                if REID_WITHIN_VIDEO_DEBUG:\n                    vlog(\n                        f"  [F{frame_count}] ReID relink(lost-gallery): "\n                        f"tracker={tracker_id} -> gid={best_gid} sim={best_score:.3f} "\n                        f"(streak met)"\n                    )\n                id_map[tracker_id] = best_gid\n                # Merge new view into gallery views\n                lost_gallery[best_gid]["views"] = ema_update_views(\n                    lost_gallery[best_gid]["views"], emb)\n                # Restore track memory from gallery\n                track_memory[best_gid] = {\n                    **lost_gallery[best_gid]["track_snapshot"],\n                    "missing": 0,\n                    "views": lost_gallery[best_gid]["views"],\n                    "pre_occ_views": None,       # ← restore excluded fields\n                    "was_occluding": False,      # ← this was the crash\n                }\n                del lost_gallery[best_gid]\n                pending_lost_relinks.pop(tracker_id, None)\n                pending_new_trackers.pop(tracker_id, None)\n                claimed_gids.add(best_gid)\n                if WITHIN_VIDEO_VISIBILITY_LOG:\n                    vlog(\n                        f"  [F{frame_count}] ReID relink(lost-gallery,vis): "\n                        f"tracker={tracker_id} -> gid={best_gid} "\n                        f"sim={best_score:.3f} vis={vis_ratio*100:.1f}%"\n                    )\n                return best_gid\n            return None\n        else:\n            pending_lost_relinks.pop(tracker_id, None)\n\n        # Step 2.5 — cross-video persistent gallery\n        gid = match_persistent_gallery(emb, box, cx, cy, tracker_id, claimed_gids)\n        if gid is not None:\n            pending_new_trackers.pop(tracker_id, None)\n            pending_lost_relinks.pop(tracker_id, None)\n            pending_cross_video_relinks.pop(tracker_id, None)\n            return gid\n        cv_slot = pending_cross_video_relinks.get(tracker_id)\n        if cv_slot and cv_slot.get("streak", 0) < max(1, CROSS_VIDEO_MIN_STREAK):\n            return None\n        slot = pending_new_trackers.get(tracker_id)\n        if slot and slot.get("ambiguous_hold", False):\n            return None\n\n        # Step 3 — delayed assignment for new tracker:\n        # accumulate a few embeddings, try stronger cross-video match using\n        # averaged embedding, then fallback to creating a new ID.\n        avg_emb, agg_count = get_pending_avg_embedding(tracker_id, emb)\n        if avg_emb is not None and agg_count >= NEW_ID_AGG_FRAMES:\n            gid = match_persistent_gallery(\n                avg_emb, box, cx, cy, tracker_id, claimed_gids,\n                sim_floor=NEW_ID_PERSIST_SIM_THRESHOLD,\n            )\n            if gid is not None:\n                pending_new_trackers.pop(tracker_id, None)\n                pending_lost_relinks.pop(tracker_id, None)\n                pending_cross_video_relinks.pop(tracker_id, None)\n                return gid\n            cv_slot = pending_cross_video_relinks.get(tracker_id)\n            if cv_slot and cv_slot.get("streak", 0) < max(1, CROSS_VIDEO_MIN_STREAK):\n                return None\n            slot = pending_new_trackers.get(tracker_id)\n            if slot and slot.get("ambiguous_hold", False):\n                return None\n        elif emb is not None:\n            # Not enough evidence yet; skip assigning a global ID this frame.\n            return None\n\n        # Fallback: create new person ID.\n        new_id = next_gid\n        next_gid += 1\n        id_map[tracker_id] = new_id\n        pending_new_trackers.pop(tracker_id, None)\n        pending_lost_relinks.pop(tracker_id, None)\n        pending_cross_video_relinks.pop(tracker_id, None)\n        claimed_gids.add(new_id)\n        return new_id\n\n    # ------------------------------------------------------------------\n    cap = cv2.VideoCapture(video_path)\n    if not cap.isOpened():\n        vlog(f"  Could not open {video_path}")\n        return []\n\n    fps   = cap.get(cv2.CAP_PROP_FPS) or 25\n    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n    new_w = RESIZE_WIDTH\n    new_h = int(h * RESIZE_WIDTH / w)\n    out_fps = max(1.0, fps / FRAME_SKIP)\n\n    writer = cv2.VideoWriter(\n        tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (new_w, new_h))\n\n    frame_count = 0\n    t0 = time.time()\n\n    while cap.isOpened():\n        ret, frame = cap.read()\n        if not ret:\n            break\n        frame_count += 1\n        if MAX_FRAMES_PER_VIDEO and frame_count > MAX_FRAMES_PER_VIDEO:\n            vlog(f"  Reached MAX_FRAMES_PER_VIDEO={MAX_FRAMES_PER_VIDEO}, stopping early")\n            break\n        if frame_count % FRAME_SKIP != 0:\n            continue\n\n        if frame_count % 200 == 0:\n            vlog(f"  [{frame_count/max(total,1)*100:5.1f}%] "\n                 f"frame {frame_count}/{total}  events={len(events)}  "\n                 f"elapsed={time.time()-t0:.0f}s")\n\n        frame = cv2.resize(frame, (new_w, new_h))\n        clean_frame = frame.copy()\n        SEG_CROP_CACHE.clear()\n        results = yolo_model.track(frame, persist=True,\n                                   tracker=TRACKER_PATH,\n                                   classes=[0], conf=0.3, iou=0.5)\n        current_ids = set()\n\n        zy1 = new_h - ZONE_TOP_OFFSET\n        zy2 = new_h - ZONE_BOTTOM_OFFSET\n        cv2.rectangle(frame, (ZONE_LEFT, zy1), (ZONE_RIGHT, zy2), (0,0,255), 2)\n        cv2.putText(frame, "EXIT", (ZONE_LEFT - 45, zy1 - 8),\n                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)\n        cv2.putText(frame, "ENTRY", (ZONE_RIGHT - 10, zy1 - 8),\n                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,120,0), 2)\n\n        if results[0].boxes.id is not None:\n            raw_boxes = results[0].boxes.xyxy.cpu().numpy()\n            raw_ids   = results[0].boxes.id.cpu().numpy().astype(int)\n            raw_confs = results[0].boxes.conf.cpu().numpy()\n\n            valid = []\n            for b, tid, conf in zip(raw_boxes, raw_ids, raw_confs):\n                if (b[3] - b[1]) < MIN_BOX_HEIGHT:\n                    continue\n                if (b[2] - b[0]) * (b[3] - b[1]) < MIN_BOX_AREA:\n                    continue\n                vis_ratio = get_visible_ratio_in_frame(b, new_w, new_h)\n                if vis_ratio < MIN_VISIBLE_RATIO:\n                    if WITHIN_VIDEO_VISIBILITY_LOG:\n                        vlog(\n                            f"  [F{frame_count}] ReID skip(low visibility): "\n                            f"tracker={tid} vis={vis_ratio*100:.1f}% "\n                            f"< {MIN_VISIBLE_RATIO*100:.1f}%"\n                        )\n                    continue\n                valid.append((b, tid, conf))\n\n            # Occlusion detection\n            occ_tids = set()\n            for i in range(len(valid)):\n                for j in range(i+1, len(valid)):\n                    if box_iou(valid[i][0], valid[j][0]) > OCCLUSION_IOU_THRESHOLD:\n                        occ_tids.add(valid[i][1])\n                        occ_tids.add(valid[j][1])\n\n            # ID assignment\n            assignments = []\n            claimed_gids = set()\n            for box, tid, conf in valid:\n                cx, cy = get_centroid(box)\n                occ = tid in occ_tids\n\n                # Lock during occlusion if continuously tracked\n                if (occ and tid in id_map\n                        and id_map[tid] in track_memory\n                        and track_memory[id_map[tid]]["missing"] == 0\n                        and id_map[tid] not in claimed_gids):\n                    gid = id_map[tid]\n                    pending_tracker_state.pop(tid, None)\n                    claimed_gids.add(gid)\n                else:\n                    gid = assign_or_relink(frame, box, cx, cy, tid, claimed_gids)\n                    if gid is None:\n                        update_pending_tracker_state(tid, box, new_h)\n                        continue\n\n                assignments.append((gid, box, tid, occ, float(conf)))\n                pending_tracker_state.pop(tid, None)\n\n            # Update tracks\n            for gid, box, tid, occ, det_conf in assignments:\n                raw_box = box.copy()\n                cx, cy = get_centroid(box)\n                bw, bh = box[2]-box[0], box[3]-box[1]\n                current_ids.add(gid)\n\n                if gid not in track_memory:\n                    init_emb = extract_embedding(frame, box, cache_key=tid)\n                    f_u_init, f_l_init = extract_fashion_pair(frame, box, cache_key=tid)\n                    fx, _ = get_foot(box)\n                    pre = pop_pending_tracker_state(tid)\n                    track_memory[gid] = {\n                        "first_cx": cx, "last_cx": cx, "last_cy": cy,\n                        "first_foot_x": fx, "last_foot_x": fx,\n                        "zone_entry_foot_x": (pre.get("zone_entry_foot_x") if pre else None), "last_zone_foot_x": (pre.get("last_zone_foot_x") if pre else None),\n                        "zone_entry_center_x": (pre.get("zone_entry_center_x") if pre else None), "last_zone_center_x": (pre.get("last_zone_center_x") if pre else None),\n                        "last_zone_box": (pre.get("last_zone_box") if pre else None), "last_foot_side": (pre.get("last_foot_side") if pre else None),\n                        "last_event_box": box.copy(), "last_event_foot_x": fx,\n                        "zone_center_history": (pre.get("zone_center_history", []) if pre else []),\n                        "zone_width_history": (pre.get("zone_width_history", []) if pre else []),\n                        "zone_height_history": (pre.get("zone_height_history", []) if pre else []),\n                        "smooth_box": box, "missing": 0, "seen_frames": 1,\n                        "was_in_door_zone": (pre.get("was_in_door_zone", False) if pre else False),\n                        "last_in_door_zone": (pre.get("last_in_door_zone", False) if pre else False),\n                        "state": None, "last_event": None, "last_w": bw, "last_h": bh,\n                        "has_entry_history": False,\n                        "zone_samples": (pre.get("samples", 0) if pre else 0),\n                        "last_det_conf": 1.0,\n                        "allow_carryover_exit": False,\n                        "last_crop_path": None,\n                        "views": [init_emb] if init_emb is not None else [],\n                        "fashion_upper_init": f_u_init,\n                        "fashion_lower_init": f_l_init,\n                        "pre_occ_views": None,\n                        "was_occluding": False,\n                        "pending_carryover_exit_check": False,\n                    }\n                    if SAVE_REID_DEBUG_CROPS and init_emb is not None:\n                        save_reid_view_crop(clean_frame, box, gid, vname, frame_count, "init_view")\n\n                mem = track_memory[gid]\n                was_occ = mem["was_occluding"]\n                event_box_ok = is_event_box_stable(mem, raw_box)\n\n                # Snapshot views just before occlusion\n                if not was_occ and occ:\n                    mem["pre_occ_views"] = mem["views"].copy()\n\n                # Restore clean views right after occlusion ends\n                if was_occ and not occ:\n                    if mem["pre_occ_views"] is not None:\n                        mem["views"] = mem["pre_occ_views"]\n                        mem["pre_occ_views"] = None\n\n                mem["was_occluding"] = occ\n                mem["smooth_box"] = smooth_box(mem["smooth_box"], box)\n                box = mem["smooth_box"]\n                event_box = raw_box if event_box_ok else mem.get("last_event_box", box)\n                fx, fy = get_foot(event_box)\n                mem["last_cx"] = cx\n                mem["last_cy"] = cy\n                if event_box_ok:\n                    mem["last_foot_x"] = fx\n                mem["last_w"]  = bw\n                mem["last_h"]  = bh\n                mem["missing"] = 0\n                mem["seen_frames"] += 1\n                mem["last_det_conf"] = det_conf\n\n                # Accumulate diverse views on clean frames\n                if not occ and mem["seen_frames"] % 8 == 0:\n                    new_emb, seg_used_os, seg_reason_os = extract_embedding(\n                        frame, box, return_seg_used=True, cache_key=tid\n                    )\n                    if new_emb is not None:\n                        sim_init_os = cosine_sim(new_emb, mem["views"][0]) if mem.get("views") else -1.0\n                        sim_views_os = avg_sim_against_views(new_emb, mem.get("views", []))\n                        curr_fu, curr_fl, seg_used_fc, seg_reason_fc = extract_fashion_pair(\n                            frame, box, return_seg_used=True, cache_key=tid\n                        )\n                        sim_fc = fashion_pair_similarity(\n                            curr_fu, curr_fl,\n                            mem.get("fashion_upper_init"),\n                            mem.get("fashion_lower_init"),\n                        )\n                        fused = sim_views_os\n                        if sim_fc >= 0.0:\n                            a = max(0.0, min(1.0, FASHIONCLIP_WITHIN_ALPHA))\n                            fused = (1.0 - a) * sim_views_os + a * sim_fc\n                        if REID_UPDATE_VIEW_LOG:\n                            vlog(\n                                f"  [F{frame_count}] ReID update-view score: gid={gid} "\n                                f"os_init={sim_init_os:.3f} os_views={sim_views_os:.3f} "\n                                f"fc={sim_fc:.3f} fused={fused:.3f} "\n                                f"th={REID_UPDATE_VIEW_MIN_SIM:.3f} "\n                                f"th_os={REID_UPDATE_VIEW_MIN_OS:.3f} "\n                                f"th_fc={REID_UPDATE_VIEW_MIN_FC:.3f} "\n                                f"seg_os={\'Y\' if seg_used_os else \'N\'} "\n                                f"seg_fc={\'Y\' if seg_used_fc else \'N\'} "\n                                f"seg_reason_os={seg_reason_os} "\n                                f"seg_reason_fc={seg_reason_fc}"\n                            )\n                        if REID_SEG_STRICT and (not seg_used_os or not seg_used_fc):\n                            if REID_UPDATE_VIEW_LOG:\n                                vlog(\n                                    f"  [F{frame_count}] ReID update-view: gid={gid} blocked "\n                                    f"(seg_strict os={seg_reason_os} fc={seg_reason_fc})"\n                                )\n                        elif (fused >= REID_UPDATE_VIEW_MIN_SIM\n                                and sim_views_os >= REID_UPDATE_VIEW_MIN_OS\n                                and sim_fc >= REID_UPDATE_VIEW_MIN_FC):\n                            mem["views"] = ema_update_views(mem["views"], new_emb)\n                            if SAVE_REID_DEBUG_CROPS:\n                                save_reid_view_crop(clean_frame, box, gid, vname, frame_count, "update_view")\n                            if REID_UPDATE_VIEW_LOG:\n                                vlog(f"  [F{frame_count}] ReID update-view: gid={gid} accepted")\n                        else:\n                            if REID_UPDATE_VIEW_LOG:\n                                vlog(f"  [F{frame_count}] ReID update-view: gid={gid} blocked")\n\n                # Save a crop for every detection of this gid (these crops\n                # are the source for offline embedding-based matching).\n                crop_path = save_crop(clean_frame, box, gid, vname, frame_count)\n                if crop_path is not None:\n                    if det_conf >= MIN_EVENT_DET_CONF:\n                        mem["last_crop_path"] = crop_path\n\n                # Use smoothed box center for zone-state updates every frame.\n                # This keeps state progression even when raw box stability gate fails.\n                state_box = box\n                in_door = is_in_door_zone_center(state_box, new_h)\n                prev_in_door = mem["last_in_door_zone"]\n                if event_box_ok:\n                    mem["last_event_box"] = raw_box.copy()\n                    mem["last_event_foot_x"] = fx\n\n                mem["last_in_door_zone"] = in_door\n                scx, _ = get_centroid(state_box)\n                sfx, _ = get_foot(state_box)\n                if in_door:\n                    mem["was_in_door_zone"] = True\n                    mem["zone_samples"] = mem.get("zone_samples", 0) + 1\n                    if mem.get("zone_entry_center_x") is None:\n                        mem["zone_entry_center_x"] = scx\n                    mem["last_zone_center_x"] = scx\n                    if mem.get("zone_entry_foot_x") is None:\n                        mem["zone_entry_foot_x"] = sfx\n                    mem["last_zone_foot_x"] = sfx\n                    mem["last_zone_box"] = state_box.copy()\n                    mem["last_foot_side"] = "INSIDE"\n                    append_event_history(mem, state_box)\n                else:\n                    mem["last_foot_side"] = foot_zone_status(state_box, new_h)\n\n                # If this person was matched from previous videos\' persistent\n                # gallery and is now already outside the zone, count as Exit.\n                if mem.get("pending_carryover_exit_check", False):\n                    if (not in_door\n                            and mem.get("seen_frames", 0) >= MIN_CARRYOVER_SEEN_FRAMES\n                            and mem.get("last_det_conf", 0.0) >= MIN_EVENT_DET_CONF):\n                        did_emit = emit_event(gid, "Exit", frame_count, crop_path)\n                        if EVENT_DEBUG and did_emit:\n                            print(f"  [F{frame_count}] gid {gid} EVENT carry-over: Exit")\n                    mem["pending_carryover_exit_check"] = False\n\n                if EVENT_DEBUG and (in_door != prev_in_door):\n                    print(\n                        f"  [F{frame_count}] gid {gid} zone {\'IN\' if in_door else \'OUT\'} "\n                        f"(scx={scx:.1f}, zone_start={mem.get(\'zone_entry_center_x\')}, "\n                        f"last_zone={mem.get(\'last_zone_center_x\')})"\n                    )\n\n                if prev_in_door and not in_door:\n                    event_name = classify_door_event(mem, state_box, new_h)\n                    if event_name is not None:\n                        if EVENT_DEBUG:\n                            print(f"  [F{frame_count}] gid {gid} EVENT on leave: {event_name}")\n                        emit_event(gid, event_name, frame_count, crop_path)\n\n                # Draw\n                x1, y1, x2, y2 = map(int, box)\n                color = ((255,120,0) if gid in event_display and\n                         event_display[gid]["event"] == "ENTRY" else\n                         (0,0,255) if gid in event_display and\n                         event_display[gid]["event"] == "EXIT" else\n                         (0,255,0))\n                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)\n                n_views = len(mem["views"])\n                label = f"ID {gid} [{n_views}v]"\n                if gid in event_display:\n                    label += f" | {event_display[gid][\'event\']}"\n                if occ:\n                    label += " [L]"\n                (tw, th), _ = cv2.getTextSize(\n                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)\n                cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), (0,0,0), -1)\n                cv2.putText(frame, label, (x1, y1-5),\n                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)\n\n        # Disappeared tracks\n        for gid in list(track_memory.keys()):\n            if gid not in current_ids:\n                mem = track_memory[gid]\n                mem["missing"] += 1\n\n                if mem["missing"] == 1:\n                    event_box = mem.get("last_event_box", mem["smooth_box"])\n                    event_name = classify_door_event(\n                        mem, event_box, new_h, disappeared=True)\n                    if event_name is not None:\n                        if EVENT_DEBUG:\n                            print(f"  [F{frame_count}] gid {gid} EVENT on disappear: {event_name}")\n                        emit_event(\n                            gid,\n                            event_name,\n                            frame_count,\n                            mem.get("last_crop_path"),\n                        )\n\n                if (mem["missing"] == 1\n                        and mem["seen_frames"] >= MIN_SEEN_FOR_GALLERY\n                        and mem["views"]):\n                    keep_until_end = not is_at_disappear_boundary(\n                        mem["smooth_box"], new_w, new_h)\n                    lost_gallery[gid] = {\n                        "center": (mem["last_cx"], mem["last_cy"]),\n                        "frame":  frame_count,\n                        "views":  mem["views"].copy(),\n                        "keep_until_end": keep_until_end,\n                        # Snapshot full track state so we can restore it on relink\n                        "track_snapshot": {\n                            k: v for k, v in mem.items()\n                            if k not in ("missing", "views",\n                                         "pre_occ_views", "was_occluding")\n                        }\n                    }\n\n                # can_expire = is_at_disappear_boundary(mem["smooth_box"], new_w, new_h)\n                # if can_expire and mem["missing"] > REID_MAX_GAP:\n                #     del track_memory[gid]\n                #     lost_gallery.pop(gid, None)\n                #     for k in [k for k, v in id_map.items() if v == gid]:\n                #         del id_map[k]\n\n        # for gid in list(lost_gallery.keys()):\n        #     if (not lost_gallery[gid].get("keep_until_end", False)\n        #             and frame_count - lost_gallery[gid]["frame"] > REID_MAX_GAP):\n        #         lost_gallery.pop(gid, None)\n\n        for gid in list(event_display.keys()):\n            event_display[gid]["ttl"] -= 1\n            if event_display[gid]["ttl"] <= 0:\n                del event_display[gid]\n\n        cv2.putText(frame, f"Events: {len(events)}", (10, new_h-10),\n                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)\n        cv2.putText(frame, vname, (10, 30),\n                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)\n        writer.write(frame)\n\n    unresolved_ambiguous = []\n    for tid, slot in pending_new_trackers.items():\n        if slot.get("ambiguous_hold", False) and tid not in id_map:\n            unresolved_ambiguous.append((tid, slot.get("ambiguous_since")))\n    if unresolved_ambiguous:\n        for tid, since in unresolved_ambiguous:\n            vlog(\n                f"  [FLAG] Unresolved ambiguous ID until video end: "\n                f"tracker={tid}, since_frame={since}"\n            )\n\n    cap.release()\n    writer.release()\n    shutil.copy2(tmp_vid, out_vid)\n    os.remove(tmp_vid)\n    with open(out_json, "w") as f:\n        json.dump(events, f, indent=2)\n\n    # Push final views into the cross-video persistent gallery so the next\n    # video can match the same people back to the same gids.\n    if CROSS_VIDEO_REID:\n        def _commit_views(gid, views, fashion_upper_init=None, fashion_lower_init=None):\n            if not views:\n                return\n            if gid not in persistent_gallery:\n                persistent_gallery[gid] = {"views": []}\n            for v in views:\n                persistent_gallery[gid]["views"] = maybe_add_view(\n                    persistent_gallery[gid]["views"], v)\n            if fashion_upper_init is not None:\n                persistent_gallery[gid]["fashion_upper_init"] = fashion_upper_init\n            if fashion_lower_init is not None:\n                persistent_gallery[gid]["fashion_lower_init"] = fashion_lower_init\n\n        for gid, mem in track_memory.items():\n            if (mem.get("seen_frames", 0) >= MIN_SEEN_FOR_GALLERY\n                    and mem.get("last_event") == "Entry"):\n                _commit_views(\n                    gid,\n                    mem.get("views"),\n                    mem.get("fashion_upper_init"),\n                    mem.get("fashion_lower_init"),\n                )\n        for gid, lost in lost_gallery.items():\n            last_event = (lost.get("track_snapshot") or {}).get("last_event")\n            if last_event == "Entry":\n                snap = lost.get("track_snapshot", {})\n                _commit_views(\n                    gid,\n                    lost.get("views"),\n                    snap.get("fashion_upper_init"),\n                    snap.get("fashion_lower_init"),\n                )\n\n    cross_state["next_gid"] = next_gid\n\n    vlog(f"  Done — {len(events)} events, {frame_count} frames, "\n         f"{time.time()-t0:.1f}s")\n    if CROSS_VIDEO_REID:\n        vlog(f"  Persistent gallery now: {sorted(persistent_gallery.keys())}")\n    return events\n\n\n# ==============================\n# BATCH RUNNER\n# ==============================\n\ndef find_videos(folder):\n    exts = ("*.mp4","*.MP4","*.avi","*.AVI","*.mov","*.MOV","*.mkv","*.MKV")\n    paths = []\n    for ext in exts:\n        paths.extend(glob.glob(os.path.join(folder, ext)))\n    return sorted(paths)\n\n\ndef clear_output_folder(output_dir):\n    if not os.path.isdir(output_dir):\n        os.makedirs(output_dir, exist_ok=True)\n        return\n    for name in os.listdir(output_dir):\n        path = os.path.join(output_dir, name)\n        if os.path.isdir(path):\n            shutil.rmtree(path, ignore_errors=True)\n        else:\n            try:\n                os.remove(path)\n            except FileNotFoundError:\n                pass\n\nif __name__ == "__main__":\n    clear_output_folder(OUTPUT_BASE)\n    videos = find_videos(VIDEO_FOLDER)\n    if not videos:\n        print(f"No videos found in {VIDEO_FOLDER}")\n        exit(1)\n\n    print(f"Found {len(videos)} video(s):")\n    for i, v in enumerate(videos, 1):\n        print(f"  {i}. {os.path.basename(v)}")\n\n    all_events = {}\n    cross_state = {"next_gid": 1, "persistent_gallery": {}}\n    for vp in videos:\n        # if \'3.mp4\' not in vp:\n        #     continue\n        print(vp)\n        #if vp != \'/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/5.MP4\' and vp != \'/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/6.MP4\' and vp != \'/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/7.MP4\' and vp != \'/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/8.MP4\' :\n        # if vp != \'/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/2.mp4\' and vp != \'/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/1.mp4\' and vp != \'/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/5.MP4\':    \n        #     continue\n        evts = process_video(vp, OUTPUT_BASE, cross_state)\n        if evts:\n            all_events[os.path.basename(vp)] = evts\n\n    summary = os.path.join(OUTPUT_BASE, "all_events_summary.json")\n    with open(summary, "w") as f:\n        json.dump(all_events, f, indent=2)\n\n    total = sum(len(v) for v in all_events.values())\n    print(f"\\nBatch complete — {len(videos)} videos, {total} total events")\n    print(f"Summary: {summary}")\n'
_ENTRY_SOURCE = _ENTRY_SOURCE.replace('os.path.join(BASE_DIR, "yolo26s.pt")', 'os.path.join(BASE_DIR, "models", "detection", "yolo26s.pt")')
_ENTRY_SOURCE = _ENTRY_SOURCE.replace('os.path.join(BASE_DIR, "custom_tracker.yaml")', 'os.path.join(BASE_DIR, "models", "tracker", "custom_tracker.yaml")')
_ENTRY_SOURCE = _ENTRY_SOURCE.replace('os.path.join(BASE_DIR, "yoloe-11l-seg.pt")', 'os.path.join(BASE_DIR, "models", "segmentation", "yoloe-11l-seg.pt")')
_ENTRY_SOURCE = _ENTRY_SOURCE.replace('os.path.join(BASE_DIR, "models", "osnet_x1_0_msmt17.pt")', 'os.path.join(BASE_DIR, "models", "reid", "osnet_x1_0_msmt17.pt")')
_ENTRY_NS = {"__file__": __file__, "__name__": "_integrated_entry"}
exec(_ENTRY_SOURCE, _ENTRY_NS)
IntegratedEntry = _IntegratedModuleProxy(_ENTRY_NS)
REID_FASHION_DEBUG_DIR = None


def disable_integrated_entry_crops():
    IntegratedEntry.CROPS_DIR = None

    def _save_crop_disabled(*args, **kwargs):
        return None

    IntegratedEntry.save_crop = _save_crop_disabled


disable_integrated_entry_crops()
IntegratedEntry.REID_SEG_STRICT = False
IntegratedEntry.REID_UPDATE_VIEW_MIN_FC = 0.60
IntegratedEntry.LAST_FASHIONCLIP_ERROR = "uninitialized"
IntegratedEntry.LAST_FASHION_PAIR_STATUS = "uninitialized"


def _fashionclip_repo_name(model_name):
    alias_map = {
        "fashion-clip": "patrickjohncyh/fashion-clip",
    }
    raw = str(model_name or "").strip()
    return alias_map.get(raw, raw)


def _load_integrated_fashionclip_model_compat(model_name, auth_token=None):
    if CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError("transformers_clip_unavailable")
    repo_name = _fashionclip_repo_name(model_name)
    model_kwargs = {}
    processor_kwargs = {}
    if auth_token:
        model_kwargs["token"] = auth_token
        processor_kwargs["token"] = auth_token
    try:
        clip_model = CLIPModel.from_pretrained(
            repo_name,
            local_files_only=True,
            **model_kwargs,
        )
        clip_processor = CLIPProcessor.from_pretrained(
            repo_name,
            local_files_only=True,
            **processor_kwargs,
        )
    except Exception:
        clip_model = CLIPModel.from_pretrained(repo_name, **model_kwargs)
        clip_processor = CLIPProcessor.from_pretrained(repo_name, **processor_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)
    clip_model.eval()
    return SimpleNamespace(
        model_name=repo_name,
        model=clip_model,
        preprocess=clip_processor,
        device=device,
    )


def ensure_integrated_fashionclip_model():
    if not bool(getattr(IntegratedEntry, "FASHIONCLIP_ENABLE", False)):
        return False
    if getattr(IntegratedEntry, "fashionclip_model", None) is not None:
        return True
    model_name = getattr(IntegratedEntry, "FASHIONCLIP_MODEL_NAME", "fashion-clip")
    auth_token = getattr(IntegratedEntry, "FASHIONCLIP_AUTH_TOKEN", None)
    try:
        IntegratedEntry.fashionclip_model = _load_integrated_fashionclip_model_compat(
            model_name,
            auth_token=auth_token,
        )
        print(f"FashionCLIP recovered via compatibility loader: {model_name}")
        return True
    except Exception as e:
        print(f"Warning: FashionCLIP compatibility loader failed for {model_name}: {e}")
        return False


ensure_integrated_fashionclip_model()


@torch.no_grad()
def _extract_integrated_fashionclip_embedding_logged(crop_bgr, part_label="unknown"):
    def _coerce_embedding_tensor(value):
        if value is None:
            return None
        if hasattr(value, "image_embeds") and getattr(value, "image_embeds") is not None:
            return _coerce_embedding_tensor(getattr(value, "image_embeds"))
        elif hasattr(value, "pooler_output") and getattr(value, "pooler_output") is not None:
            return _coerce_embedding_tensor(getattr(value, "pooler_output"))
        elif hasattr(value, "last_hidden_state") and getattr(value, "last_hidden_state") is not None:
            hidden = getattr(value, "last_hidden_state")
            if isinstance(hidden, torch.Tensor) and hidden.ndim >= 2:
                return _coerce_embedding_tensor(hidden[:, 0, ...] if hidden.ndim > 2 else hidden)
            return _coerce_embedding_tensor(hidden)
        elif isinstance(value, np.ndarray):
            t = torch.from_numpy(value).float()
        elif isinstance(value, torch.Tensor):
            t = value.detach().float()
        else:
            try:
                t = torch.tensor(value, dtype=torch.float32)
            except Exception:
                return None
        if t.numel() == 0:
            return None
        if t.ndim == 0:
            return None
        if t.ndim > 1:
            t = t.reshape(t.shape[0], -1) if t.shape[0] == 1 else t.reshape(-1)
        if t.ndim == 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.ndim != 1:
            t = t.view(-1)
        return t

    def _direct_model_embedding(pil_image):
        model_obj = getattr(IntegratedEntry, "fashionclip_model", None)
        preprocess = getattr(model_obj, "preprocess", None)
        clip_model = getattr(model_obj, "model", None)
        device = getattr(model_obj, "device", torch.device("cpu"))
        if preprocess is None or clip_model is None:
            raise RuntimeError("fashionclip_missing_preprocess_or_model")
        batch = preprocess(images=[pil_image], return_tensors="pt")
        batch = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }
        with torch.no_grad():
            if hasattr(clip_model, "get_image_features"):
                raw = clip_model.get_image_features(**batch)
            else:
                raw = clip_model(**batch)
        return raw

    if getattr(IntegratedEntry, "fashionclip_model", None) is None:
        IntegratedEntry.LAST_FASHIONCLIP_ERROR = f"{part_label}:model_unavailable"
        return None
    if crop_bgr is None:
        IntegratedEntry.LAST_FASHIONCLIP_ERROR = f"{part_label}:crop_none"
        return None
    if getattr(crop_bgr, "size", 0) == 0:
        IntegratedEntry.LAST_FASHIONCLIP_ERROR = f"{part_label}:crop_empty"
        return None
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        raw = _direct_model_embedding(pil)
        if raw is None:
            IntegratedEntry.LAST_FASHIONCLIP_ERROR = f"{part_label}:encode_none"
            return None
        t = _coerce_embedding_tensor(raw)
        if t is None:
            IntegratedEntry.LAST_FASHIONCLIP_ERROR = (
                f"{part_label}:unsupported_output:{type(raw).__name__}"
            )
            return None
        if t.numel() == 0:
            IntegratedEntry.LAST_FASHIONCLIP_ERROR = f"{part_label}:empty_tensor"
            return None
        IntegratedEntry.LAST_FASHIONCLIP_ERROR = f"{part_label}:ok"
        return F.normalize(t.unsqueeze(0), p=2, dim=1).squeeze(0)
    except Exception as e:
        IntegratedEntry.LAST_FASHIONCLIP_ERROR = f"{part_label}:{type(e).__name__}:{e}"
        return None


IntegratedEntry.extract_fashionclip_embedding = _extract_integrated_fashionclip_embedding_logged
_ENTRY_NS["extract_fashionclip_embedding"] = _extract_integrated_fashionclip_embedding_logged


def _extract_integrated_fashion_pair_default(frame, box, return_seg_used=False, cache_key=None):
    try:
        crop, seg_used, seg_reason = IntegratedEntry._crop_with_person_segmentation(
            frame, box, return_used=True, cache_key=cache_key
        )
    except Exception:
        crop, seg_used, seg_reason = None, False, "helper_error"
    if crop is None or getattr(crop, "size", 0) == 0:
        return (None, None, seg_used, seg_reason) if return_seg_used else (None, None)
    h = crop.shape[0]
    if h < 24:
        return (None, None, seg_used, "small_crop") if return_seg_used else (None, None)
    split = int(h * 0.55)
    upper = crop[:split, :]
    lower = crop[split:, :]
    upper_emb = IntegratedEntry.extract_fashionclip_embedding(upper, part_label="upper")
    upper_reason = getattr(IntegratedEntry, "LAST_FASHIONCLIP_ERROR", "upper:unknown")
    lower_emb = IntegratedEntry.extract_fashionclip_embedding(lower, part_label="lower")
    lower_reason = getattr(IntegratedEntry, "LAST_FASHIONCLIP_ERROR", "lower:unknown")
    pair_status = f"{upper_reason}; {lower_reason}"
    IntegratedEntry.LAST_FASHION_PAIR_STATUS = pair_status
    if return_seg_used:
        if upper_emb is None and lower_emb is None:
            return upper_emb, lower_emb, seg_used, f"{seg_reason}|{pair_status}"
        if upper_emb is None or lower_emb is None:
            return upper_emb, lower_emb, seg_used, f"{seg_reason}|partial|{pair_status}"
        return upper_emb, lower_emb, seg_used, f"{seg_reason}|{pair_status}"
    return upper_emb, lower_emb


def _integrated_fashion_pair_similarity_default(curr_u, curr_l, ref_u, ref_l):
    weighted_num = 0.0
    weighted_den = 0.0
    if curr_u is not None and ref_u is not None:
        weighted_num += float(IntegratedEntry.FASHIONCLIP_UPPER_WEIGHT) * float(IntegratedEntry.cosine_sim(curr_u, ref_u))
        weighted_den += float(IntegratedEntry.FASHIONCLIP_UPPER_WEIGHT)
    if curr_l is not None and ref_l is not None:
        weighted_num += float(IntegratedEntry.FASHIONCLIP_LOWER_WEIGHT) * float(IntegratedEntry.cosine_sim(curr_l, ref_l))
        weighted_den += float(IntegratedEntry.FASHIONCLIP_LOWER_WEIGHT)
    if weighted_den <= 0.0:
        return -1.0
    return weighted_num / weighted_den


IntegratedEntry.extract_fashion_pair = _extract_integrated_fashion_pair_default
IntegratedEntry.fashion_pair_similarity = _integrated_fashion_pair_similarity_default
_ENTRY_NS["extract_fashion_pair"] = _extract_integrated_fashion_pair_default
_ENTRY_NS["fashion_pair_similarity"] = _integrated_fashion_pair_similarity_default


def save_reid_fashion_debug_crops(frame, box, gid, vname, frame_count, tag, base_dir=None):
    out_root = base_dir or REID_FASHION_DEBUG_DIR
    if not out_root:
        return []
    os.makedirs(out_root, exist_ok=True)
    gid_dir = os.path.join(out_root, f"ID{int(gid)}")
    os.makedirs(gid_dir, exist_ok=True)

    try:
        seg_crop, _seg_used, seg_reason = IntegratedEntry._crop_with_person_segmentation(
            frame, box, return_used=True
        )
    except Exception:
        seg_crop, seg_reason = None, "helper_error"

    written = []
    base_name = f"{vname}_F{int(frame_count):06d}_{tag}"

    if seg_crop is not None and getattr(seg_crop, "size", 0) > 0 and seg_reason == "ok":
        seg_path = os.path.join(gid_dir, f"{base_name}_seg_{seg_reason}.jpg")
        cv2.imwrite(seg_path, seg_crop)
        written.append(seg_path)

    return written


_original_integrated_save_reid_view_crop = IntegratedEntry.save_reid_view_crop


def _save_reid_view_crop_with_fashion(frame, box, gid, vname, frame_count, tag):
    path = _original_integrated_save_reid_view_crop(frame, box, gid, vname, frame_count, tag)
    try:
        save_reid_fashion_debug_crops(frame, box, gid, vname, frame_count, tag)
    except Exception:
        pass
    return path


IntegratedEntry.save_reid_view_crop = _save_reid_view_crop_with_fashion
_ENTRY_NS["save_reid_view_crop"] = _save_reid_view_crop_with_fashion


def sync_persistent_gallery_from_reid_views(reid_views_dir, cross_state, log_fn=None):
    def log(msg):
        if log_fn is not None:
            log_fn(msg)
    if not reid_views_dir or not os.path.isdir(reid_views_dir):
        return 0
    persistent_gallery = cross_state.setdefault("persistent_gallery", {})
    loaded_paths = cross_state.setdefault("persistent_gallery_view_paths", {})
    added = 0
    for name in sorted(os.listdir(reid_views_dir)):
        if not name.startswith("ID"):
            continue
        gid_str = name[2:]
        if not gid_str.isdigit():
            continue
        gid = int(gid_str)
        gid_dir = os.path.join(reid_views_dir, name)
        if not os.path.isdir(gid_dir):
            continue
        seen_paths = loaded_paths.setdefault(gid, set())
        for image_name in sorted(os.listdir(gid_dir)):
            if not image_name.lower().endswith(".jpg"):
                continue
            image_path = os.path.join(gid_dir, image_name)
            if image_path in seen_paths:
                continue
            img = cv2.imread(image_path)
            if img is None or img.size == 0:
                continue
            h, w = img.shape[:2]
            log(
                f"[Detect] gallery embed start gid={gid} type=osnet "
                f"image={image_name}"
            )
            emb = IntegratedEntry.extract_embedding(img, [0, 0, w, h])
            if emb is None:
                log(
                    f"[Detect] gallery embed skip gid={gid} type=osnet "
                    f"image={image_name} reason=no_embedding"
                )
                continue
            if gid not in persistent_gallery:
                persistent_gallery[gid] = {"views": []}
            persistent_gallery[gid]["views"] = IntegratedEntry.maybe_add_view(
                persistent_gallery[gid].get("views", []),
                emb,
            )
            seen_paths.add(image_path)
            added += 1
            log(
                f"[Detect] gallery embed done gid={gid} type=osnet "
                f"image={image_name}"
            )

    # Hydrate FashionCLIP reference embeddings from segmented debug crops if available.
    fashion_root = REID_FASHION_DEBUG_DIR
    if not fashion_root and reid_views_dir:
        candidate = os.path.join(os.path.dirname(reid_views_dir), "reid_fashion_views")
        if os.path.isdir(candidate):
            fashion_root = candidate
    if fashion_root and os.path.isdir(fashion_root):
        for name in sorted(os.listdir(fashion_root)):
            if not name.startswith("ID"):
                continue
            gid_str = name[2:]
            if not gid_str.isdigit():
                continue
            gid = int(gid_str)
            gid_dir = os.path.join(fashion_root, name)
            if not os.path.isdir(gid_dir):
                continue
            if gid not in persistent_gallery:
                persistent_gallery[gid] = {"views": []}
            if (
                persistent_gallery[gid].get("fashion_upper_init") is not None
                and persistent_gallery[gid].get("fashion_lower_init") is not None
            ):
                continue
            fashion_images = sorted(
                image_name for image_name in os.listdir(gid_dir)
                if image_name.lower().endswith(".jpg")
            )
            for image_name in fashion_images:
                image_path = os.path.join(gid_dir, image_name)
                img = cv2.imread(image_path)
                if img is None or img.size == 0:
                    continue
                h, w = img.shape[:2]
                log(
                    f"[Detect] gallery embed start gid={gid} type=fashion_pair "
                    f"image={image_name} parts=upper,lower"
                )
                f_u, f_l = IntegratedEntry.extract_fashion_pair(img, [0, 0, w, h])
                if f_u is None and f_l is None:
                    pair_status = getattr(IntegratedEntry, "LAST_FASHION_PAIR_STATUS", "unknown")
                    log(
                        f"[Detect] gallery embed skip gid={gid} type=fashion_pair "
                        f"image={image_name} reason=no_embedding status={pair_status}"
                    )
                    continue
                if f_u is not None:
                    persistent_gallery[gid]["fashion_upper_init"] = f_u
                if f_l is not None:
                    persistent_gallery[gid]["fashion_lower_init"] = f_l
                pair_status = getattr(IntegratedEntry, "LAST_FASHION_PAIR_STATUS", "unknown")
                log(
                    f"[Detect] gallery embed done gid={gid} type=fashion_pair "
                    f"image={image_name} upper={'Y' if f_u is not None else 'N'} "
                    f"lower={'Y' if f_l is not None else 'N'} status={pair_status}"
                )
                break
    return added


try:
    import torchreid
except Exception:
    torchreid = None
try:
    from transformers import (
        AutoImageProcessor,
        Dinov2Model,
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        AutoModelForImageTextToText,
    )
except Exception:
    AutoImageProcessor = None
    Dinov2Model = None
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None
    AutoModelForImageTextToText = None
# /Users/fredjackyong/Documents/kebunapp/theft_detection/Output_kiosk/scanned_items/ID1/item_001/sample_F000233_bottle_T-1.jpg
# /Users/fredjackyong/Documents/kebunapp/theft_detection/Output_kiosk/scanned_items/ID1/item_001/sample_F000234_bottle_T-1.jpg
# These two image kacau la

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SESSIONS_ROOT = os.path.join(BASE_DIR, "session")
DEFAULT_SESSION_DIR = os.path.join(BASE_DIR, "session", "session_1")
ENTRY_VIDEO_FOLDER = os.environ.get(
    "DETECT_ENTRY_VIDEO_FOLDER",
    DEFAULT_SESSION_DIR if os.path.isdir(DEFAULT_SESSION_DIR) else os.path.join(BASE_DIR, "new_video"),
)
VIDEO_FOLDER = os.environ.get(
    "DETECT_KIOSK_VIDEO_FOLDER",
    DEFAULT_SESSION_DIR if os.path.isdir(DEFAULT_SESSION_DIR) else os.path.join(BASE_DIR, "new_video", "kiosk"),
)
OUTPUT_BASE = os.environ.get(
    "DETECT_OUTPUT_BASE",
    os.path.join(BASE_DIR, "Output_detect"),
)
DETECT_CLEAR_OUTPUT = os.environ.get("DETECT_CLEAR_OUTPUT", "1") == "1"
DETECT_KIOSK_LLM_ONLY_IF_RESULT_EXISTS = os.environ.get(
    "DETECT_KIOSK_LLM_ONLY_IF_RESULT_EXISTS",
    "1",
) == "1"
DETECT_INTER_SESSION_SLEEP_SEC = int(
    os.environ.get("DETECT_INTER_SESSION_SLEEP_SEC", "30")
)
DETECT_SKIP_SESSION_NAMES = {
    x.strip().lower()
    for x in os.environ.get("DETECT_SKIP_SESSION_NAMES", "session_09").split(",")
    if x.strip()
}
LOGS_OUTPUT_DIRNAME = os.environ.get("DETECT_LOGS_OUTPUT_DIRNAME", "logs")
VIDEO_OUTPUT_DIRNAME = os.environ.get("DETECT_VIDEO_OUTPUT_DIRNAME", "video")
ENTRY_OUTPUT_DIRNAME = os.environ.get("DETECT_ENTRY_OUTPUT_DIRNAME", "entry")
KIOSK_OUTPUT_DIRNAME = os.environ.get("DETECT_KIOSK_OUTPUT_DIRNAME", "kiosk")
PERSON_CROPS_DIRNAME = os.environ.get("KIOSK_PERSON_CROPS_DIRNAME", "person_crops")
SAVE_PERSON_CROPS = os.environ.get("KIOSK_SAVE_PERSON_CROPS", "1") == "1"
PERSON_MODEL_PATH = _env_or_model_path(
    "KIOSK_PERSON_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "detection", "yolo26s.pt"),
    os.path.join(BASE_DIR, "yolo26s.pt"),
)
TRACKER_PATH = _env_or_model_path(
    "KIOSK_TRACKER_PATH",
    os.path.join(BASE_DIR, "models", "tracker", "custom_tracker.yaml"),
    os.path.join(BASE_DIR, "custom_tracker.yaml"),
)

FRAME_SKIP = int(os.environ.get("KIOSK_FRAME_SKIP", "8"))
RESIZE_WIDTH = int(os.environ.get("KIOSK_RESIZE_WIDTH", "960"))
MIN_BOX_HEIGHT = int(os.environ.get("KIOSK_MIN_BOX_HEIGHT", "120"))
MIN_BOX_AREA = int(os.environ.get("KIOSK_MIN_BOX_AREA", "1800"))
MIN_VISIBLE_RATIO = float(os.environ.get("KIOSK_MIN_VISIBLE_RATIO", "0.70"))
KIOSK_PERSON_MIN_BOX_HEIGHT = int(os.environ.get("KIOSK_PERSON_MIN_BOX_HEIGHT", "70"))
KIOSK_PERSON_MIN_BOX_AREA = int(os.environ.get("KIOSK_PERSON_MIN_BOX_AREA", "900"))
KIOSK_PERSON_MIN_VISIBLE_RATIO = float(os.environ.get("KIOSK_PERSON_MIN_VISIBLE_RATIO", "0.20"))
KIOSK_DISABLE_PERSON_SIZE_FILTER = os.environ.get("KIOSK_DISABLE_PERSON_SIZE_FILTER", "1") == "1"
KIOSK_ASSOC_MAX_MISSING = int(os.environ.get("KIOSK_ASSOC_MAX_MISSING", "30"))
KIOSK_ITEM_ONLY_MODE = os.environ.get("KIOSK_ITEM_ONLY_MODE", "1") == "1"
KIOSK_FALLBACK_ID_MAX_DIST = float(os.environ.get("KIOSK_FALLBACK_ID_MAX_DIST", "90"))
KIOSK_FALLBACK_ID_MAX_MISSING = int(os.environ.get("KIOSK_FALLBACK_ID_MAX_MISSING", "20"))
GROUP_TRACKING_ENABLE = os.environ.get("KIOSK_GROUP_TRACKING_ENABLE", "1") == "1"
DEFAULT_GROUP_ID = int(os.environ.get("KIOSK_DEFAULT_GROUP_ID", "1"))
GROUP_EVIDENCE_ENABLE = os.environ.get("KIOSK_GROUP_EVIDENCE_ENABLE", "1") == "1"
GROUP_EVIDENCE_DIRNAME = os.environ.get("KIOSK_GROUP_EVIDENCE_DIRNAME", "group_evidence")
GROUP_EVIDENCE_MAX_IMAGES = int(os.environ.get("KIOSK_GROUP_EVIDENCE_MAX_IMAGES", "0"))
GROUP_EVIDENCE_MAX_IMAGES_PER_MINUTE = int(os.environ.get("KIOSK_GROUP_EVIDENCE_MAX_IMAGES_PER_MINUTE", "10"))
GROUP_LLM_INPUT_DIRNAME = os.environ.get("KIOSK_GROUP_LLM_INPUT_DIRNAME", "llm_input")
GROUP_VLM_PROVIDER = os.environ.get("KIOSK_GROUP_VLM_PROVIDER", "gemini").strip().lower()
GROUP_VLM_ENABLE = os.environ.get("KIOSK_GROUP_VLM_ENABLE", "1") == "1"
GROUP_VLM_REQUIRED = os.environ.get("KIOSK_GROUP_VLM_REQUIRED", "1") == "1"
DEFAULT_GROUP_VLM_MODEL = os.path.join(BASE_DIR, "models", "Qwen3-VL-2B-Instruct")
GROUP_VLM_MODEL_NAME = os.environ.get(
    "KIOSK_GROUP_VLM_MODEL_NAME",
    DEFAULT_GROUP_VLM_MODEL if os.path.isdir(DEFAULT_GROUP_VLM_MODEL) else "Qwen/Qwen3-VL-2B-Instruct",
)
GROUP_VLM_LOCAL_ONLY = os.environ.get("KIOSK_GROUP_VLM_LOCAL_ONLY", "1") == "1"
GROUP_VLM_MAX_NEW_TOKENS = int(os.environ.get("KIOSK_GROUP_VLM_MAX_NEW_TOKENS", "512"))
GROUP_OPENAI_MODEL = os.environ.get("KIOSK_GROUP_OPENAI_MODEL", "gpt-5.4")
GROUP_OPENAI_API_KEY_PATH = os.environ.get(
    "KIOSK_GROUP_OPENAI_API_KEY_PATH",
    os.path.join(BASE_DIR, "chatgpt_API.txt"),
)
GROUP_OPENAI_BASE_URL = os.environ.get("KIOSK_GROUP_OPENAI_BASE_URL", "https://api.openai.com/v1")
GROUP_OPENAI_TIMEOUT_SEC = int(os.environ.get("KIOSK_GROUP_OPENAI_TIMEOUT_SEC", "180"))
GROUP_OPENAI_RETRY_COUNT = int(os.environ.get("KIOSK_GROUP_OPENAI_RETRY_COUNT", "4"))
GROUP_OPENAI_RATE_LIMIT_RETRY_SCHEDULE_SEC = [
    int(x.strip()) for x in os.environ.get(
        "KIOSK_GROUP_OPENAI_RATE_LIMIT_RETRY_SCHEDULE_SEC",
        "60,300,600",
    ).split(",") if x.strip()
]
GROUP_GEMINI_MODEL = os.environ.get("KIOSK_GROUP_GEMINI_MODEL", "gemini-3-flash-preview")
GROUP_GEMINI_COMPARE_MODELS = [
    x.strip() for x in os.environ.get(
        "KIOSK_GROUP_GEMINI_COMPARE_MODELS",
        "",
    ).split(",") if x.strip()
]
GROUP_GEMINI_API_KEY_PATH = os.environ.get(
    "KIOSK_GROUP_GEMINI_API_KEY_PATH",
    os.path.join(BASE_DIR, "gemini_API.txt"),
)
GROUP_GEMINI_BASE_URL = os.environ.get(
    "KIOSK_GROUP_GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)
GROUP_GEMINI_THINKING_BUDGET = int(os.environ.get("KIOSK_GROUP_GEMINI_THINKING_BUDGET", "0"))
GROUP_GEMINI_TIMEOUT_SEC = int(os.environ.get("KIOSK_GROUP_GEMINI_TIMEOUT_SEC", "180"))
GROUP_GEMINI_RETRY_COUNT = int(os.environ.get("KIOSK_GROUP_GEMINI_RETRY_COUNT", "4"))
GROUP_GEMINI_RATE_LIMIT_RETRY_SCHEDULE_SEC = [
    int(x.strip()) for x in os.environ.get(
        "KIOSK_GROUP_GEMINI_RATE_LIMIT_RETRY_SCHEDULE_SEC",
        "60,300,600",
    ).split(",") if x.strip()
]
GROUP_ANTHROPIC_MODEL = os.environ.get("KIOSK_GROUP_ANTHROPIC_MODEL", "claude-haiku-4-5")
GROUP_ANTHROPIC_API_KEY_PATH = os.environ.get(
    "KIOSK_GROUP_ANTHROPIC_API_KEY_PATH",
    os.path.join(BASE_DIR, "anthropic_API.txt"),
)
GROUP_ANTHROPIC_BASE_URL = os.environ.get("KIOSK_GROUP_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
GROUP_ANTHROPIC_VERSION = os.environ.get("KIOSK_GROUP_ANTHROPIC_VERSION", "2023-06-01")
GROUP_ANTHROPIC_TIMEOUT_SEC = int(os.environ.get("KIOSK_GROUP_ANTHROPIC_TIMEOUT_SEC", "180"))
GROUP_ANTHROPIC_RETRY_COUNT = int(os.environ.get("KIOSK_GROUP_ANTHROPIC_RETRY_COUNT", "4"))
GROUP_ANTHROPIC_RATE_LIMIT_RETRY_SCHEDULE_SEC = [
    int(x.strip()) for x in os.environ.get(
        "KIOSK_GROUP_ANTHROPIC_RATE_LIMIT_RETRY_SCHEDULE_SEC",
        "60,300,600",
    ).split(",") if x.strip()
]
FAST_QWEN_EVIDENCE_MODE = os.environ.get("KIOSK_FAST_QWEN_EVIDENCE_MODE", "1") == "1"
FAST_EVIDENCE_EXIT_GAP = int(os.environ.get("KIOSK_FAST_EVIDENCE_EXIT_GAP", "30"))
FAST_HAND_EVENT_EXPORT_ENABLE = os.environ.get("KIOSK_FAST_HAND_EVENT_EXPORT_ENABLE", "1") == "1"
FAST_HAND_EVENT_GAP = int(os.environ.get("KIOSK_FAST_HAND_EVENT_GAP", "18"))
FAST_HAND_EVENT_PAD = int(os.environ.get("KIOSK_FAST_HAND_EVENT_PAD", "35"))
FAST_HAND_EVENT_DEDUP_IOU = float(os.environ.get("KIOSK_FAST_HAND_EVENT_DEDUP_IOU", "0.70"))
FAST_HAND_EVENT_DEDUP_SIM = float(os.environ.get("KIOSK_FAST_HAND_EVENT_DEDUP_SIM", "0.96"))
FAST_HAND_EVENT_LOG = os.environ.get("KIOSK_FAST_HAND_EVENT_LOG", "1") == "1"
FAST_HAND_EVENT_SAVE_ALL = os.environ.get("KIOSK_FAST_HAND_EVENT_SAVE_ALL", "1") == "1"
FAST_HAND_EVENT_WEAK_ENABLE = os.environ.get("KIOSK_FAST_HAND_EVENT_WEAK_ENABLE", "1") == "1"
FAST_HAND_EVENT_WEAK_MAX_GAP = int(os.environ.get("KIOSK_FAST_HAND_EVENT_WEAK_MAX_GAP", "180"))
FAST_HAND_EVENT_WEAK_GAP = int(os.environ.get("KIOSK_FAST_HAND_EVENT_WEAK_GAP", "9"))
FAST_HAND_ITEM_RESAVE_GAP = int(os.environ.get("KIOSK_FAST_HAND_ITEM_RESAVE_GAP", "64"))
FAST_HAND_ITEM_SAME_IOU = float(os.environ.get("KIOSK_FAST_HAND_ITEM_SAME_IOU", "0.45"))
FAST_HAND_ITEM_SAME_SIM = float(os.environ.get("KIOSK_FAST_HAND_ITEM_SAME_SIM", "0.92"))
FAST_HOLD_CONFIRM_ENABLE = os.environ.get("KIOSK_FAST_HOLD_CONFIRM_ENABLE", "1") == "1"
FAST_HOLD_CONFIRM_CONF = float(os.environ.get("KIOSK_FAST_HOLD_CONFIRM_CONF", "0.18"))
FAST_HOLD_CONFIRM_MIN_SIDE = int(os.environ.get("KIOSK_FAST_HOLD_CONFIRM_MIN_SIDE", "12"))
FAST_HOLD_CONFIRM_MIN_AREA = int(os.environ.get("KIOSK_FAST_HOLD_CONFIRM_MIN_AREA", "120"))
FAST_HOLD_CONFIRM_FALLBACK_ENABLE = os.environ.get("KIOSK_FAST_HOLD_CONFIRM_FALLBACK_ENABLE", "1") == "1"
FAST_BODY_CARRY_ENABLE = os.environ.get("KIOSK_FAST_BODY_CARRY_ENABLE", "1") == "1"
FAST_BAG_EVENT_EXPORT_ENABLE = os.environ.get("KIOSK_FAST_BAG_EVENT_EXPORT_ENABLE", "1") == "1"
FAST_BAG_EVENT_GAP = int(os.environ.get("KIOSK_FAST_BAG_EVENT_GAP", "24"))
FAST_PRE_EXIT_CARRY_GAP = int(os.environ.get("KIOSK_FAST_PRE_EXIT_CARRY_GAP", "96"))
FAST_PRE_EXIT_CARRY_OFFSETS = [
    int(x.strip()) for x in os.environ.get(
        "KIOSK_FAST_PRE_EXIT_CARRY_OFFSETS",
        "0,8,16,24",
    ).split(",") if x.strip()
]
GROUP_EVIDENCE_MIN_FRAME_GAP = int(os.environ.get("KIOSK_GROUP_EVIDENCE_MIN_FRAME_GAP", "18"))
GROUP_EVIDENCE_DIVERSE_SIM = float(os.environ.get("KIOSK_GROUP_EVIDENCE_DIVERSE_SIM", "0.985"))
GROUP_EVIDENCE_CLUSTER_MAX_GAP = int(os.environ.get("KIOSK_GROUP_EVIDENCE_CLUSTER_MAX_GAP", "180"))

# ReID for stable customer IDs in kiosk videos.
REID_SIM_THRESHOLD = float(os.environ.get("KIOSK_REID_SIM_THRESHOLD", "0.40"))
SPATIAL_GATE = int(os.environ.get("KIOSK_SPATIAL_GATE", "220"))
MIN_SEEN_FOR_GALLERY = int(os.environ.get("KIOSK_MIN_SEEN_FOR_GALLERY", "6"))
USE_ENTRY_REID_TRACKING = os.environ.get("KIOSK_USE_ENTRY_REID_TRACKING", "1") == "1"
PERSON_REID_MODEL_NAME = os.environ.get("KIOSK_PERSON_REID_MODEL_NAME", "osnet_x1_0")
PERSON_REID_WEIGHTS_PATH = os.environ.get(
    "KIOSK_PERSON_REID_WEIGHTS_PATH",
    _first_existing_path(
        os.path.join(BASE_DIR, "models", "reid", "osnet_x1_0_msmt17.pt"),
        os.path.join(BASE_DIR, "models", "osnet_x1_0_msmt17.pt"),
    ) or os.path.join(BASE_DIR, "models", "reid", "osnet_x1_0_msmt17.pt"),
)
PERSON_REID_INPUT_HEIGHT = int(os.environ.get("KIOSK_PERSON_REID_INPUT_HEIGHT", "256"))
PERSON_REID_INPUT_WIDTH = int(os.environ.get("KIOSK_PERSON_REID_INPUT_WIDTH", "128"))
PERSON_REID_MIN_BOX_HEIGHT = int(os.environ.get("KIOSK_PERSON_REID_MIN_BOX_HEIGHT", "80"))
PERSON_REID_MIN_BOX_WIDTH = int(os.environ.get("KIOSK_PERSON_REID_MIN_BOX_WIDTH", "32"))
PERSON_REID_UPDATE_EVERY = int(os.environ.get("KIOSK_PERSON_REID_UPDATE_EVERY", "8"))
KIOSK_REID_USE_SEGMENTATION = os.environ.get("KIOSK_REID_USE_SEGMENTATION", "0") == "1"
PERSON_REID_NEW_VIEW_GAP = int(os.environ.get("KIOSK_PERSON_REID_NEW_VIEW_GAP", "9"))
PERSON_REID_VIEW_DIVERSITY_THRESH = float(
    os.environ.get("KIOSK_PERSON_REID_VIEW_DIVERSITY_THRESH", "0.75")
)

# Cross-video carry-over from Entry videos into kiosk tracking.
CROSS_VIDEO_REID = os.environ.get("DETECT_CROSS_VIDEO_REID", "1") == "1"
CROSS_VIDEO_REID_SIM_THRESHOLD = float(os.environ.get("DETECT_CROSS_VIDEO_REID_SIM_THRESHOLD", "0.10"))
CROSS_VIDEO_MIN_VISIBLE_RATIO = float(os.environ.get("DETECT_CROSS_VIDEO_MIN_VISIBLE_RATIO", "0.70"))
CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1 = os.environ.get("DETECT_CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1", "0") == "1"
CROSS_VIDEO_DELAY_ON_AMBIGUOUS = os.environ.get("DETECT_CROSS_VIDEO_DELAY_ON_AMBIGUOUS", "1") == "1"
CROSS_VIDEO_MIN_STREAK = int(os.environ.get("DETECT_CROSS_VIDEO_MIN_STREAK", "2"))
NEW_ID_AGG_FRAMES = int(os.environ.get("DETECT_NEW_ID_AGG_FRAMES", "2"))
KIOSK_ENTRY_MATCH_WAIT_FRAMES = int(os.environ.get("DETECT_KIOSK_ENTRY_MATCH_WAIT_FRAMES", "4"))
KIOSK_CROSS_VIDEO_ASSIGN_MIN_STREAK = int(
    os.environ.get("DETECT_KIOSK_CROSS_VIDEO_ASSIGN_MIN_STREAK", "3")
)
KIOSK_CROSS_VIDEO_AMBIGUITY_GAP = float(
    os.environ.get("DETECT_KIOSK_CROSS_VIDEO_AMBIGUITY_GAP", "0.06")
)
KIOSK_CROSS_VIDEO_CONTESTED_MIN_STREAK = int(
    os.environ.get("DETECT_KIOSK_CROSS_VIDEO_CONTESTED_MIN_STREAK", "5")
)
KIOSK_CROSS_VIDEO_CONTESTED_GAP = float(
    os.environ.get("DETECT_KIOSK_CROSS_VIDEO_CONTESTED_GAP", "0.09")
)
KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_STREAK = int(
    os.environ.get("DETECT_KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_STREAK", "8")
)
KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_MIN_SIM = float(
    os.environ.get("DETECT_KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_MIN_SIM", "0.48")
)
KIOSK_REID_UPDATE_MIN_ASSIGNED_FRAMES = int(
    os.environ.get("DETECT_KIOSK_REID_UPDATE_MIN_ASSIGNED_FRAMES", "64")
)
KIOSK_STICKY_REID_MARGIN = float(
    os.environ.get("DETECT_KIOSK_STICKY_REID_MARGIN", "0.08")
)
KIOSK_REID_UPDATE_VERIFY_MIN_OS = float(
    os.environ.get("DETECT_KIOSK_REID_UPDATE_VERIFY_MIN_OS", "0.52")
)
KIOSK_REID_UPDATE_VERIFY_MIN_FC = float(
    os.environ.get("DETECT_KIOSK_REID_UPDATE_VERIFY_MIN_FC", "0.55")
)
KIOSK_REID_UPDATE_VERIFY_MIN_FUSED = float(
    os.environ.get("DETECT_KIOSK_REID_UPDATE_VERIFY_MIN_FUSED", "0.54")
)
KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_OS = float(
    os.environ.get("DETECT_KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_OS", "0.80")
)
KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_FC = float(
    os.environ.get("DETECT_KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_FC", "0.60")
)
KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_FUSED = float(
    os.environ.get("DETECT_KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_FUSED", "0.78")
)
KIOSK_ENTRY_REID_UPDATE_MAX_PEOPLE = int(
    os.environ.get("DETECT_KIOSK_ENTRY_REID_UPDATE_MAX_PEOPLE", "1")
)
KIOSK_REID_UPDATE_COMPETE_MARGIN = float(
    os.environ.get("DETECT_KIOSK_REID_UPDATE_COMPETE_MARGIN", "0.02")
)
NEW_ID_PERSIST_SIM_THRESHOLD = float(
    os.environ.get(
        "DETECT_NEW_ID_PERSIST_SIM_THRESHOLD",
        str(CROSS_VIDEO_REID_SIM_THRESHOLD),
    )
)

# Kiosk interaction settings.
INTERACT_START_FRAMES = int(os.environ.get("KIOSK_INTERACT_START_FRAMES", "9999"))
INTERACT_END_MISSING_FRAMES = int(os.environ.get("KIOSK_INTERACT_END_MISSING_FRAMES", "9999"))
INTERACT_ZONE_EXPAND = int(os.environ.get("KIOSK_INTERACT_ZONE_EXPAND", "12"))
CUSTOMER_EXIT_MISSING_FRAMES = int(os.environ.get("KIOSK_CUSTOMER_EXIT_MISSING_FRAMES", "20"))
ITEM_PLACE_GAP_FRAMES = int(os.environ.get("KIOSK_ITEM_PLACE_GAP_FRAMES", "12"))
ITEM_HIDDEN_BODY_PAD = int(os.environ.get("KIOSK_ITEM_HIDDEN_BODY_PAD", "60"))
ITEM_HIDDEN_MAX_OWNER_MISSING = int(os.environ.get("KIOSK_ITEM_HIDDEN_MAX_OWNER_MISSING", "12"))
ITEM_OCCLUDED_PLACE_GAP_FRAMES = int(os.environ.get("KIOSK_ITEM_OCCLUDED_PLACE_GAP_FRAMES", "10"))
ITEM_EXIT_CONFIRM_FRAMES = int(os.environ.get("KIOSK_ITEM_EXIT_CONFIRM_FRAMES", "3"))

# Product highlighting on kiosk videos.
PRODUCT_DETECT_ENABLE = os.environ.get("KIOSK_PRODUCT_DETECT_ENABLE", "1") == "1"
PRODUCT_MODEL_PATH = _env_or_model_path(
    "KIOSK_PRODUCT_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "detection", "yolo26s.pt"),
    os.path.join(BASE_DIR, "yolo26s.pt"),
)
PRODUCT_USE_TRACK = os.environ.get("KIOSK_PRODUCT_USE_TRACK", "1") == "1"
GROUND_DINO_ENABLE = os.environ.get("KIOSK_GROUND_DINO_ENABLE", "1") == "1"
GROUND_DINO_MODEL_NAME = os.environ.get("KIOSK_GROUND_DINO_MODEL_NAME", "IDEA-Research/grounding-dino-base")
GROUND_DINO_TEXT_PROMPT = os.environ.get(
    "KIOSK_GROUND_DINO_TEXT_PROMPT",
    "bottle . carton . pack . case . box . can . cup . detergent bottle . retail product ."
)
PRODUCT_CONF = float(os.environ.get("KIOSK_PRODUCT_CONF", "0.05"))
PRODUCT_FALLBACK_TO_PERSON_MODEL = os.environ.get("KIOSK_PRODUCT_FALLBACK_TO_PERSON_MODEL", "1") == "1"
PRODUCT_MIN_AREA = int(os.environ.get("KIOSK_PRODUCT_MIN_AREA", "500"))
PRODUCT_MIN_SIDE = int(os.environ.get("KIOSK_PRODUCT_MIN_SIDE", "28"))
# COCO IDs commonly useful for handheld/retail items:
# bottle(39), wine glass(40), cup(41), bowl(45),
# banana(46), apple(47), sandwich(48), orange(49), broccoli(50),
# carrot(51), hot dog(52), pizza(53), donut(54), cake(55)
PRODUCT_CLASS_IDS = [
    int(x.strip()) for x in os.environ.get(
        "KIOSK_PRODUCT_CLASS_IDS",
        "39,40,41,45,46,47,48,49,50,51,52,53,54,55",
    ).split(",") if x.strip()
]
PRODUCT_PROMPTS = [
    p.strip() for p in os.environ.get(
        "KIOSK_PRODUCT_PROMPTS",
        "bottle,drink can,cup,snack pack,box,bread"
    ).split(",") if p.strip()
]
PRODUCT_CLUSTER_IOU = float(os.environ.get("KIOSK_PRODUCT_CLUSTER_IOU", "0.55"))
PRODUCT_ROI_TOP_K = int(os.environ.get("KIOSK_PRODUCT_ROI_TOP_K", "3"))
PRODUCT_STABLE_FRAMES = int(os.environ.get("KIOSK_PRODUCT_STABLE_FRAMES", "10"))
PRODUCT_CLASS_PRIORITY = {
    "bottle": 4,
    "cup": 2,
    "wine glass": 1,
    "bowl": 1,
}
PRODUCT_INCLUDE_LABEL_KEYWORDS = [
    s.strip().lower() for s in os.environ.get(
        "KIOSK_PRODUCT_INCLUDE_LABEL_KEYWORDS",
        "bottle,carton,case,pack,bottle carton,carton of bottles,drink case,beverage case,shrink-wrapped,drink can,can,cup,snack,box,bread,product"
    ).split(",") if s.strip()
]
PRODUCT_ASSOC_MAX_DIST = int(os.environ.get("KIOSK_PRODUCT_ASSOC_MAX_DIST", "420"))
# If 0, disable distance cap and always use nearest person.
PRODUCT_ASSOC_DISABLE_DIST_CAP = os.environ.get("KIOSK_PRODUCT_ASSOC_DISABLE_DIST_CAP", "0") == "1"
PRODUCT_EMPTY_LABEL_MIN_CONF = float(os.environ.get("KIOSK_PRODUCT_EMPTY_LABEL_MIN_CONF", "0.18"))
SCANNED_ITEMS_DIRNAME = os.environ.get("KIOSK_SCANNED_ITEMS_DIRNAME", "scanned_items")
SCANNED_FRAMES_DIRNAME = os.environ.get("KIOSK_SCANNED_FRAMES_DIRNAME", "scanned_frames")
REFERENCE_HITS_DIRNAME = os.environ.get("KIOSK_REFERENCE_HITS_DIRNAME", "reference_hits")
HOLD_FAILS_DIRNAME = os.environ.get("KIOSK_HOLD_FAILS_DIRNAME", "hold_fails")
PRODUCT_KEY_GRID = int(os.environ.get("KIOSK_PRODUCT_KEY_GRID", "20"))
PRODUCT_RECOUNT_GAP_FRAMES = int(os.environ.get("KIOSK_PRODUCT_RECOUNT_GAP_FRAMES", "24"))
PRODUCT_DEBUG_LOG = os.environ.get("KIOSK_PRODUCT_DEBUG_LOG", "1") == "1"
PRODUCT_SKIP_LOG = os.environ.get("KIOSK_PRODUCT_SKIP_LOG", "0") == "1"
HOLD_FAIL_SUMMARY_LOG = os.environ.get("KIOSK_HOLD_FAIL_SUMMARY_LOG", "1") == "1"
SAVE_HOLD_FAIL_CROPS = os.environ.get("KIOSK_SAVE_HOLD_FAIL_CROPS", "1") == "1"
PRODUCT_MIN_TRACK_FRAMES = int(os.environ.get("KIOSK_PRODUCT_MIN_TRACK_FRAMES", "1"))
PRODUCT_MATCH_MIN_STREAK = int(os.environ.get("KIOSK_PRODUCT_MATCH_MIN_STREAK", "3"))
PRODUCT_PERSON_PAD = int(os.environ.get("KIOSK_PRODUCT_PERSON_PAD", "40"))
HOLD_DETECT_MIN_STREAK = int(os.environ.get("KIOSK_HOLD_DETECT_MIN_STREAK", "1"))
ITEM_MATCH_IOU_TH = float(os.environ.get("KIOSK_ITEM_MATCH_IOU_TH", "0.30"))
ITEM_MATCH_MAX_DIST = int(os.environ.get("KIOSK_ITEM_MATCH_MAX_DIST", "90"))
ITEM_MATCH_MAX_GAP = int(os.environ.get("KIOSK_ITEM_MATCH_MAX_GAP", "40"))
ITEM_MATCH_MAX_GAP_HOLDING = int(os.environ.get("KIOSK_ITEM_MATCH_MAX_GAP_HOLDING", "180"))
ITEM_NEW_MIN_STREAK = int(os.environ.get("KIOSK_ITEM_NEW_MIN_STREAK", "6"))
ITEM_NEW_MIN_CONF = float(os.environ.get("KIOSK_ITEM_NEW_MIN_CONF", "0.10"))
ITEM_NEW_FORCE_FAR_DIST = int(os.environ.get("KIOSK_ITEM_NEW_FORCE_FAR_DIST", "180"))
ITEM_PENDING_GRID = int(os.environ.get("KIOSK_ITEM_PENDING_GRID", "40"))
ITEM_DISPLAY_MEMORY_FRAMES = int(os.environ.get("KIOSK_ITEM_DISPLAY_MEMORY_FRAMES", "24"))
ITEM_HIDDEN_RELINK_MAX_GAP = int(os.environ.get("KIOSK_ITEM_HIDDEN_RELINK_MAX_GAP", "36"))
ITEM_HIDDEN_RELINK_MIN_IOU = float(os.environ.get("KIOSK_ITEM_HIDDEN_RELINK_MIN_IOU", "0.20"))
ITEM_HIDDEN_RELINK_MAX_DIST = int(os.environ.get("KIOSK_ITEM_HIDDEN_RELINK_MAX_DIST", "80"))
ITEM_HIDDEN_RELINK_MIN_CSIM = float(os.environ.get("KIOSK_ITEM_HIDDEN_RELINK_MIN_CSIM", "0.65"))
ITEM_HIDDEN_RELINK_MIN_SIM = float(os.environ.get("KIOSK_ITEM_HIDDEN_RELINK_MIN_SIM", "0.30"))
ITEM_SOFT_RELINK_MAX_GAP = int(os.environ.get("KIOSK_ITEM_SOFT_RELINK_MAX_GAP", "60"))
ITEM_SOFT_RELINK_MIN_IOU = float(os.environ.get("KIOSK_ITEM_SOFT_RELINK_MIN_IOU", "0.15"))
ITEM_SOFT_RELINK_MAX_DIST = int(os.environ.get("KIOSK_ITEM_SOFT_RELINK_MAX_DIST", "110"))
ITEM_SOFT_RELINK_MIN_CSIM = float(os.environ.get("KIOSK_ITEM_SOFT_RELINK_MIN_CSIM", "0.55"))
ITEM_SOFT_RELINK_MIN_SIM = float(os.environ.get("KIOSK_ITEM_SOFT_RELINK_MIN_SIM", "0.20"))
PRODUCT_OVERLAY_ENABLE = os.environ.get("KIOSK_PRODUCT_OVERLAY_ENABLE", "1") == "1"
HOLDING_DISPLAY_TTL = int(os.environ.get("KIOSK_HOLDING_DISPLAY_TTL", "20"))
HOLDING_REQUIRE_HANDS = os.environ.get("KIOSK_HOLDING_REQUIRE_HANDS", "1") == "1"
PRODUCT_REQUIRE_HAND_ACTIVITY = os.environ.get("KIOSK_PRODUCT_REQUIRE_HAND_ACTIVITY", "1") == "1"
HAND_ROI_PAD = int(os.environ.get("KIOSK_HAND_ROI_PAD", "140"))
PHONE_SUPPRESS_ENABLE = os.environ.get("KIOSK_PHONE_SUPPRESS_ENABLE", "1") == "1"
PHONE_IOU_SUPPRESS_TH = float(os.environ.get("KIOSK_PHONE_IOU_SUPPRESS_TH", "0.25"))
POSE_ENABLE = os.environ.get("KIOSK_POSE_ENABLE", "1") == "1"
POSE_MODEL_PATH = _env_or_model_path(
    "KIOSK_POSE_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "pose", "yolo26l-pose.pt"),
    os.path.join(BASE_DIR, "yolo26l-pose.pt"),
)
POSE_KPT_CONF_TH = float(os.environ.get("KIOSK_POSE_KPT_CONF_TH", "0.35"))
POSE_HAND_NEAR_PX = int(os.environ.get("KIOSK_POSE_HAND_NEAR_PX", "85"))
POSE_ONLY_LOG_MODE = os.environ.get("KIOSK_POSE_ONLY_LOG_MODE", "0") == "1"
POSE_LOG_EVERY_N = int(os.environ.get("KIOSK_POSE_LOG_EVERY_N", "1"))
PERSON_DEBUG_LOG = os.environ.get("KIOSK_PERSON_DEBUG_LOG", "1") == "1"
DINO_ENABLE = os.environ.get("KIOSK_DINO_ENABLE", "1") == "1"
DINO_MODEL_NAME = os.environ.get("KIOSK_DINO_MODEL_NAME", "facebook/dinov2-base")
DINO_MATCH_WEIGHT = float(os.environ.get("KIOSK_DINO_MATCH_WEIGHT", "0.35"))
DINO_MIN_SIM_FOR_MATCH = float(os.environ.get("KIOSK_DINO_MIN_SIM_FOR_MATCH", "0.45"))
SCANNER_FILTER_ENABLE = os.environ.get("KIOSK_SCANNER_FILTER_ENABLE", "1") == "1"
SCANNER_REF_PATH = os.environ.get(
    "KIOSK_SCANNER_REF_PATH",
    os.path.join(BASE_DIR, "references", "images", "scanner.jpg"),
)
SCANNER_DINO_SIM_TH = float(os.environ.get("KIOSK_SCANNER_DINO_SIM_TH", "0.72"))
SCANNER_COLOR_SIM_TH = float(os.environ.get("KIOSK_SCANNER_COLOR_SIM_TH", "0.60"))
PRODUCT_REFERENCE_ENABLE = os.environ.get("KIOSK_PRODUCT_REFERENCE_ENABLE", "1") == "1"
PRODUCT_REFERENCE_DIR = os.environ.get(
    "KIOSK_PRODUCT_REFERENCE_DIR",
    os.path.join(BASE_DIR, "product_training_data", "references"),
)
PRODUCT_REFERENCE_PATH = os.environ.get(
    "KIOSK_PRODUCT_REFERENCE_PATH",
    os.path.join(BASE_DIR, "product_training_data", "references", "bottle_carton.npz"),
)
PRODUCT_REFERENCE_DINO_SIM_TH = float(os.environ.get("KIOSK_PRODUCT_REFERENCE_DINO_SIM_TH", "0.72"))
PRODUCT_REFERENCE_COLOR_SIM_TH = float(os.environ.get("KIOSK_PRODUCT_REFERENCE_COLOR_SIM_TH", "0.55"))
ITEM_LABEL_DISAGREE_MIN_SIM = float(os.environ.get("KIOSK_ITEM_LABEL_DISAGREE_MIN_SIM", "0.70"))
ITEM_LOCK_MIN_SIM = float(os.environ.get("KIOSK_ITEM_LOCK_MIN_SIM", "0.55"))
COLOR_MATCH_WEIGHT = float(os.environ.get("KIOSK_COLOR_MATCH_WEIGHT", "0.25"))
COLOR_MIN_SIM_FOR_MATCH = float(os.environ.get("KIOSK_COLOR_MIN_SIM_FOR_MATCH", "0.35"))
ITEM_EMB_MIN_AREA = int(os.environ.get("KIOSK_ITEM_EMB_MIN_AREA", "1200"))
ITEM_EMB_MIN_SIDE = int(os.environ.get("KIOSK_ITEM_EMB_MIN_SIDE", "24"))
ITEM_EMB_MIN_SHARPNESS = float(os.environ.get("KIOSK_ITEM_EMB_MIN_SHARPNESS", "35.0"))
ITEM_EMB_MIN_TENENGRAD = float(os.environ.get("KIOSK_ITEM_EMB_MIN_TENENGRAD", "28.0"))
ITEM_SAVE_MIN_CONF = float(os.environ.get("KIOSK_ITEM_SAVE_MIN_CONF", "0.30"))
ITEM_UPDATE_MIN_SIM = float(os.environ.get("KIOSK_ITEM_UPDATE_MIN_SIM", "0.65"))
ITEM_UPDATE_MIN_CSIM = float(os.environ.get("KIOSK_ITEM_UPDATE_MIN_CSIM", "0.45"))
ITEM_ANCHOR_MIN_SIM = float(os.environ.get("KIOSK_ITEM_ANCHOR_MIN_SIM", "0.55"))
ITEM_ANCHOR_MIN_CSIM = float(os.environ.get("KIOSK_ITEM_ANCHOR_MIN_CSIM", "0.35"))
ITEM_LABEL_SWITCH_MIN_SIM = float(os.environ.get("KIOSK_ITEM_LABEL_SWITCH_MIN_SIM", "0.60"))
ITEM_LABEL_SWITCH_MIN_CSIM = float(os.environ.get("KIOSK_ITEM_LABEL_SWITCH_MIN_CSIM", "0.40"))

pose_model = None
if POSE_ENABLE and ((not FAST_QWEN_EVIDENCE_MODE) or FAST_HAND_EVENT_EXPORT_ENABLE):
    if os.path.isfile(POSE_MODEL_PATH):
        try:
            pose_model = YOLO(POSE_MODEL_PATH)
            print(f"Kiosk pose ready: {POSE_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: cannot load pose model {POSE_MODEL_PATH}: {e}")
    else:
        print(f"Warning: pose model not found: {POSE_MODEL_PATH}")

product_model = None
if PRODUCT_DETECT_ENABLE and (not FAST_QWEN_EVIDENCE_MODE):
    if os.path.isfile(PRODUCT_MODEL_PATH):
        try:
            product_model = YOLO(PRODUCT_MODEL_PATH)
            print(f"Kiosk product model ready: {PRODUCT_MODEL_PATH}")
            # Disabled for stability: set_classes() can fail on some YOLOE checkpoints/builds.
            # Run product model in plain detection mode.
        except Exception as e:
            print(f"Warning: cannot load product model {PRODUCT_MODEL_PATH}: {e}")
    else:
        print(f"Warning: product model not found: {PRODUCT_MODEL_PATH} (fallback to base YOLO)")

dino_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
person_model = YOLO(PERSON_MODEL_PATH)
person_reid_model = None
PERSON_REID_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((PERSON_REID_INPUT_HEIGHT, PERSON_REID_INPUT_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format")
    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def _filter_incompatible_keys(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            continue
        filtered[key] = value
    return filtered


def load_person_reid_model():
    if not USE_ENTRY_REID_TRACKING:
        return None
    if torchreid is None:
        print("Warning: torchreid unavailable, falling back to tracker IDs in kiosk mode.")
        return None
    try:
        model = torchreid.models.build_model(
            name=PERSON_REID_MODEL_NAME,
            num_classes=1000,
            pretrained=False,
        )
        if os.path.exists(PERSON_REID_WEIGHTS_PATH):
            checkpoint = torch.load(PERSON_REID_WEIGHTS_PATH, map_location="cpu", weights_only=False)
            state_dict = _extract_state_dict(checkpoint)
            state_dict = _filter_incompatible_keys(model, state_dict)
            model.load_state_dict(state_dict, strict=False)
            print(f"Kiosk person ReID ready: {PERSON_REID_MODEL_NAME} ({PERSON_REID_WEIGHTS_PATH})")
        else:
            print(
                f"Warning: kiosk person ReID weights not found: {PERSON_REID_WEIGHTS_PATH}. "
                "Falling back to tracker IDs."
            )
            return None
        return model.to(dino_device).eval()
    except Exception as e:
        print(f"Warning: cannot load kiosk person ReID model: {e}")
        return None


person_reid_model = load_person_reid_model()

dino_processor = None
dino_model = None
if DINO_ENABLE and (not FAST_QWEN_EVIDENCE_MODE):
    if AutoImageProcessor is None or Dinov2Model is None:
        print("Warning: transformers not available, DINOv2 disabled.")
    else:
        try:
            dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
            dino_model = Dinov2Model.from_pretrained(DINO_MODEL_NAME).to(dino_device).eval()
            print(f"Kiosk DINOv2 ready: {DINO_MODEL_NAME}")
        except Exception as e:
            print(f"Warning: cannot load DINOv2 model {DINO_MODEL_NAME}: {e}")
            dino_processor = None
            dino_model = None

gdino_processor = None
gdino_model = None
if GROUND_DINO_ENABLE and (not FAST_QWEN_EVIDENCE_MODE):
    if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
        print("Warning: transformers Grounding DINO classes unavailable, disabling Ground DINO.")
    else:
        try:
            gdino_processor = AutoProcessor.from_pretrained(GROUND_DINO_MODEL_NAME)
            gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUND_DINO_MODEL_NAME
            ).to(dino_device).eval()
            print(f"Kiosk Grounding DINO ready: {GROUND_DINO_MODEL_NAME}")
        except Exception as e:
            print(f"Warning: cannot load Grounding DINO {GROUND_DINO_MODEL_NAME}: {e}")
            gdino_processor = None
            gdino_model = None

group_vlm_processor = None
group_vlm_model = None
group_openai_api_key = None
group_gemini_api_key = None
group_anthropic_api_key = None


def load_openai_api_key():
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    if os.path.isfile(GROUP_OPENAI_API_KEY_PATH):
        try:
            with open(GROUP_OPENAI_API_KEY_PATH, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    return None


def load_anthropic_api_key():
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key
    if os.path.isfile(GROUP_ANTHROPIC_API_KEY_PATH):
        try:
            with open(GROUP_ANTHROPIC_API_KEY_PATH, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    return None


def load_gemini_api_key():
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    if os.path.isfile(GROUP_GEMINI_API_KEY_PATH):
        try:
            with open(GROUP_GEMINI_API_KEY_PATH, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    return None


if GROUP_VLM_REQUIRED and not GROUP_VLM_ENABLE:
    raise RuntimeError("Group VLM is mandatory. Enable it with KIOSK_GROUP_VLM_ENABLE=1.")
if GROUP_VLM_ENABLE:
    if GROUP_VLM_PROVIDER == "openai":
        group_openai_api_key = load_openai_api_key()
        if not group_openai_api_key:
            if GROUP_VLM_REQUIRED:
                raise RuntimeError(
                    "OpenAI group VLM is mandatory, but no API key was found. "
                    "Set OPENAI_API_KEY or provide chatgpt_API.txt."
                )
            print("Warning: OpenAI API key not found, group VLM disabled.")
        else:
            print(f"Kiosk group VLM ready: OpenAI {GROUP_OPENAI_MODEL}")
    elif GROUP_VLM_PROVIDER == "gemini":
        group_gemini_api_key = load_gemini_api_key()
        if not group_gemini_api_key:
            if GROUP_VLM_REQUIRED:
                raise RuntimeError(
                    "Gemini group VLM is mandatory, but no API key was found. "
                    "Set GEMINI_API_KEY or provide gemini_API.txt."
                )
            print("Warning: Gemini API key not found, group VLM disabled.")
        else:
            print(f"Kiosk group VLM ready: Gemini {GROUP_GEMINI_MODEL}")
    elif GROUP_VLM_PROVIDER == "anthropic":
        group_anthropic_api_key = load_anthropic_api_key()
        if not group_anthropic_api_key:
            if GROUP_VLM_REQUIRED:
                raise RuntimeError(
                    "Anthropic group VLM is mandatory, but no API key was found. "
                    "Set ANTHROPIC_API_KEY or provide anthropic_API.txt."
                )
            print("Warning: Anthropic API key not found, group VLM disabled.")
        else:
            print(f"Kiosk group VLM ready: Anthropic {GROUP_ANTHROPIC_MODEL}")
    else:
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            if GROUP_VLM_REQUIRED:
                raise RuntimeError("Qwen3-VL is mandatory, but transformers image-text classes are unavailable.")
            print("Warning: transformers image-text classes unavailable, group VLM disabled.")
        else:
            try:
                group_vlm_processor = AutoProcessor.from_pretrained(
                    GROUP_VLM_MODEL_NAME,
                    local_files_only=GROUP_VLM_LOCAL_ONLY,
                )
                group_vlm_model = AutoModelForImageTextToText.from_pretrained(
                    GROUP_VLM_MODEL_NAME,
                    local_files_only=GROUP_VLM_LOCAL_ONLY,
                ).to(dino_device).eval()
                print(f"Kiosk group VLM ready: {GROUP_VLM_MODEL_NAME}")
            except Exception as e:
                if GROUP_VLM_REQUIRED:
                    raise RuntimeError(
                        f"Qwen3-VL is mandatory, but cannot load group VLM {GROUP_VLM_MODEL_NAME}: {e}"
                    ) from e
                print(f"Warning: cannot load group VLM {GROUP_VLM_MODEL_NAME}: {e}")
                group_vlm_processor = None
                group_vlm_model = None



def get_visible_ratio_in_frame(box, frame_w, frame_h):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area = bw * bh
    vx1 = max(0.0, min(float(frame_w), x1))
    vy1 = max(0.0, min(float(frame_h), y1))
    vx2 = max(0.0, min(float(frame_w), x2))
    vy2 = max(0.0, min(float(frame_h), y2))
    if vx2 <= vx1 or vy2 <= vy1:
        return 0.0
    vis = (vx2 - vx1) * (vy2 - vy1)
    return float(vis / max(1.0, area))


def ema_update_views(views, new_emb, alpha=0.15):
    if new_emb is None:
        return views
    if views is None:
        views = []
    if len(views) == 0:
        return [new_emb]
    # simple cosine-best EMA update for standalone kiosk mode
    q = np.asarray(new_emb, dtype=np.float32).reshape(-1)
    sims = []
    for v in views:
        vv = np.asarray(v, dtype=np.float32).reshape(-1)
        denom = (np.linalg.norm(q) * np.linalg.norm(vv)) + 1e-9
        sims.append(float(np.dot(q, vv) / denom))
    best_idx = int(np.argmax(sims))
    best = np.asarray(views[best_idx], dtype=np.float32).reshape(-1)
    upd = (1.0 - alpha) * best + alpha * q
    nrm = np.linalg.norm(upd) + 1e-9
    upd = upd / nrm
    out = list(views)
    out[best_idx] = upd
    return out


def is_new_view_candidate(views, new_emb, same_view_sim=PERSON_REID_VIEW_DIVERSITY_THRESH):
    if new_emb is None:
        return False
    if not views:
        return True
    sims = [cosine_sim_tensor(new_emb, v) for v in views if v is not None]
    if not sims:
        return True
    return max(sims) <= float(same_view_sim)


@torch.no_grad()
def extract_person_reid_embedding(frame, box):
    if person_reid_model is None:
        return None
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if (y2 - y1) < PERSON_REID_MIN_BOX_HEIGHT or (x2 - x1) < PERSON_REID_MIN_BOX_WIDTH:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = PERSON_REID_TRANSFORM(rgb).unsqueeze(0).to(dino_device)
        feat = person_reid_model(t)
        emb = F.normalize(feat, p=2, dim=1).cpu().squeeze(0)
        return emb
    except Exception:
        return None


@torch.no_grad()
def extract_customer_library_embedding(frame, box, cache_key=None):
    if not KIOSK_REID_USE_SEGMENTATION:
        return extract_person_reid_embedding(frame, box)
    try:
        emb = IntegratedEntry.extract_embedding(frame, box, cache_key=cache_key)
        if emb is not None:
            return emb
    except Exception:
        pass
    return extract_person_reid_embedding(frame, box)


def extract_customer_library_fashion_pair(frame, box, cache_key=None):
    if not KIOSK_REID_USE_SEGMENTATION:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        h = crop.shape[0]
        if h < 24:
            return None, None
        split = int(h * 0.55)
        upper = crop[:split, :]
        lower = crop[split:, :]
        return (
            IntegratedEntry.extract_fashionclip_embedding(upper),
            IntegratedEntry.extract_fashionclip_embedding(lower),
        )
    try:
        return IntegratedEntry.extract_fashion_pair(frame, box, cache_key=cache_key)
    except Exception:
        return None, None


def customer_library_fashion_pair_similarity(curr_u, curr_l, ref_u, ref_l):
    fc_total, _fc_upper, _fc_lower = customer_library_fashion_pair_similarity_breakdown(
        curr_u, curr_l, ref_u, ref_l
    )
    return fc_total


def customer_library_fashion_pair_similarity_breakdown(curr_u, curr_l, ref_u, ref_l):
    weighted_num = 0.0
    weighted_den = 0.0
    sim_upper = -1.0
    sim_lower = -1.0
    upper_weight = float(getattr(IntegratedEntry, "FASHIONCLIP_UPPER_WEIGHT", 0.8))
    lower_weight = float(getattr(IntegratedEntry, "FASHIONCLIP_LOWER_WEIGHT", 0.2))
    if curr_u is not None and ref_u is not None:
        sim_upper = cosine_sim_tensor(curr_u, ref_u)
        weighted_num += upper_weight * sim_upper
        weighted_den += upper_weight
    if curr_l is not None and ref_l is not None:
        sim_lower = cosine_sim_tensor(curr_l, ref_l)
        weighted_num += lower_weight * sim_lower
        weighted_den += lower_weight
    if weighted_den <= 0.0:
        return -1.0, sim_upper, sim_lower
    return (weighted_num / weighted_den), sim_upper, sim_lower


def customer_library_fashion_cross_alpha():
    return max(0.0, min(1.0, float(getattr(IntegratedEntry, "FASHIONCLIP_CROSS_ALPHA", 0.25))))


def customer_library_fashion_within_alpha():
    return max(0.0, min(1.0, float(getattr(IntegratedEntry, "FASHIONCLIP_WITHIN_ALPHA", 0.20))))


def fashion_pair_presence_flags(curr_u, curr_l, ref_u, ref_l):
    return (
        "Y" if curr_u is not None else "N",
        "Y" if curr_l is not None else "N",
        "Y" if ref_u is not None else "N",
        "Y" if ref_l is not None else "N",
    )


def cosine_sim_tensor(a, b):
    if a is None or b is None:
        return -1.0
    if not torch.is_tensor(a):
        a = torch.tensor(np.asarray(a), dtype=torch.float32)
    if not torch.is_tensor(b):
        b = torch.tensor(np.asarray(b), dtype=torch.float32)
    a = F.normalize(a.view(1, -1), p=2, dim=1).squeeze(0)
    b = F.normalize(b.view(1, -1), p=2, dim=1).squeeze(0)
    return float(torch.dot(a.view(-1), b.view(-1)).clamp(-1.0, 1.0))


def avg_sim_against_views_tensor(query_emb, views):
    if query_emb is None or not views:
        return -1.0
    sims = [cosine_sim_tensor(query_emb, v) for v in views if v is not None]
    if not sims:
        return -1.0
    return float(sum(sims) / max(1, len(sims)))


def normalize_gallery_view_tensor(view):
    if view is None:
        return None
    if torch.is_tensor(view):
        t = view.detach().cpu().float().view(-1)
    else:
        arr = np.asarray(view, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        t = torch.tensor(arr, dtype=torch.float32).view(-1)
    if t.numel() == 0:
        return None
    return F.normalize(t.view(1, -1), p=2, dim=1).squeeze(0).cpu()


def ema_update_persistent_gallery_views(views, new_emb, alpha=0.15):
    new_t = normalize_gallery_view_tensor(new_emb)
    if new_t is None:
        return list(views or [])
    existing = []
    for v in (views or []):
        vt = normalize_gallery_view_tensor(v)
        if vt is not None:
            existing.append(vt)
    if not existing:
        return [new_t]
    sims = [cosine_sim_tensor(new_t, v) for v in existing]
    best_idx = int(np.argmax(sims))
    best = existing[best_idx]
    updated = F.normalize(
        ((1.0 - alpha) * best + alpha * new_t).view(1, -1),
        p=2,
        dim=1,
    ).squeeze(0).cpu()
    out = list(existing)
    out[best_idx] = updated
    return out


def get_pending_avg_embedding(pending_new_trackers, tracker_id, emb):
    if emb is None:
        return None, 0
    slot = pending_new_trackers.setdefault(
        tracker_id,
        {"embs": [], "ambiguous_hold": False, "samples": 0},
    )
    slot["samples"] = int(slot.get("samples", 0)) + 1
    slot["embs"].append(emb)
    if len(slot["embs"]) > max(1, NEW_ID_AGG_FRAMES):
        slot["embs"] = slot["embs"][-max(1, NEW_ID_AGG_FRAMES):]
    count = len(slot["embs"])
    if count <= 0:
        return None, 0
    stacked = torch.stack([
        v if torch.is_tensor(v) else torch.tensor(np.asarray(v), dtype=torch.float32)
        for v in slot["embs"]
    ], dim=0)
    avg = stacked.mean(dim=0, keepdim=True)
    avg = F.normalize(avg, p=2, dim=1).squeeze(0)
    return avg, count


def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)


@torch.no_grad()
def extract_product_embedding(frame, box):
    if dino_model is None or dino_processor is None:
        return None
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = dino_processor(images=pil, return_tensors="pt").to(dino_device)
        out = dino_model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1).squeeze(0).detach().cpu().numpy()
        return emb
    except Exception:
        return None


def emb_cos_sim(a, b):
    if a is None or b is None:
        return -1.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / den)


def extract_color_hist_embedding(frame, box, h_bins=18, s_bins=16):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    if hist is None:
        return None
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    nrm = np.linalg.norm(hist) + 1e-9
    return hist / nrm


def load_product_reference(npz_path):
    if not os.path.isfile(npz_path):
        return None
    try:
        ref = np.load(npz_path, allow_pickle=True)
        return {
            "label": str(ref["label"].item() if hasattr(ref["label"], "item") else ref["label"]),
            "image": str(ref["image"].item() if hasattr(ref["image"], "item") else ref["image"]),
            "model": str(ref["model"].item() if hasattr(ref["model"], "item") else ref["model"]),
            "dino_emb": np.asarray(ref["dino_emb"], dtype=np.float32),
            "color_emb": np.asarray(ref["color_emb"], dtype=np.float32),
        }
    except Exception:
        return None


def load_product_references(reference_dir, reference_path=None):
    refs = []
    seen = set()
    if reference_path:
        rp = os.path.abspath(reference_path)
        if os.path.isfile(rp):
            seen.add(rp)
            ref = load_product_reference(rp)
            if ref is not None:
                ref["path"] = rp
                refs.append(ref)
    if reference_dir and os.path.isdir(reference_dir):
        for npz_path in sorted(glob.glob(os.path.join(reference_dir, "*.npz"))):
            ap = os.path.abspath(npz_path)
            if ap in seen:
                continue
            ref = load_product_reference(ap)
            if ref is not None:
                ref["path"] = ap
                refs.append(ref)
    return refs


def label_to_prompt_phrases(label):
    raw = str(label or "").strip().lower().replace("_", " ")
    raw = " ".join(raw.split())
    if not raw:
        return []
    tokens = raw.split()
    token_set = set(tokens)
    phrases = [raw]

    def add(phrase):
        phrase = " ".join(str(phrase).strip().lower().split())
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    # Generic packaging variants.
    if "carton" in token_set:
        add(raw.replace("carton", "pack"))
        add(raw.replace("carton", "case"))
        add(raw.replace("carton", "box"))
        add(f"{raw} package")
    if "pack" in token_set:
        add(raw.replace("pack", "carton"))
        add(raw.replace("pack", "case"))
    if "case" in token_set:
        add(raw.replace("case", "carton"))
        add(raw.replace("case", "pack"))

    # Water / bottle style variants.
    if "water" in token_set or "mineral" in token_set:
        add(raw.replace("mineral", "mineral water"))
        add("water bottle pack")
        add("bottled water pack")
        add("carton of water bottles")
    if "bottle" in token_set:
        add("carton of bottles")
        add("shrink-wrapped bottle pack")
        add("beverage case")
        add("drink case")
    if "carton" in token_set and "bottle" not in token_set:
        add(f"{raw} bottle pack")
        add(f"{raw} bottled water pack")
    if "mineral" in token_set and "carton" in token_set:
        add("mineral water carton")
        add("mineral water pack")
        add("summer mineral water carton")

    return phrases


def build_ground_dino_prompt(base_prompt, product_references):
    phrases = []
    seen = set()

    for chunk in str(base_prompt or "").split("."):
        phrase = " ".join(chunk.strip().lower().split())
        if phrase and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)

    for ref in product_references:
        for phrase in label_to_prompt_phrases(ref.get("label", "")):
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)

    if not phrases:
        return "retail product ."
    return " . ".join(phrases) + " ."


def merge_rois(rois, iou_th=0.2):
    merged = []
    for roi in rois:
        placed = False
        for idx, cur in enumerate(merged):
            if box_iou(roi, cur) >= iou_th:
                merged[idx] = [
                    min(cur[0], roi[0]),
                    min(cur[1], roi[1]),
                    max(cur[2], roi[2]),
                    max(cur[3], roi[3]),
                ]
                placed = True
                break
        if not placed:
            merged.append(list(roi))
    return merged


@torch.no_grad()
def detect_products_ground_dino(frame_bgr, text_prompt, conf_th):
    if gdino_model is None or gdino_processor is None:
        return []
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = gdino_processor(images=pil, text=text_prompt, return_tensors="pt").to(dino_device)
        outputs = gdino_model(**inputs)
        target_sizes = torch.tensor([[pil.height, pil.width]], device=dino_device)
        processed = gdino_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=float(conf_th),
            text_threshold=0.25,
            target_sizes=target_sizes,
        )
        if not processed:
            return []
        out = processed[0]
        boxes = out.get("boxes", [])
        scores = out.get("scores", [])
        labels = out.get("labels", [])
        dets = []
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [int(v) for v in b.detach().cpu().tolist()]
            dets.append({
                "box": [x1, y1, x2, y2],
                "cls_name": str(l),
                "conf": float(s.detach().cpu().item()),
                "track_id": -1,
            })
        return dets
    except Exception:
        return []


def assess_item_crop_quality(frame, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    if w < ITEM_EMB_MIN_SIDE or h < ITEM_EMB_MIN_SIDE:
        return False, 0.0, 0.0, "small_side"
    if area < ITEM_EMB_MIN_AREA:
        return False, 0.0, 0.0, "small_area"
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0.0, 0.0, "empty_crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    if sharp < ITEM_EMB_MIN_SHARPNESS:
        return False, sharp, tenengrad, "blurry_lap"
    if tenengrad < ITEM_EMB_MIN_TENENGRAD:
        return False, sharp, tenengrad, "blurry_tenengrad"
    return True, sharp, tenengrad, "ok"


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))

# Defaults are tuned for a 960-width resized frame.
KIOSK_1_BOX_STR = os.environ.get("KIOSK_1_BOX", "0,250,180,540")
KIOSK_2_BOX_STR = os.environ.get("KIOSK_2_BOX", "180,230,430,540")
EXIT_BOX_STR = os.environ.get("KIOSK_EXIT_BOX", "760,0,959,539")


def parse_box(s):
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"Invalid box spec: {s}")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box dimensions: {s}")
    return [x1, y1, x2, y2]


def expand_box(box, pad, w, h):
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w - 1, x2 + pad),
        min(h - 1, y2 + pad),
    ]


def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)


def box_center_in_box(box, target_box):
    cx, cy = box_center(box)
    return point_in_box(cx, cy, target_box)


def union_boxes(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def points_to_box(points, pad, w, h):
    pts = [(int(x), int(y)) for x, y in points if x is not None and y is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [
        max(0, min(xs) - int(pad)),
        max(0, min(ys) - int(pad)),
        min(w - 1, max(xs) + int(pad)),
        min(h - 1, max(ys) + int(pad)),
    ]


def extract_box_signature(frame, box, size=16):
    if frame is None or box is None:
        return None
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    thumb = thumb.reshape(-1)
    norm = float(np.linalg.norm(thumb)) + 1e-9
    return thumb / norm


def extract_image_signature(image_path, size=24):
    if not image_path or not os.path.isfile(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    thumb = thumb.reshape(-1)
    norm = float(np.linalg.norm(thumb)) + 1e-9
    return thumb / norm


def torso_bag_zone(person_box):
    if person_box is None:
        return None
    x1, y1, x2, y2 = map(int, person_box)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return [
        int(x1 + 0.18 * w),
        int(y1 + 0.28 * h),
        int(x1 + 0.82 * w),
        int(y1 + 0.96 * h),
    ]


def _normalized_evidence_tag(tag):
    norm = str(tag or "").strip().lower()
    if norm.startswith("pre_exit_carry"):
        return "pre_exit_carry"
    return norm


def _evidence_tag_family(tag):
    norm = _normalized_evidence_tag(tag)
    if norm in ("hand_hold", "hand_hold_weak", "body_carry"):
        return "hand_hold"
    if norm in ("person_exit", "pre_exit_carry"):
        return "exit_view"
    return norm


def _cluster_evidence_records(sorted_records):
    clusters = []
    current = []
    current_family = None
    current_last_frame = None
    for rec in sorted_records:
        family = _evidence_tag_family(rec.get("tag"))
        frame_idx = int(rec.get("frame", -1))
        same_cluster = (
            current
            and family == current_family
            and current_last_frame is not None
            and frame_idx - current_last_frame <= GROUP_EVIDENCE_CLUSTER_MAX_GAP
        )
        if same_cluster:
            current.append(rec)
        else:
            if current:
                clusters.append(current)
            current = [rec]
            current_family = family
        current_last_frame = frame_idx
    if current:
        clusters.append(current)
    return clusters


def _evidence_tag_priority(tag):
    norm = _normalized_evidence_tag(tag)
    priorities = {
        "pre_exit_carry": 50,
        "hand_hold": 40,
        "hand_hold_weak": 30,
        "body_carry": 20,
        "person_exit": 10,
    }
    return int(priorities.get(norm, 0))


def select_diverse_evidence_records(evidence_records, fps, max_images=None):
    if max_images is not None and max_images <= 0:
        return []
    sorted_records = sorted(
        evidence_records,
        key=lambda rec: int(rec.get("frame", -1)),
    )
    clusters = _cluster_evidence_records(sorted_records)
    chosen = []
    chosen_keys = set()
    prev_sig = None
    prev_frame = -10_000

    def rec_key(rec):
        return (int(rec.get("frame", -1)), str(rec.get("image", "")))

    def try_add(rec, force=False):
        nonlocal prev_sig, prev_frame
        key = rec_key(rec)
        if key in chosen_keys:
            return
        cur_frame = int(rec.get("frame", -1))
        if not force and cur_frame - prev_frame < GROUP_EVIDENCE_MIN_FRAME_GAP:
            return
        sig = extract_image_signature(rec.get("image"))
        if not force and sig is not None and prev_sig is not None:
            if emb_cos_sim(sig, prev_sig) >= GROUP_EVIDENCE_DIVERSE_SIM:
                return
        chosen.append(rec)
        chosen_keys.add(key)
        prev_sig = sig
        prev_frame = cur_frame

    for cluster in clusters:
        if not cluster:
            continue
        first = cluster[0]
        last = cluster[-1]
        family = _evidence_tag_family(first.get("tag"))
        force_boundaries = family in ("hand_hold", "exit_view")
        late_strong_hand_hold = None
        if family == "hand_hold":
            for rec in reversed(cluster):
                if _normalized_evidence_tag(rec.get("tag", "")) == "hand_hold":
                    late_strong_hand_hold = rec
                    break
        forced_records = []
        if family == "exit_view":
            for rec in cluster:
                if _normalized_evidence_tag(rec.get("tag", "")) == "pre_exit_carry":
                    forced_records.append(rec)
        try_add(first, force=force_boundaries)
        for rec in cluster[1:-1]:
            try_add(rec, force=False)
        if late_strong_hand_hold is not None:
            try_add(late_strong_hand_hold, force=True)
        for rec in forced_records:
            try_add(rec, force=True)
        if len(cluster) > 1:
            try_add(last, force=force_boundaries)

    chosen = sorted(chosen, key=lambda rec: int(rec.get("frame", -1)))

    by_frame = {}
    for rec in chosen:
        frame_idx = int(rec.get("frame", -1))
        prev = by_frame.get(frame_idx)
        if prev is None:
            by_frame[frame_idx] = rec
            continue
        prev_score = (
            _evidence_tag_priority(prev.get("tag", "")),
            len(str(prev.get("image", ""))),
        )
        cur_score = (
            _evidence_tag_priority(rec.get("tag", "")),
            len(str(rec.get("image", ""))),
        )
        if cur_score > prev_score:
            by_frame[frame_idx] = rec
    chosen = [by_frame[k] for k in sorted(by_frame.keys())]

    if max_images is not None:
        forced = [
            rec for rec in chosen
            if _normalized_evidence_tag(rec.get("tag", "")) == "pre_exit_carry"
        ]
        regular = [
            rec for rec in chosen
            if _normalized_evidence_tag(rec.get("tag", "")) != "pre_exit_carry"
        ]
        if len(forced) >= max_images:
            return forced[:max_images]
        return forced + regular[: max_images - len(forced)]
    return chosen


def is_same_hand_item(curr_box, curr_sig, prev_box, prev_sig):
    iou_ok = False
    sim_ok = False
    if curr_box is not None and prev_box is not None:
        iou_ok = box_iou(curr_box, prev_box) >= FAST_HAND_ITEM_SAME_IOU
    if curr_sig is not None and prev_sig is not None:
        sim_ok = emb_cos_sim(curr_sig, prev_sig) >= FAST_HAND_ITEM_SAME_SIM
    return iou_ok or sim_ok


def point_to_box_distance(x, y, box):
    x1, y1, x2, y2 = box
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0.0
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def expand_xyxy(box, pad, w, h):
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w - 1, x2 + pad),
        min(h - 1, y2 + pad),
    ]


def product_box_size_ok(box):
    x1, y1, x2, y2 = map(int, box)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    return (
        w >= PRODUCT_MIN_SIDE
        and h >= PRODUCT_MIN_SIDE
        and area >= PRODUCT_MIN_AREA
    ), w, h, area


def product_det_keepable(det):
    cls_name = " ".join(str(det.get("cls_name", "")).split())
    conf = float(det.get("conf", 0.0))
    if cls_name:
        return True, ""
    if conf < PRODUCT_EMPTY_LABEL_MIN_CONF:
        return False, f"weak_empty_label conf={conf:.3f} min={PRODUCT_EMPTY_LABEL_MIN_CONF:.2f}"
    return True, ""


def detect_handheld_object_box(frame, roi_box):
    if frame is None or roi_box is None:
        return None
    x1, y1, x2, y2 = map(int, roi_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    names_map = person_model.names if hasattr(person_model, "names") else {}

    def pick_best(res, allow_any=False):
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return None
        best_box = None
        best_conf = -1.0
        cls_arr = res[0].boxes.cls.cpu().numpy().astype(int) if res[0].boxes.cls is not None else None
        for idx, (pb, conf) in enumerate(zip(res[0].boxes.xyxy.cpu().numpy(), res[0].boxes.conf.cpu().numpy())):
            cid = int(cls_arr[idx]) if cls_arr is not None and idx < len(cls_arr) else -1
            if allow_any:
                if cid == 0:
                    continue
            bx1, by1, bx2, by2 = map(int, pb)
            bw = max(0, bx2 - bx1)
            bh = max(0, by2 - by1)
            area = bw * bh
            if bw < FAST_HOLD_CONFIRM_MIN_SIDE or bh < FAST_HOLD_CONFIRM_MIN_SIDE or area < FAST_HOLD_CONFIRM_MIN_AREA:
                continue
            if allow_any:
                cname = str(names_map.get(cid, "")).lower()
                if cname in ("person",):
                    continue
            c = float(conf)
            if c > best_conf:
                best_conf = c
                best_box = [int(bx1 + x1), int(by1 + y1), int(bx2 + x1), int(by2 + y1)]
        return best_box

    try:
        res = person_model(roi, classes=PRODUCT_CLASS_IDS, conf=FAST_HOLD_CONFIRM_CONF, iou=0.5, verbose=False)
        best = pick_best(res, allow_any=False)
        if best is not None:
            return best
        if FAST_HOLD_CONFIRM_FALLBACK_ENABLE:
            res_any = person_model(roi, conf=FAST_HOLD_CONFIRM_CONF, iou=0.5, verbose=False)
            return pick_best(res_any, allow_any=True)
    except Exception:
        return None
    return None


def save_product_crop(frame, box, out_dir, gid, item_id, vname, frame_count, label, track_id=None):
    if not out_dir:
        return None
    gid_dir = os.path.join(out_dir, f"ID{gid}")
    item_dir = os.path.join(gid_dir, f"item_{int(item_id):03d}")
    os.makedirs(item_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    track_part = f"_T{int(track_id)}" if track_id is not None else ""
    path = os.path.join(item_dir, f"{vname}_F{frame_count:06d}_{label}{track_part}.jpg")
    cv2.imwrite(path, frame[y1:y2, x1:x2])
    return path


def save_product_crop_unknown(frame, box, out_dir, vname, frame_count, label, track_id=None):
    if not out_dir:
        return None
    unk_dir = os.path.join(out_dir, "Unknown")
    os.makedirs(unk_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    track_part = f"_T{int(track_id)}" if track_id is not None else ""
    path = os.path.join(unk_dir, f"{vname}_F{frame_count:06d}_{label}{track_part}.jpg")
    cv2.imwrite(path, frame[y1:y2, x1:x2])
    return path


def save_item_full_frame(frame, box, out_dir, gid, item_id, vname, frame_count, label):
    if not out_dir:
        return None
    gid_dir = os.path.join(out_dir, f"ID{gid}")
    item_dir = os.path.join(gid_dir, f"item_{int(item_id):03d}")
    os.makedirs(item_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, box)
    snap = frame.copy()
    color = (0, 255, 255)
    cv2.rectangle(snap, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} item_{int(item_id):03d}"
    cv2.putText(snap, txt, (max(0, x1), max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    path = os.path.join(item_dir, f"{vname}_F{frame_count:06d}_{label}_full.jpg")
    cv2.imwrite(path, snap)
    return path


def get_person_hand_points(frame, person_box):
    """Return list of wrist points [(x,y), ...] in full-frame coords."""
    if pose_model is None:
        return []
    x1, y1, x2, y2 = map(int, person_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return []
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    try:
        res = pose_model(crop, conf=0.25, verbose=False)
    except Exception:
        return []
    if not res or res[0].keypoints is None or res[0].keypoints.data is None:
        return []
    k = res[0].keypoints
    if k.xy is None or k.conf is None or k.xy.shape[0] == 0:
        return []
    confs = k.conf.detach().cpu().numpy()
    idx = int(np.argmax(confs.mean(axis=1)))
    xy = k.xy[idx].detach().cpu().numpy()  # (17,2)
    cf = confs[idx]
    wrists = []
    for wi in (9, 10):  # COCO wrists
        if wi < len(cf) and cf[wi] >= POSE_KPT_CONF_TH:
            wx, wy = xy[wi]
            wrists.append((int(x1 + wx), int(y1 + wy)))
    return wrists


def find_videos(folder):
    exts = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def clear_output_folder(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def move_finished_video(work_dir, video_dir, video_path):
    vname = os.path.splitext(os.path.basename(video_path))[0]
    src = os.path.join(work_dir, f"{vname}_output.mp4")
    dst = os.path.join(video_dir, f"{vname}_output.mp4")
    if os.path.isfile(src):
        os.makedirs(video_dir, exist_ok=True)
        shutil.move(src, dst)


def _upper_name(path):
    return os.path.basename(path).upper()


def _leading_number(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(\d+)", stem)
    return int(m.group(1)) if m else 10**9


def _video_kind(path):
    upper = _upper_name(path)
    if "KIOSK" in upper:
        return "kiosk"
    if "ENTRY" in upper:
        return "entry"
    return "entry"


def find_session_dirs(root_dir):
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
            and name.lower().startswith("session_")
            and name.lower() not in DETECT_SKIP_SESSION_NAMES
        ]
    )


def group_id_for_person(gid):
    return int(DEFAULT_GROUP_ID) if GROUP_TRACKING_ENABLE else int(gid)


def item_counts_as_with_customer(item_state):
    return item_state in ("with_customer", "with_customer_hidden", "carried_out")


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, set):
        return [make_json_safe(v) for v in sorted(value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonicalize_summary_item_type(item_type):
    raw = " ".join(str(item_type or "item").strip().lower().split())
    if not raw:
        return "item"

    alias_groups = [
        (
            "water bottle multipack",
            (
                "large water bottle multipack",
                "packaged water (multi-bottle pack)",
                "multi-bottle pack of water",
                "packaged water",
                "water multipack",
                "water bottle pack",
                "bottled water pack",
                "pack of water bottles",
            ),
        ),
        (
            "red-capped bottle",
            (
                "bottle (red cap, beverage or cleaning liquid)",
                "red cap bottle",
                "red-capped bottle",
                "red capped bottle",
            ),
        ),
        (
            "snack bar",
            (
                "snack bar or candy bar",
                "candy bar",
                "snack bar",
                "small snack bar",
            ),
        ),
    ]
    for canonical, aliases in alias_groups:
        if raw == canonical:
            return canonical
        for alias in aliases:
            if raw == alias:
                return canonical

    if "water" in raw and any(token in raw for token in ("multipack", "multi-bottle", "bottle pack", "bottled water", "pack")):
        return "water bottle multipack"
    if "red" in raw and "bottle" in raw and "cap" in raw:
        return "red-capped bottle"
    if "snack" in raw and "bar" in raw:
        return "snack bar"
    if "candy" in raw and "bar" in raw:
        return "snack bar"
    return raw


def _normalize_summary_items(items):
    normalized = []
    for item in items or []:
        try:
            count = int(item.get("count", 0))
        except Exception:
            count = 0
        normalized.append({
            "type": _canonicalize_summary_item_type(item.get("type", "item")),
            "count": max(0, count),
            "confidence": float(item.get("confidence", 0.0)),
        })
    return normalized


def frame_to_timestamp(frame_idx, fps):
    try:
        fps_value = float(fps)
    except Exception:
        fps_value = 0.0
    if fps_value <= 0.0 or frame_idx is None:
        return None
    total_seconds = max(0.0, float(frame_idx) / fps_value)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000.0))
    if milliseconds >= 1000:
        milliseconds -= 1000
        seconds += 1
        if seconds >= 60:
            seconds -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def build_timestamp_range(frames, fps):
    clean_frames = []
    for frame in frames or []:
        try:
            clean_frames.append(int(frame))
        except Exception:
            continue
    if not clean_frames:
        return None
    start_frame = min(clean_frames)
    end_frame = max(clean_frames)
    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": frame_to_timestamp(start_frame, fps),
        "end_time": frame_to_timestamp(end_frame, fps),
    }


def build_timestamp_window(start_frames, end_frames, fps):
    start_values = []
    end_values = []
    for frame in start_frames or []:
        try:
            start_values.append(int(frame))
        except Exception:
            continue
    for frame in end_frames or []:
        try:
            end_values.append(int(frame))
        except Exception:
            continue
    if not start_values and not end_values:
        return None
    if not start_values:
        start_values = list(end_values)
    if not end_values:
        end_values = list(start_values)
    start_frame = min(start_values)
    end_frame = max(end_values)
    if end_frame < start_frame:
        end_frame = start_frame
    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": frame_to_timestamp(start_frame, fps),
        "end_time": frame_to_timestamp(end_frame, fps),
    }


def _is_kiosk_interaction_event(event_name):
    return str(event_name or "") in {
        "HandHoldView",
        "HandHoldWeakView",
        "BagPocketView",
        "ItemPicked",
        "ItemRepicked",
        "ItemHeld",
        "ItemHiddenOnCustomer",
        "ItemOccludedAtKiosk",
        "ItemPlacedAtKiosk",
        "ItemCarriedOut",
    }


def build_kiosk_group_event(video_name, group_id, summary, member_ids, event_timeline=None, fps=0.0):
    summary = summary or {}
    persons = []
    overall_total = 0
    per_person_items = summary.get("per_person_items") or []
    fallback_member_ids = [int(pid) for pid in member_ids or []]
    timeline = list(event_timeline or [])
    interaction_timeline = [
        ev for ev in timeline
        if _is_kiosk_interaction_event(ev.get("event"))
    ]
    group_exit_frames = [
        ev.get("frame")
        for ev in timeline
        if str(ev.get("event", "")) == "Exit"
    ]
    group_timestamp_range = build_timestamp_window(
        [ev.get("frame") for ev in interaction_timeline],
        group_exit_frames,
        fps,
    )
    if group_timestamp_range is None:
        group_timestamp_range = build_timestamp_range(
            [ev.get("frame") for ev in timeline],
            fps,
        )
    if not per_person_items and fallback_member_ids:
        per_person_items = [{"person_id": pid} for pid in fallback_member_ids]

    for idx, person in enumerate(per_person_items):
        visible_items = _normalize_summary_items(person.get("visible_items"))
        visible_total = sum(int(item["count"]) for item in visible_items)
        try:
            suspected_hidden_count = int(person.get("suspected_hidden_count", 0))
        except Exception:
            suspected_hidden_count = 0
        carried_out_count = person.get("carried_out_count")
        try:
            carried_out_count = int(carried_out_count) if carried_out_count is not None else visible_total
        except Exception:
            carried_out_count = visible_total
        carried_out_count = max(carried_out_count, visible_total)
        person_id = person.get("person_id")
        if person_id is None and idx < len(fallback_member_ids):
            person_id = fallback_member_ids[idx]
        person_timeline = []
        if person_id is not None:
            for ev in timeline:
                try:
                    if int(ev.get("person_id")) == int(person_id):
                        person_timeline.append(ev)
                except Exception:
                    continue
        person_interaction_timeline = [
            ev for ev in person_timeline
            if _is_kiosk_interaction_event(ev.get("event"))
        ]
        person_exit_frames = [
            ev.get("frame")
            for ev in person_timeline
            if str(ev.get("event", "")) == "Exit"
        ]
        persons.append({
            "person_id": int(person_id) if person_id is not None else None,
            "items_taken_out": visible_items,
            "total_items_taken_out": int(carried_out_count),
            "suspected_hidden_count": max(0, suspected_hidden_count),
            "confidence": float(person.get("confidence", summary.get("confidence", 0.0))),
            "reasoning_summary": str(person.get("reasoning_summary", summary.get("reasoning_summary", ""))),
            "timestamp_range": build_timestamp_window(
                [ev.get("frame") for ev in person_interaction_timeline],
                person_exit_frames,
                fps,
            ) if person_interaction_timeline else group_timestamp_range,
        })
        overall_total += int(carried_out_count)

    if overall_total <= 0:
        try:
            overall_total = int(summary.get("suspected_total_count", summary.get("confirmed_visible_count", 0)))
        except Exception:
            overall_total = 0

    return {
        "event": "KioskCarryOutSummary",
        "video": str(video_name),
        "group_id": int(group_id),
        "timestamp_range": group_timestamp_range,
        "persons": persons,
        "total_items_taken_out": int(overall_total),
    }


def save_group_evidence_frame(
    frame,
    out_dir,
    group_id,
    frame_count,
    tag,
    note,
    primary_box=None,
    secondary_box=None,
    person_id=None,
    kiosk_label=None,
    suffix="",
):
    group_dir = os.path.join(out_dir, f"group_{int(group_id):03d}")
    os.makedirs(group_dir, exist_ok=True)
    snap = frame.copy()
    path = os.path.join(group_dir, f"F{int(frame_count):06d}_{tag}{suffix}.jpg")
    cv2.imwrite(path, snap)
    return path


def extract_json_object(text):
    text = str(text or "").strip()
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:idx + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


def try_repair_json_by_trimming_lines(text):
    text = str(text or "").strip()
    if not text:
        return None, ""
    start = text.find("{")
    if start < 0:
        return None, ""
    core = text[start:].strip()
    lines = core.splitlines()
    if not lines:
        return None, ""

    for i in range(len(lines), 0, -1):
        prefix = "\n".join(lines[:i]).rstrip()
        open_count = prefix.count("{")
        close_count = prefix.count("}")
        missing_closes = max(0, open_count - close_count)
        suffixes = [""]
        if missing_closes > 0:
            suffixes.append("\n" + ("}" * missing_closes))
            suffixes.append("}" * missing_closes)
        for suffix in suffixes:
            candidate = (prefix + suffix).strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate), candidate
            except Exception:
                continue
    return None, ""


def build_group_vlm_prompt(group_id, member_ids, event_timeline):
    short_timeline = []
    for ev in event_timeline[:20]:
        short_timeline.append(
            {
                "frame": int(ev.get("frame", -1)),
                "event": str(ev.get("event", "")),
                "label": ev.get("label"),
                "item_id": ev.get("item_id"),
                "person_id": ev.get("person_id"),
            }
        )
    return (
        f"You are reviewing evidence images for retail loss analysis.\n"
        f"Group ID: {int(group_id)}\n"
        f"Known member IDs: {', '.join(str(int(x)) for x in sorted(member_ids)) or 'unknown'}\n"
        f"Timeline hints: {json.dumps(short_timeline, ensure_ascii=True)}\n\n"
        f"Return strict JSON only with this schema:\n"
        f"{{"
        f"\"group_id\": {int(group_id)}, "
        f"\"left_store\": true|false, "
        f"\"confirmed_visible_count\": integer, "
        f"\"suspected_total_count\": integer, "
        f"\"visible_items\": [{{\"type\": string, \"count\": integer, \"confidence\": number}}], "
        f"\"per_person_items\": [{{\"person_id\": integer|null, \"status\": string, \"visible_items\": [{{\"type\": string, \"count\": integer, \"confidence\": number}}], \"suspected_hidden_count\": integer, \"carried_out_count\": integer, \"confidence\": number}}], "
        f"\"customers_left_with_items\": [{{\"person_id\": integer|null, \"carried_out_count\": integer, \"confidence\": number}}], "
        f"\"hidden_item_suspected\": true|false, "
        f"\"confidence\": number, "
        f"\"reasoning_summary\": string"
        f"}}\n"
        f"Important rules:\n"
        f"1. Multiple customers may be present. Do not assume all visible items belong to one customer.\n"
        f"2. Track items per person whenever possible using the member IDs and timeline hints.\n"
        f"3. Count items that were carried out by a customer even if that customer already left the store before the last frame.\n"
        f"4. Use the final frames as strong evidence, but also use earlier frames if they show a customer leaving with their own items.\n"
        f"5. Do not double-count the same item across multiple frames or across multiple people.\n"
        f"6. If uncertain, be conservative in confirmed_visible_count and use suspected_total_count for likely hidden items.\n"
        f"7. In per_person_items, set status to one of: in_store, left_store, or uncertain.\n"
        f"Focus on identifying which items each customer carried out, and the overall total carried out by the group."
    )

def build_empty_group_summary(group_id):
    return {
        "group_id": int(group_id),
        "left_store": False,
        "confirmed_visible_count": 0,
        "suspected_total_count": 0,
        "visible_items": [],
        "per_person_items": [],
        "customers_left_with_items": [],
        "hidden_item_suspected": False,
        "confidence": 0.0,
        "reasoning_summary": "No evidence images were available for this group.",
    }


def validate_group_summary_payload(payload, expected_group_id=None):
    if not isinstance(payload, dict):
        return False, "not_object"
    required_keys = [
        "group_id",
        "left_store",
        "confirmed_visible_count",
        "suspected_total_count",
        "visible_items",
        "per_person_items",
        "customers_left_with_items",
        "hidden_item_suspected",
        "confidence",
        "reasoning_summary",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return False, f"missing_keys:{','.join(missing)}"
    if expected_group_id is not None:
        try:
            if int(payload.get("group_id")) != int(expected_group_id):
                return False, f"group_id_mismatch:{payload.get('group_id')}"
        except Exception:
            return False, "group_id_invalid"
    for list_key in ("visible_items", "per_person_items", "customers_left_with_items"):
        if not isinstance(payload.get(list_key), list):
            return False, f"{list_key}_not_list"
    if not isinstance(payload.get("reasoning_summary"), str):
        return False, "reasoning_summary_not_string"
    return True, "ok"


def encode_image_as_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def encode_image_as_base64_payload(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return mime_type, encoded


def strip_markdown_code_fence(text):
    text = str(text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_from_messy_text(text):
    candidates = []
    raw_text = str(text or "").strip()
    if not raw_text:
        return None, ""
    candidates.append(raw_text)

    stripped = strip_markdown_code_fence(raw_text)
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    # Handle partial leading fence such as "```json" without a closing fence.
    partial = re.sub(r"^\s*```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE).strip()
    if partial and partial not in candidates:
        candidates.append(partial)

    for candidate in candidates:
        parsed = extract_json_object(candidate)
        if parsed is not None:
            return parsed, candidate
        repaired, repaired_text = try_repair_json_by_trimming_lines(candidate)
        if repaired is not None:
            return repaired, repaired_text
    return None, stripped if stripped else raw_text


def dump_anthropic_failure(prompt, raw_response, raw_text, cleaned_text):
    debug_dir = os.path.join(BASE_DIR, "anthropic_failures")
    os.makedirs(debug_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(debug_dir, f"anthropic_failure_{stamp}.json")
    payload = {
        "provider": "anthropic",
        "model": GROUP_ANTHROPIC_MODEL,
        "prompt": str(prompt or ""),
        "raw_response": str(raw_response or ""),
        "raw_text": str(raw_text or ""),
        "cleaned_text": str(cleaned_text or ""),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def dump_gemini_failure(prompt, raw_response, raw_text, cleaned_text):
    debug_dir = os.path.join(BASE_DIR, "gemini_failures")
    os.makedirs(debug_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(debug_dir, f"gemini_failure_{stamp}.json")
    payload = {
        "provider": "gemini",
        "model": GROUP_GEMINI_MODEL,
        "prompt": prompt,
        "raw_response": raw_response,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def extract_video_frame_image(video_path, frame_idx, image_path):
    cap = cv2.VideoCapture(video_path)
    try:
        try:
            frame_idx = max(0, int(frame_idx))
        except Exception:
            frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None or frame.size == 0:
        return None
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, frame)
    return image_path


def run_openai_timestamp_ocr(image_path):
    if not group_openai_api_key or not image_path or not os.path.isfile(image_path):
        return None
    payload = {
        "model": GROUP_OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read only the CCTV timestamp shown at the top-right of the image. "
                            "Return strict JSON only in this schema: "
                            "{\"timestamp_text\": string|null}. "
                            "Copy the visible timestamp exactly. "
                            "If unreadable, return null."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_as_data_url(image_path)},
                    },
                ],
            }
        ],
        "max_completion_tokens": 128,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{GROUP_OPENAI_BASE_URL.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {group_openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GROUP_OPENAI_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        raw_text = data["choices"][0]["message"]["content"].strip()
        parsed = extract_json_object(raw_text)
        if parsed is None:
            return None
        timestamp_text = parsed.get("timestamp_text")
        if timestamp_text is None:
            return None
        timestamp_text = str(timestamp_text).strip()
        return timestamp_text or None
    except Exception:
        return None


def load_json_file(path, default=None):
    if not path or not os.path.isfile(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def parse_session_answer_key(answer_path):
    if not answer_path or not os.path.isfile(answer_path):
        return None
    answers = {}
    try:
        with open(answer_path, "r") as f:
            for raw_line in f:
                line = str(raw_line or "").strip()
                if not line:
                    continue
                m = re.match(r"^(session_\d+)\s*:\s*(\d+)\s*$", line, flags=re.IGNORECASE)
                if not m:
                    continue
                answers[str(m.group(1)).lower()] = int(m.group(2))
    except Exception:
        return None
    return answers


def build_overall_session_answer_comparison(sessions_root, session_dirs, answer_path):
    answer_key = parse_session_answer_key(answer_path)
    if not answer_key:
        return None

    by_model = {}
    per_session = {}

    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir).lower()
        runtime_output_base = os.path.join(session_dir, "output")
        logs_output_dir = os.path.join(runtime_output_base, LOGS_OUTPUT_DIRNAME)
        expected = answer_key.get(session_name)
        per_session.setdefault(session_name, {
            "expected_total_items": expected,
            "models": {},
        })
        if not os.path.isdir(logs_output_dir):
            continue
        for name in sorted(os.listdir(logs_output_dir)):
            if not name.lower().startswith("result_") or not name.lower().endswith(".json"):
                continue
            path = os.path.join(logs_output_dir, name)
            if not os.path.isfile(path):
                continue
            payload = load_json_file(path, default=None)
            if not isinstance(payload, dict):
                continue
            predicted = payload.get("total_item_for_all_person")
            try:
                predicted = int(predicted)
            except Exception:
                continue
            model_key = os.path.splitext(name)[0]
            match = None if expected is None else int(predicted) == int(expected)
            per_session[session_name]["models"][model_key] = {
                "result_path": path,
                "predicted_total_items": int(predicted),
                "match": match,
            }
            model_row = by_model.setdefault(model_key, {
                "model_key": model_key,
                "sessions": {},
                "exact_match_count": 0,
                "comparable_session_count": 0,
                "accuracy": None,
            })
            model_row["sessions"][session_name] = {
                "expected_total_items": expected,
                "predicted_total_items": int(predicted),
                "match": match,
                "result_path": path,
            }
            if expected is not None:
                model_row["comparable_session_count"] += 1
                if match:
                    model_row["exact_match_count"] += 1

    models = []
    for model_key, row in by_model.items():
        comparable = int(row.get("comparable_session_count", 0))
        exact = int(row.get("exact_match_count", 0))
        row["accuracy"] = round(float(exact) / float(comparable), 6) if comparable > 0 else None
        models.append(row)

    models = sorted(
        models,
        key=lambda item: (
            -1 if item.get("accuracy") is None else -float(item.get("accuracy")),
            str(item.get("model_key", "")),
        ),
    )

    payload = {
        "answer_path": answer_path,
        "sessions_root": sessions_root,
        "session_count_in_answer_key": len(answer_key),
        "models": make_json_safe(models),
        "per_session": make_json_safe(per_session),
    }
    output_path = os.path.join(sessions_root, "overall_answer_comparison.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    payload["output_path"] = output_path
    return payload


def current_vlm_provider_and_model(provider_override=None, model_override=None):
    provider = str(provider_override if provider_override is not None else GROUP_VLM_PROVIDER or "").strip().lower()
    if provider == "openai":
        return provider, str(model_override if model_override is not None else GROUP_OPENAI_MODEL or "").strip()
    if provider == "gemini":
        return provider, str(model_override if model_override is not None else GROUP_GEMINI_MODEL or "").strip()
    if provider == "anthropic":
        return provider, str(model_override if model_override is not None else GROUP_ANTHROPIC_MODEL or "").strip()
    return provider, str(model_override if model_override is not None else GROUP_VLM_MODEL_NAME or "").strip()


def _safe_int(value, default=0):
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def normalize_vlm_usage(provider, raw_usage):
    raw_usage = raw_usage or {}
    provider = str(provider or "").strip().lower()
    if provider == "openai":
        input_tokens = _safe_int(raw_usage.get("prompt_tokens"))
        output_tokens = _safe_int(raw_usage.get("completion_tokens"))
        total_tokens = _safe_int(raw_usage.get("total_tokens"), input_tokens + output_tokens)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": _safe_int(((raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens"))),
            "reasoning_tokens": _safe_int(((raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))),
            "raw_usage": raw_usage,
        }
    if provider == "gemini":
        input_tokens = _safe_int(raw_usage.get("promptTokenCount"))
        output_tokens = _safe_int(raw_usage.get("candidatesTokenCount"))
        total_tokens = _safe_int(raw_usage.get("totalTokenCount"), input_tokens + output_tokens)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": _safe_int(raw_usage.get("cachedContentTokenCount")),
            "reasoning_tokens": _safe_int(raw_usage.get("thoughtsTokenCount")),
            "raw_usage": raw_usage,
        }
    if provider == "anthropic":
        input_tokens = _safe_int(raw_usage.get("input_tokens"))
        output_tokens = _safe_int(raw_usage.get("output_tokens"))
        total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": _safe_int(raw_usage.get("cache_read_input_tokens")),
            "reasoning_tokens": 0,
            "raw_usage": raw_usage,
        }
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "raw_usage": raw_usage,
    }


def estimate_vlm_cost_usd(provider, model, usage):
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip().lower()
    usage = usage or {}
    input_tokens = float(_safe_int(usage.get("input_tokens")))
    output_tokens = float(_safe_int(usage.get("output_tokens")))
    cached_input_tokens = float(_safe_int(usage.get("cached_input_tokens")))

    in_rate = None
    out_rate = None
    cached_in_rate = None

    if provider == "openai":
        if model == "gpt-4o-mini":
            in_rate, cached_in_rate, out_rate = 0.60, 0.30, 2.40
        elif model == "gpt-4.1":
            in_rate, cached_in_rate, out_rate = 2.00, 0.50, 8.00
        elif model == "gpt-5.4":
            in_rate, cached_in_rate, out_rate = 2.50, 0.25, 15.00
        elif model == "gpt-5.4-mini":
            in_rate, cached_in_rate, out_rate = 0.75, 0.075, 4.50
    elif provider == "gemini":
        if model == "gemini-3.5-flash":
            in_rate, cached_in_rate, out_rate = 1.50, 0.15, 9.00
        elif model == "gemini-3-flash-preview":
            in_rate, cached_in_rate, out_rate = 0.50, 0.05, 3.00
        elif model == "gemini-3.1-flash-lite":
            in_rate, cached_in_rate, out_rate = 0.25, 0.025, 1.50
        elif model == "gemini-2.5-flash":
            in_rate, cached_in_rate, out_rate = 0.90, 0.09, 5.40
    elif provider == "anthropic":
        if "haiku" in model:
            in_rate, cached_in_rate, out_rate = 0.80, 0.08, 4.00
        elif "sonnet" in model:
            in_rate, cached_in_rate, out_rate = 3.00, 0.30, 15.00
        elif "opus" in model:
            in_rate, cached_in_rate, out_rate = 15.00, 1.50, 75.00

    if in_rate is None or out_rate is None:
        return None

    uncached_input_tokens = max(0.0, input_tokens - cached_input_tokens)
    total_cost = (
        (uncached_input_tokens / 1_000_000.0) * in_rate
        + (cached_input_tokens / 1_000_000.0) * (cached_in_rate if cached_in_rate is not None else in_rate)
        + (output_tokens / 1_000_000.0) * out_rate
    )
    return round(float(total_cost), 8)


def build_vlm_response_meta(provider, model, raw_usage=None, raw_text=None, status="ok"):
    usage = normalize_vlm_usage(provider, raw_usage)
    return {
        "provider": str(provider or ""),
        "model": str(model or ""),
        "status": str(status or "ok"),
        "usage": usage,
        "estimated_cost_usd": estimate_vlm_cost_usd(provider, model, usage),
        "raw_text": raw_text,
    }


def get_video_fps_safe(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()
    try:
        fps = float(fps)
    except Exception:
        fps = 0.0
    return fps if fps > 0.0 else 25.0


def format_session_time_ref(video_name, frame_idx, fps):
    ts = frame_to_timestamp(frame_idx, fps)
    if ts is None:
        return None
    return f"{video_name} {ts}"


def choose_person_reference_image(logs_output_dir, person_id):
    candidates = [
        os.path.join(logs_output_dir, "reid_fashion_views", f"ID{int(person_id)}"),
        os.path.join(logs_output_dir, "reid_views", f"ID{int(person_id)}"),
    ]
    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        images = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(".jpg")
        )
        if images:
            return images[0]
    return None


def build_session_result(runtime_output_base, ordered_videos, all_events, group_summary_suffix=""):
    logs_output_dir = os.path.join(runtime_output_base, LOGS_OUTPUT_DIRNAME)
    time_refs_dir = os.path.join(logs_output_dir, "time_refs")
    os.makedirs(time_refs_dir, exist_ok=True)
    video_order = {
        os.path.basename(vp): idx
        for idx, vp in enumerate(ordered_videos)
    }
    video_paths = {
        os.path.basename(vp): vp
        for vp in ordered_videos
    }
    fps_by_video = {
        os.path.basename(vp): get_video_fps_safe(vp)
        for vp in ordered_videos
    }

    def sort_key(item):
        video_name, frame_idx = item
        return (video_order.get(video_name, 10**9), int(frame_idx))

    def format_ref(item):
        if item is None:
            return None
        video_name, frame_idx = item
        return format_session_time_ref(video_name, frame_idx, fps_by_video.get(video_name, 25.0))

    def resolve_cctv_time(label, item):
        if item is None:
            return None
        video_name, frame_idx = item
        video_path = video_paths.get(video_name)
        if video_path:
            ref_image = os.path.join(
                time_refs_dir,
                f"{label}_{os.path.splitext(video_name)[0]}_F{int(frame_idx):06d}.jpg",
            )
            saved = extract_video_frame_image(video_path, frame_idx, ref_image)
            if saved:
                ocr_time = run_openai_timestamp_ocr(saved)
                if ocr_time:
                    return ocr_time
        return format_ref(item)

    entry_points = []
    leave_points = []
    kiosk_start_points = []
    kiosk_end_points = []

    for vp in ordered_videos:
        video_name = os.path.basename(vp)
        video_kind = _video_kind(vp)
        for ev in all_events.get(video_name, []) or []:
            event_name = str(ev.get("event", ""))
            frame_idx = ev.get("frame")
            if frame_idx is None:
                continue
            if video_kind != "kiosk" and event_name == "Entry":
                entry_points.append((video_name, int(frame_idx)))
            if video_kind != "kiosk" and event_name == "Exit":
                leave_points.append((video_name, int(frame_idx)))
            if video_kind == "kiosk" and _is_kiosk_interaction_event(event_name):
                kiosk_start_points.append((video_name, int(frame_idx)))
                kiosk_end_points.append((video_name, int(frame_idx)))

    group_id = None
    total_items_for_all_person = 0
    persons_map = {}
    has_any_llm_input_images = False
    usage_summary = {
        "provider": None,
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "estimated_cost_usd": 0.0,
        "groups_count": 0,
    }

    if os.path.isdir(logs_output_dir):
        for name in sorted(os.listdir(logs_output_dir)):
            video_dir = os.path.join(logs_output_dir, name)
            if not os.path.isdir(video_dir):
                continue
            group_summary_path = os.path.join(video_dir, f"{name}_group_summaries{group_summary_suffix}.json")
            if not os.path.isfile(group_summary_path):
                group_summary_path = os.path.join(video_dir, f"{name}_group_summaries.json")
            payloads = load_json_file(group_summary_path, default=[]) or []
            for payload in payloads:
                llm_input_images = [
                    path for path in (payload.get("llm_input_images") or [])
                    if path and os.path.isfile(path)
                ]
                if llm_input_images:
                    has_any_llm_input_images = True
                kiosk_summary = payload.get("kiosk_event_summary") or {}
                vlm_meta = payload.get("vlm_meta") or {}
                usage = vlm_meta.get("usage") or {}
                if vlm_meta:
                    if usage_summary["provider"] is None and vlm_meta.get("provider") is not None:
                        usage_summary["provider"] = vlm_meta.get("provider")
                    if usage_summary["model"] is None and vlm_meta.get("model") is not None:
                        usage_summary["model"] = vlm_meta.get("model")
                    usage_summary["input_tokens"] += _safe_int(usage.get("input_tokens"))
                    usage_summary["output_tokens"] += _safe_int(usage.get("output_tokens"))
                    usage_summary["total_tokens"] += _safe_int(usage.get("total_tokens"))
                    usage_summary["cached_input_tokens"] += _safe_int(usage.get("cached_input_tokens"))
                    usage_summary["reasoning_tokens"] += _safe_int(usage.get("reasoning_tokens"))
                    try:
                        usage_summary["estimated_cost_usd"] += float(vlm_meta.get("estimated_cost_usd") or 0.0)
                    except Exception:
                        pass
                    usage_summary["groups_count"] += 1
                if group_id is None and kiosk_summary.get("group_id") is not None:
                    group_id = int(kiosk_summary.get("group_id"))
                try:
                    total_items_for_all_person = max(
                        int(total_items_for_all_person),
                        int(kiosk_summary.get("total_items_taken_out", 0)),
                    )
                except Exception:
                    pass
                for person in kiosk_summary.get("persons") or []:
                    person_id = person.get("person_id")
                    if person_id is None:
                        continue
                    person_id = int(person_id)
                    try:
                        current_total = int(person.get("total_items_taken_out", 0))
                    except Exception:
                        current_total = 0
                    if person_id not in persons_map:
                        persons_map[person_id] = {
                            "id": person_id,
                            "image": choose_person_reference_image(logs_output_dir, person_id),
                            "total_item": current_total,
                            "item_list": _normalize_summary_items(person.get("items_taken_out")),
                        }
                    else:
                        persons_map[person_id]["total_item"] = max(
                            int(persons_map[person_id].get("total_item", 0)),
                            current_total,
                        )
                        existing_items = {
                            (str(item.get("type", "")), int(item.get("count", 0))): item
                            for item in (persons_map[person_id].get("item_list") or [])
                        }
                        for item in _normalize_summary_items(person.get("items_taken_out")):
                            key = (str(item.get("type", "")), int(item.get("count", 0)))
                            existing = existing_items.get(key)
                            if existing is None:
                                existing_items[key] = item
                            else:
                                try:
                                    if float(item.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                                        existing_items[key] = item
                                except Exception:
                                    pass
                        persons_map[person_id]["item_list"] = list(existing_items.values())
                        if not persons_map[person_id].get("image"):
                            persons_map[person_id]["image"] = choose_person_reference_image(logs_output_dir, person_id)

    if group_id is None:
        group_id = 1

    if not has_any_llm_input_images:
        return {
            "status": "no_image",
            "reason": "llm_input_images_empty",
            "group-id": int(group_id),
            "person": [],
            "total_item_for_all_person": 0,
            "enter_time": resolve_cctv_time("enter_time", min(entry_points, key=sort_key)) if entry_points else None,
            "kiosk_start_time": resolve_cctv_time("kiosk_start_time", min(kiosk_start_points, key=sort_key)) if kiosk_start_points else None,
            "kiosk_end_time": resolve_cctv_time("kiosk_end_time", max(kiosk_end_points, key=sort_key)) if kiosk_end_points else None,
            "leave_time": resolve_cctv_time("leave_time", max(leave_points, key=sort_key)) if leave_points else None,
            "vlm_usage_summary": make_json_safe(usage_summary),
        }

    return {
        "group-id": int(group_id),
        "person": [
            {
                "id": int(person["id"]),
                "image": person.get("image"),
                "total_item": int(person.get("total_item", 0)),
                "item_list": make_json_safe(person.get("item_list") or []),
            }
            for person in sorted(persons_map.values(), key=lambda item: int(item["id"]))
        ],
        "total_item_for_all_person": int(total_items_for_all_person),
        "enter_time": resolve_cctv_time("enter_time", min(entry_points, key=sort_key)) if entry_points else None,
        "kiosk_start_time": resolve_cctv_time("kiosk_start_time", min(kiosk_start_points, key=sort_key)) if kiosk_start_points else None,
        "kiosk_end_time": resolve_cctv_time("kiosk_end_time", max(kiosk_end_points, key=sort_key)) if kiosk_end_points else None,
        "leave_time": resolve_cctv_time("leave_time", max(leave_points, key=sort_key)) if leave_points else None,
        "vlm_usage_summary": make_json_safe({
            **usage_summary,
            "estimated_cost_usd": round(float(usage_summary["estimated_cost_usd"]), 8),
        }),
    }


def current_vlm_result_filename(provider_override=None, model_override=None):
    provider = str(provider_override if provider_override is not None else GROUP_VLM_PROVIDER or "").strip().lower()
    if provider == "openai":
        model = str(model_override if model_override is not None else GROUP_OPENAI_MODEL or "").strip().lower()
        if model == "gpt-4o-mini":
            return "result_gpt4o-mini.json"
        if model == "gpt-4.1":
            return "result_gpt4.1.json"
        if model == "gpt-5.4":
            return "result_gpt5.4.json"
        if model == "gpt-5.4-mini":
            return "result_gpt5.4mini.json"
        model_slug = re.sub(r"[^a-z0-9.-]+", "", model) or "openai"
        return f"result_{model_slug}.json"
    if provider == "gemini":
        model = str(model_override if model_override is not None else GROUP_GEMINI_MODEL or "").strip().lower()
        if model == "gemini-3.5-flash":
            return "result_gemini35flash.json"
        if model == "gemini-3.5-flash-preview":
            return "result_gemini35flashpreview.json"
        if model == "gemini-3-flash-preview":
            return "result_gemini3flash.json"
        if model == "gemini-2.5-flash":
            return "result_gemini25flash.json"
        if model == "gemini-3.1-flash-lite":
            return "result_gemini31flashlite.json"
        model_slug = re.sub(r"[^a-z0-9.-]+", "", model) or "gemini"
        return f"result_{model_slug}.json"
    if provider == "anthropic":
        model = str(GROUP_ANTHROPIC_MODEL or "").strip().lower()
        if "haiku" in model:
            return "result_claudehaiku.json"
        if "sonnet" in model:
            return "result_claudesonnet.json"
        if "opus" in model:
            return "result_claudeopus.json"
        model_slug = re.sub(r"[^a-z0-9.-]+", "", model) or "anthropic"
        return f"result_{model_slug}.json"
    provider_slug = re.sub(r"[^a-z0-9.-]+", "", provider) or "vlm"
    return f"result_{provider_slug}.json"


def current_vlm_result_path(runtime_output_base, provider_override=None, model_override=None):
    return os.path.join(
        runtime_output_base,
        LOGS_OUTPUT_DIRNAME,
        current_vlm_result_filename(provider_override=provider_override, model_override=model_override),
    )


def current_vlm_summary_suffix(provider_override=None, model_override=None):
    filename = current_vlm_result_filename(
        provider_override=provider_override,
        model_override=model_override,
    )
    stem = os.path.splitext(filename)[0]
    if stem.startswith("result_"):
        stem = stem[len("result_"):]
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "", stem)
    return f"_{stem}" if stem else ""


def write_vlm_raw_text(group_dir, summary_suffix, vlm_meta):
    if not group_dir:
        return None
    raw_text = ""
    if isinstance(vlm_meta, dict):
        raw_text = str(vlm_meta.get("raw_text") or "")
    if not raw_text.strip():
        return None
    os.makedirs(group_dir, exist_ok=True)
    suffix = str(summary_suffix or "")
    path = os.path.join(group_dir, f"raw_response{suffix}.txt" if suffix else "raw_response.txt")
    with open(path, "w") as f:
        f.write(raw_text)
        if not raw_text.endswith("\n"):
            f.write("\n")
    return path


def write_model_rerun_error(runtime_output_base, provider, model, error, tb_text):
    logs_output_dir = os.path.join(runtime_output_base, LOGS_OUTPUT_DIRNAME)
    os.makedirs(logs_output_dir, exist_ok=True)
    model_slug = re.sub(r"[^a-z0-9.-]+", "", str(model or "").strip().lower()) or "model"
    provider_slug = re.sub(r"[^a-z0-9.-]+", "", str(provider or "").strip().lower()) or "provider"
    path = os.path.join(logs_output_dir, f"rerun_error_{provider_slug}_{model_slug}.txt")
    with open(path, "w") as f:
        f.write(f"provider={provider}\n")
        f.write(f"model={model}\n")
        f.write(f"error={error}\n")
        if tb_text:
            if not str(tb_text).endswith("\n"):
                tb_text = str(tb_text) + "\n"
            f.write(tb_text)
    return path


def active_gemini_models():
    models = [str(GROUP_GEMINI_MODEL or "").strip()]
    models.extend(GROUP_GEMINI_COMPARE_MODELS)
    seen = set()
    ordered = []
    for model in models:
        key = str(model or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(model.strip())
    return ordered


def has_any_result_prefixed_file(runtime_output_base):
    logs_output_dir = os.path.join(runtime_output_base, LOGS_OUTPUT_DIRNAME)
    if not os.path.isdir(logs_output_dir):
        return False
    for name in os.listdir(logs_output_dir):
        if not name.lower().startswith("result_"):
            continue
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(logs_output_dir, name)
        if os.path.isfile(path):
            return True
    return False


def group_dir_has_evidence_images(group_dir):
    if not group_dir or not os.path.isdir(group_dir):
        return False
    for name in os.listdir(group_dir):
        path = os.path.join(group_dir, name)
        if not os.path.isfile(path):
            continue
        lower_name = name.lower()
        if lower_name.endswith((".jpg", ".jpeg", ".png")):
            return True
    return False


def should_rerun_kiosk_llm_only(runtime_output_base, ordered_videos=None):
    if not DETECT_KIOSK_LLM_ONLY_IF_RESULT_EXISTS:
        return False
    if not has_any_result_prefixed_file(runtime_output_base):
        return False

    logs_output_dir = os.path.join(runtime_output_base, LOGS_OUTPUT_DIRNAME)
    kiosk_output_dirs = []
    if ordered_videos:
        for vp in ordered_videos:
            if _video_kind(vp) != "kiosk":
                continue
            vname = os.path.splitext(os.path.basename(vp))[0]
            kiosk_output_dirs.append(os.path.join(logs_output_dir, vname))
    elif os.path.isdir(logs_output_dir):
        for name in sorted(os.listdir(logs_output_dir)):
            path = os.path.join(logs_output_dir, name)
            if os.path.isdir(path) and name.upper().endswith("_KIOSK"):
                kiosk_output_dirs.append(path)

    if not kiosk_output_dirs:
        return False
    return True


def rebuild_llm_input_from_group_dir(group_dir, fps):
    if not group_dir or not os.path.isdir(group_dir):
        return [], []
    llm_input_dir = os.path.join(group_dir, GROUP_LLM_INPUT_DIRNAME)
    os.makedirs(llm_input_dir, exist_ok=True)

    for name in os.listdir(llm_input_dir):
        path = os.path.join(llm_input_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    evidence_records = []
    for name in sorted(os.listdir(group_dir)):
        if name in {GROUP_LLM_INPUT_DIRNAME, "summary.json"}:
            continue
        path = os.path.join(group_dir, name)
        if not os.path.isfile(path):
            continue
        lower_name = name.lower()
        if not lower_name.endswith((".jpg", ".jpeg", ".png")):
            continue
        frame_match = re.search(r"F(\d+)", name)
        tag_match = re.search(r"F\d+_(.+?)\.(jpg|jpeg|png)$", name, re.IGNORECASE)
        evidence_records.append(
            {
                "frame": int(frame_match.group(1)) if frame_match else -1,
                "tag": tag_match.group(1) if tag_match else os.path.splitext(name)[0],
                "image": path,
            }
        )

    selected_records = select_diverse_evidence_records(
        evidence_records,
        fps,
        GROUP_EVIDENCE_MAX_IMAGES if GROUP_EVIDENCE_MAX_IMAGES > 0 else None,
    )
    llm_input_paths = []
    for idx, rec in enumerate(selected_records, start=1):
        src_path = rec.get("image")
        if not src_path or not os.path.isfile(src_path):
            continue
        dst_name = f"{idx:02d}_{os.path.basename(src_path)}"
        dst_path = os.path.join(llm_input_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        llm_input_paths.append(dst_path)
    return selected_records, llm_input_paths


def get_openai_rate_limit_pause_sec(retry_idx):
    schedule = [max(1, int(v)) for v in GROUP_OPENAI_RATE_LIMIT_RETRY_SCHEDULE_SEC] or [60, 300, 600]
    idx = max(0, min(int(retry_idx), len(schedule) - 1))
    return float(schedule[idx])


def get_anthropic_rate_limit_pause_sec(retry_idx):
    schedule = [max(1, int(v)) for v in GROUP_ANTHROPIC_RATE_LIMIT_RETRY_SCHEDULE_SEC] or [60, 300, 600]
    idx = max(0, min(int(retry_idx), len(schedule) - 1))
    return float(schedule[idx])


def get_gemini_rate_limit_pause_sec(retry_idx):
    schedule = [max(1, int(v)) for v in GROUP_GEMINI_RATE_LIMIT_RETRY_SCHEDULE_SEC] or [60, 300, 600]
    idx = max(0, min(int(retry_idx), len(schedule) - 1))
    return float(schedule[idx])


def rerun_existing_kiosk_llm_only(runtime_output_base, ordered_videos, model_override=None, summary_suffix=None):
    logs_output_dir = os.path.join(runtime_output_base, LOGS_OUTPUT_DIRNAME)
    all_events_summary_path = os.path.join(runtime_output_base, "all_events_summary.json")
    all_events = load_json_file(all_events_summary_path, default={}) or {}
    session_name = os.path.basename(os.path.dirname(runtime_output_base)) or os.path.basename(runtime_output_base)
    provider = str(GROUP_VLM_PROVIDER or "").strip().lower()
    summary_suffix = summary_suffix if summary_suffix is not None else current_vlm_summary_suffix(
        provider_override=provider,
        model_override=model_override,
    )
    model_label = str(model_override or "").strip() or (
        GROUP_GEMINI_MODEL if provider == "gemini" else None
    ) or (
        GROUP_OPENAI_MODEL if provider == "openai" else None
    ) or (
        GROUP_ANTHROPIC_MODEL if provider == "anthropic" else None
    ) or "vlm"
    print(
        f"[Detect] kiosk-llm-only rerun session={session_name} "
        f"root={runtime_output_base} model={model_label}"
    )

    for vp in ordered_videos:
        if _video_kind(vp) != "kiosk":
            continue
        vname = os.path.splitext(os.path.basename(vp))[0]
        print(f"[Detect] kiosk-llm-only session={session_name} video={vname}")
        kiosk_output_dir = os.path.join(logs_output_dir, vname)
        group_001_dir = os.path.join(
            kiosk_output_dir,
            GROUP_EVIDENCE_DIRNAME,
            "group_001",
        )
        if not group_dir_has_evidence_images(group_001_dir):
            print(
                f"[Detect] kiosk-llm-only no-evidence session={session_name} video={vname} "
                f"reason=no_existing_group_001_images"
            )
        group_summary_path = os.path.join(kiosk_output_dir, f"{vname}_group_summaries.json")
        payloads = load_json_file(group_summary_path, default=[]) or []
        existing_events = load_json_file(
            os.path.join(kiosk_output_dir, f"{vname}_events.json"),
            default=[],
        ) or []
        fps = get_video_fps_safe(vp)
        new_payloads = []
        seen_gids = set()
        group_ids = []

        for payload in payloads:
            gid = int(payload.get("group_id", 1))
            if gid in seen_gids:
                continue
            seen_gids.add(gid)
            group_ids.append(gid)

        group_evidence_root = os.path.join(kiosk_output_dir, GROUP_EVIDENCE_DIRNAME)
        if os.path.isdir(group_evidence_root):
            for name in sorted(os.listdir(group_evidence_root)):
                m = re.fullmatch(r"group_(\d{3,})", str(name or ""))
                if not m:
                    continue
                gid = int(m.group(1))
                if gid in seen_gids:
                    continue
                seen_gids.add(gid)
                group_ids.append(gid)

        if not group_ids:
            group_ids = [1]

        for gid in group_ids:
            payload = next(
                (
                    item for item in payloads
                    if int(item.get("group_id", 1)) == int(gid)
                ),
                {},
            )
            group_dir = os.path.join(
                kiosk_output_dir,
                GROUP_EVIDENCE_DIRNAME,
                f"group_{int(gid):03d}",
            )
            selected_records, llm_input_images = rebuild_llm_input_from_group_dir(group_dir, fps)
            print(
                f"[Detect] kiosk-llm-only session={session_name} video={vname} group={gid} "
                f"selected={len(selected_records)} llm_input={len(llm_input_images)}"
            )
            member_ids = sorted(
                int(person.get("person_id"))
                for person in ((payload.get("kiosk_event_summary") or {}).get("persons") or [])
                if person.get("person_id") is not None
            )
            timeline = [
                ev for ev in existing_events
                if int(ev.get("group_id", -999999)) == int(gid)
            ]
            prompt = build_group_vlm_prompt(gid, member_ids, timeline)

            if not llm_input_images:
                summary = build_empty_group_summary(gid)
                raw_or_status = build_vlm_response_meta(
                    *current_vlm_provider_and_model(provider_override=provider, model_override=model_override),
                    raw_usage=None,
                    raw_text="no_evidence_images",
                    status="no_evidence_images",
                )
            else:
                summary, raw_or_status = run_group_vlm_summary(
                    llm_input_images,
                    prompt,
                )

            kiosk_group_event = build_kiosk_group_event(
                vname,
                gid,
                summary,
                member_ids,
                event_timeline=timeline,
                fps=fps,
            )
            new_payload = {
                "video": vname,
                "group_id": int(gid),
                "kiosk_event_summary": make_json_safe(kiosk_group_event),
                "vlm_result": make_json_safe(summary),
                "vlm_meta": make_json_safe(raw_or_status),
                "llm_input_images": list(llm_input_images),
            }
            raw_text_path = write_vlm_raw_text(group_dir, summary_suffix, raw_or_status)
            if raw_text_path:
                new_payload["vlm_raw_text_path"] = raw_text_path
            new_payloads.append(new_payload)
            os.makedirs(group_dir, exist_ok=True)
            group_summary_json_name = f"summary{summary_suffix}.json" if summary_suffix else "summary.json"
            with open(os.path.join(group_dir, group_summary_json_name), "w") as gf:
                json.dump(new_payload, gf, indent=2)

        output_group_summary_path = os.path.join(
            kiosk_output_dir,
            f"{vname}_group_summaries{summary_suffix}.json" if summary_suffix else f"{vname}_group_summaries.json",
        )
        with open(output_group_summary_path, "w") as gf:
            json.dump(new_payloads, gf, indent=2)

    session_result = build_session_result(
        runtime_output_base,
        ordered_videos,
        all_events,
        group_summary_suffix=summary_suffix,
    )
    result_payload = make_json_safe(session_result)
    with open(
        current_vlm_result_path(
            runtime_output_base,
            provider_override=provider,
            model_override=model_override,
        ),
        "w",
    ) as f:
        json.dump(result_payload, f, indent=2)
    if not summary_suffix:
        with open(os.path.join(logs_output_dir, "result.json"), "w") as f:
            json.dump(result_payload, f, indent=2)
    return all_events


def run_additional_gemini_model_comparisons(runtime_output_base, ordered_videos, all_events, base_model):
    provider = str(GROUP_VLM_PROVIDER or "").strip().lower()
    if provider != "gemini":
        return
    models = active_gemini_models()
    if len(models) <= 1:
        return
    original_model = GROUP_GEMINI_MODEL
    try:
        for model in models:
            if str(model).strip().lower() == str(base_model).strip().lower():
                continue
            print(f"[Detect] Gemini comparison rerun model={model}")
            globals()["GROUP_GEMINI_MODEL"] = model
            try:
                rerun_existing_kiosk_llm_only(
                    runtime_output_base,
                    ordered_videos,
                    model_override=model,
                    summary_suffix=current_vlm_summary_suffix(
                        provider_override="gemini",
                        model_override=model,
                    ),
                )
            except Exception as e:
                tb_text = traceback.format_exc()
                error_path = write_model_rerun_error(
                    runtime_output_base,
                    "gemini",
                    model,
                    e,
                    tb_text,
                )
                print(
                    f"[Detect] Gemini comparison rerun failed model={model} "
                    f"error={e} error_log={error_path}"
                )
    finally:
        globals()["GROUP_GEMINI_MODEL"] = original_model


def run_group_openai_summary(image_paths, prompt):
    if not group_openai_api_key:
        raise RuntimeError("OpenAI group VLM is mandatory, but API key is unavailable.")
    usable_paths = [path for path in image_paths if path and os.path.isfile(path)]
    if not usable_paths:
        raise RuntimeError("OpenAI group VLM is mandatory, but no evidence images were available for the group.")

    content = [{"type": "text", "text": prompt}]
    for path in usable_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_as_data_url(path),
                },
            }
        )
    max_token_attempts = [
        int(max(128, GROUP_VLM_MAX_NEW_TOKENS)),
        int(max(256, GROUP_VLM_MAX_NEW_TOKENS * 2)),
    ]
    last_error = None
    expected_group_id = None
    m = re.search(r"Group ID:\s*(\d+)", str(prompt or ""))
    if m:
        expected_group_id = int(m.group(1))
    provider, model = current_vlm_provider_and_model(provider_override="openai")

    for max_tokens in max_token_attempts:
        payload = {
            "model": GROUP_OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GROUP_OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {group_openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw = None
        for retry_idx in range(max(1, GROUP_OPENAI_RETRY_COUNT)):
            try:
                with urllib.request.urlopen(req, timeout=GROUP_OPENAI_TIMEOUT_SEC) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and retry_idx + 1 < max(1, GROUP_OPENAI_RETRY_COUNT):
                    pause_sec = get_openai_rate_limit_pause_sec(retry_idx)
                    print(
                        f"[Detect] OpenAI rate limit hit for image summary. "
                        f"Pausing {int(pause_sec)}s before retry {retry_idx + 2}/"
                        f"{max(1, GROUP_OPENAI_RETRY_COUNT)}."
                    )
                    time.sleep(pause_sec)
                    continue
                raise RuntimeError(f"OpenAI API error: HTTP {e.code}: {detail}") from e
            except Exception as e:
                raise RuntimeError(f"OpenAI API request failed: {e}") from e
        if raw is None:
            raise RuntimeError("OpenAI API request failed after retries.")

        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            raw_text = choice["message"]["content"].strip()
            finish_reason = str(choice.get("finish_reason", "")).strip().lower()
            meta = build_vlm_response_meta(provider, model, raw_usage=data.get("usage"), raw_text=raw_text)
        except Exception as e:
            raise RuntimeError(f"OpenAI API returned an unexpected response: {raw}") from e

        parsed = extract_json_object(raw_text)
        if parsed is not None:
            ok, reason = validate_group_summary_payload(parsed, expected_group_id=expected_group_id)
            if ok:
                return parsed, meta
            last_error = f"{raw_text}\n[invalid_summary:{reason}]"
            continue

        last_error = raw_text
        if finish_reason != "length":
            break

    raise RuntimeError(f"OpenAI API returned non-JSON output: {last_error}")


def run_group_anthropic_summary(image_paths, prompt):
    if not group_anthropic_api_key:
        raise RuntimeError("Anthropic group VLM is mandatory, but API key is unavailable.")
    usable_paths = [path for path in image_paths if path and os.path.isfile(path)]
    if not usable_paths:
        raise RuntimeError("Anthropic group VLM is mandatory, but no evidence images were available for the group.")

    content = [{"type": "text", "text": prompt}]
    for path in usable_paths:
        mime_type, encoded = encode_image_as_base64_payload(path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": encoded,
                },
            }
        )
    max_token_attempts = [
        int(max(256, GROUP_VLM_MAX_NEW_TOKENS)),
        int(max(512, GROUP_VLM_MAX_NEW_TOKENS * 2)),
        int(max(768, GROUP_VLM_MAX_NEW_TOKENS * 3)),
    ]
    last_raw = None
    last_raw_text = ""
    last_cleaned_text = ""
    expected_group_id = None
    m = re.search(r"Group ID:\s*(\d+)", str(prompt or ""))
    if m:
        expected_group_id = int(m.group(1))
    provider, model = current_vlm_provider_and_model(provider_override="anthropic")

    for max_tokens in max_token_attempts:
        payload = {
            "model": GROUP_ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GROUP_ANTHROPIC_BASE_URL.rstrip('/')}/messages",
            data=body,
            headers={
                "x-api-key": group_anthropic_api_key,
                "anthropic-version": GROUP_ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        raw = None
        for retry_idx in range(max(1, GROUP_ANTHROPIC_RETRY_COUNT)):
            try:
                with urllib.request.urlopen(req, timeout=GROUP_ANTHROPIC_TIMEOUT_SEC) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and retry_idx + 1 < max(1, GROUP_ANTHROPIC_RETRY_COUNT):
                    pause_sec = get_anthropic_rate_limit_pause_sec(retry_idx)
                    print(
                        f"[Detect] Anthropic rate limit hit for image summary. "
                        f"Pausing {int(pause_sec)}s before retry {retry_idx + 2}/"
                        f"{max(1, GROUP_ANTHROPIC_RETRY_COUNT)}."
                    )
                    time.sleep(pause_sec)
                    continue
                raise RuntimeError(f"Anthropic API error: HTTP {e.code}: {detail}") from e
            except Exception as e:
                raise RuntimeError(f"Anthropic API request failed: {e}") from e
        if raw is None:
            raise RuntimeError("Anthropic API request failed after retries.")

        try:
            data = json.loads(raw)
            blocks = data.get("content") or []
            raw_text = "\n".join(
                str(block.get("text", "")).strip()
                for block in blocks
                if str(block.get("type", "")).strip() == "text"
            ).strip()
            stop_reason = str(data.get("stop_reason", "")).strip().lower()
            meta = build_vlm_response_meta(provider, model, raw_usage=data.get("usage"), raw_text=raw_text)
        except Exception as e:
            raise RuntimeError(f"Anthropic API returned an unexpected response: {raw}") from e

        parsed, cleaned_text = extract_json_from_messy_text(raw_text)
        if parsed is not None:
            ok, reason = validate_group_summary_payload(parsed, expected_group_id=expected_group_id)
            if ok:
                meta["raw_text"] = cleaned_text
                return parsed, meta
            last_raw = raw
            last_raw_text = raw_text
            last_cleaned_text = f"{cleaned_text}\n[invalid_summary:{reason}]"
            continue

        last_raw = raw
        last_raw_text = raw_text
        last_cleaned_text = cleaned_text
        if stop_reason != "max_tokens":
            break

    dump_path = dump_anthropic_failure(prompt, last_raw, last_raw_text, last_cleaned_text)
    raise RuntimeError(
        f"Anthropic API returned non-JSON output: {last_raw_text}\n"
        f"Failure dump: {dump_path}"
    )


def run_group_gemini_summary(image_paths, prompt):
    if not group_gemini_api_key:
        raise RuntimeError("Gemini group VLM is mandatory, but API key is unavailable.")
    usable_paths = [path for path in image_paths if path and os.path.isfile(path)]
    if not usable_paths:
        raise RuntimeError("Gemini group VLM is mandatory, but no evidence images were available for the group.")

    parts = [{"text": prompt}]
    for path in usable_paths:
        mime_type, encoded = encode_image_as_base64_payload(path)
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": encoded,
                }
            }
        )

    max_token_attempts = [
        int(max(512, GROUP_VLM_MAX_NEW_TOKENS)),
        int(max(1024, GROUP_VLM_MAX_NEW_TOKENS * 2)),
        int(max(1536, GROUP_VLM_MAX_NEW_TOKENS * 3)),
        int(max(2048, GROUP_VLM_MAX_NEW_TOKENS * 4)),
    ]
    last_raw = None
    last_raw_text = ""
    last_cleaned_text = ""
    expected_group_id = None
    m = re.search(r"Group ID:\s*(\d+)", str(prompt or ""))
    if m:
        expected_group_id = int(m.group(1))
    provider, model = current_vlm_provider_and_model(provider_override="gemini")

    for max_tokens in max_token_attempts:
        payload = {
            "contents": [
                {
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "thinkingConfig": {
                    "thinkingBudget": GROUP_GEMINI_THINKING_BUDGET,
                },
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GROUP_GEMINI_BASE_URL.rstrip('/')}/models/{GROUP_GEMINI_MODEL}:generateContent?key={urllib.parse.quote(group_gemini_api_key)}",
            data=body,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw = None
        for retry_idx in range(max(1, GROUP_GEMINI_RETRY_COUNT)):
            try:
                with urllib.request.urlopen(req, timeout=GROUP_GEMINI_TIMEOUT_SEC) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                detail_lower = detail.lower()
                if (
                    (e.code == 429 or "resource_exhausted" in detail_lower or "rate limit" in detail_lower)
                    and retry_idx + 1 < max(1, GROUP_GEMINI_RETRY_COUNT)
                ):
                    pause_sec = get_gemini_rate_limit_pause_sec(retry_idx)
                    print(
                        f"[Detect] Gemini rate limit hit for image summary. "
                        f"Pausing {int(pause_sec)}s before retry {retry_idx + 2}/"
                        f"{max(1, GROUP_GEMINI_RETRY_COUNT)}."
                    )
                    time.sleep(pause_sec)
                    continue
                raise RuntimeError(f"Gemini API error: HTTP {e.code}: {detail}") from e
            except Exception as e:
                raise RuntimeError(f"Gemini API request failed: {e}") from e
        if raw is None:
            raise RuntimeError("Gemini API request failed after retries.")

        try:
            data = json.loads(raw)
            candidates = data.get("candidates") or []
            parts_out = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
            raw_text = "\n".join(str(part.get("text", "")).strip() for part in parts_out if part.get("text") is not None).strip()
            finish_reason = str(candidates[0].get("finishReason", "")).strip().upper() if candidates else ""
            meta = build_vlm_response_meta(provider, model, raw_usage=data.get("usageMetadata"), raw_text=raw_text)
        except Exception as e:
            raise RuntimeError(f"Gemini API returned an unexpected response: {raw}") from e

        parsed, cleaned_text = extract_json_from_messy_text(raw_text)
        if parsed is not None:
            ok, reason = validate_group_summary_payload(parsed, expected_group_id=expected_group_id)
            if ok:
                meta["raw_text"] = cleaned_text
                return parsed, meta
            last_raw = raw
            last_raw_text = raw_text
            last_cleaned_text = f"{cleaned_text}\n[invalid_summary:{reason}]"
            continue

        last_raw = raw
        last_raw_text = raw_text
        last_cleaned_text = cleaned_text

        looks_like_partial_json = cleaned_text.lstrip().startswith("{")
        if finish_reason != "MAX_TOKENS" and not looks_like_partial_json:
            break

    dump_path = dump_gemini_failure(prompt, last_raw, last_raw_text, last_cleaned_text)
    raise RuntimeError(
        f"Gemini API returned non-JSON output: {last_raw_text}\n"
        f"Failure dump: {dump_path}"
    )


def run_group_openai_text_summary(prompt):
    if not group_openai_api_key:
        raise RuntimeError("OpenAI group VLM is mandatory, but API key is unavailable.")
    max_token_attempts = [
        int(max(256, GROUP_VLM_MAX_NEW_TOKENS)),
        int(max(512, GROUP_VLM_MAX_NEW_TOKENS * 2)),
        int(max(768, GROUP_VLM_MAX_NEW_TOKENS * 3)),
    ]
    last_error = None
    provider, model = current_vlm_provider_and_model(provider_override="openai")

    for max_tokens in max_token_attempts:
        payload = {
            "model": GROUP_OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GROUP_OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {group_openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw = None
        for retry_idx in range(max(1, GROUP_OPENAI_RETRY_COUNT)):
            try:
                with urllib.request.urlopen(req, timeout=GROUP_OPENAI_TIMEOUT_SEC) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and retry_idx + 1 < max(1, GROUP_OPENAI_RETRY_COUNT):
                    pause_sec = get_openai_rate_limit_pause_sec(retry_idx)
                    print(
                        f"[Detect] OpenAI rate limit hit for text summary. "
                        f"Pausing {int(pause_sec)}s before retry {retry_idx + 2}/"
                        f"{max(1, GROUP_OPENAI_RETRY_COUNT)}."
                    )
                    time.sleep(pause_sec)
                    continue
                raise RuntimeError(f"OpenAI API error: HTTP {e.code}: {detail}") from e
            except Exception as e:
                raise RuntimeError(f"OpenAI API request failed: {e}") from e
        if raw is None:
            raise RuntimeError("OpenAI API request failed after retries.")
        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            raw_text = choice["message"]["content"].strip()
            finish_reason = str(choice.get("finish_reason", "")).strip().lower()
            meta = build_vlm_response_meta(provider, model, raw_usage=data.get("usage"), raw_text=raw_text)
        except Exception as e:
            raise RuntimeError(f"OpenAI API returned an unexpected response: {raw}") from e
        parsed = extract_json_object(raw_text)
        if parsed is not None:
            return parsed, meta
        last_error = raw_text
        if finish_reason != "length":
            break
    raise RuntimeError(f"OpenAI API returned non-JSON output: {last_error}")


@torch.no_grad()
def run_group_vlm_summary(image_paths, prompt):
    if GROUP_VLM_PROVIDER == "openai":
        return run_group_openai_summary(image_paths, prompt)
    if GROUP_VLM_PROVIDER == "gemini":
        return run_group_gemini_summary(image_paths, prompt)
    if GROUP_VLM_PROVIDER == "anthropic":
        return run_group_anthropic_summary(image_paths, prompt)
    if group_vlm_model is None or group_vlm_processor is None:
        raise RuntimeError("Qwen3-VL is mandatory, but the group VLM is not loaded.")
    try:
        pil_images = [Image.open(path).convert("RGB") for path in image_paths if os.path.isfile(path)]
        if not pil_images:
            raise RuntimeError("Qwen3-VL is mandatory, but no evidence images were available for the group.")
        messages = [{
            "role": "user",
            "content": ([{"type": "image"} for _ in pil_images] + [{"type": "text", "text": prompt}]),
        }]
        chat_text = group_vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = group_vlm_processor(
            text=[chat_text],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(dino_device)
        generated_ids = group_vlm_model.generate(
            **inputs,
            max_new_tokens=GROUP_VLM_MAX_NEW_TOKENS,
        )
        trimmed = []
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
            trimmed.append(out_ids[len(in_ids):])
        decoded = group_vlm_processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        raw_text = decoded[0].strip() if decoded else ""
        parsed = extract_json_object(raw_text)
        if parsed is not None:
            provider, model = current_vlm_provider_and_model()
            return parsed, build_vlm_response_meta(provider, model, raw_usage=None, raw_text=raw_text)
        raise RuntimeError(f"Qwen3-VL returned non-JSON output: {raw_text}")
    except Exception as e:
        raise RuntimeError(f"Qwen3-VL summary failed: {e}") from e


def process_kiosk_video(video_path, output_dir, cross_state):
    process_t0 = time.time()
    persistent_gallery = cross_state["persistent_gallery"]
    next_gid = cross_state["next_gid"]

    vname = os.path.splitext(os.path.basename(video_path))[0]
    use_kiosk_zone_logic = vname.endswith("_Kiosk") or KIOSK_ITEM_ONLY_MODE
    fast_qwen_evidence_mode = bool(use_kiosk_zone_logic and FAST_QWEN_EVIDENCE_MODE)
    out_vid = os.path.join(output_dir, f"{vname}_output.mp4")
    out_log = os.path.join(output_dir, f"{vname}_log.txt")
    out_json = os.path.join(output_dir, f"{vname}_events.json")
    out_tracking_json = os.path.join(output_dir, f"{vname}_tracking_summary.json")
    tmp_vid = os.path.join(output_dir, f"{vname}_temp.mp4")

    def vlog(msg):
        msg_str = str(msg)
        prefix = f"[{vname}] "
        log_msg = "\n".join(prefix + line if line else line for line in msg_str.splitlines())
        print(log_msg)
        with open(out_log, "a") as lf:
            lf.write(log_msg + "\n")

    with open(out_log, "w") as lf:
        lf.write("")

    vlog(f"\n{'='*60}\nProcessing kiosk video: {vname}\n{'='*60}")
    vlog(
        f"  Entry gallery available: {len(persistent_gallery)} known ID(s) "
        f"{sorted(int(gid) for gid in persistent_gallery.keys())}"
    )
    fashion_ref_ready = sum(
        1
        for entry in persistent_gallery.values()
        if entry.get("fashion_upper_init") is not None or entry.get("fashion_lower_init") is not None
    )
    vlog(
        f"  FashionCLIP status: model={'ready' if getattr(IntegratedEntry, 'fashionclip_model', None) is not None else 'missing'} "
        f"gallery_refs={fashion_ref_ready}/{len(persistent_gallery)}"
    )
    scanner_ref_emb = None
    scanner_ref_cemb = None
    product_references = []
    product_text_prompt = GROUND_DINO_TEXT_PROMPT
    if fast_qwen_evidence_mode:
        vlog("  Fast Qwen evidence mode enabled: skipping product detection and item tracking.")
    if (not fast_qwen_evidence_mode) and SCANNER_FILTER_ENABLE:
        try:
            if os.path.isfile(SCANNER_REF_PATH):
                scanner_img = cv2.imread(SCANNER_REF_PATH)
                if scanner_img is not None and scanner_img.size > 0:
                    scanner_ref_emb = extract_product_embedding(
                        scanner_img, [0, 0, scanner_img.shape[1], scanner_img.shape[0]]
                    )
                    scanner_ref_cemb = extract_color_hist_embedding(
                        scanner_img, [0, 0, scanner_img.shape[1], scanner_img.shape[0]]
                    )
                    vlog(f"  Scanner filter ready: {SCANNER_REF_PATH}")
                else:
                    vlog(f"  Scanner filter warning: cannot read image {SCANNER_REF_PATH}")
            else:
                vlog(f"  Scanner filter warning: file not found {SCANNER_REF_PATH}")
        except Exception as e:
            vlog(f"  Scanner filter warning: init failed: {e}")
    if (not fast_qwen_evidence_mode) and PRODUCT_REFERENCE_ENABLE:
        product_references = load_product_references(
            PRODUCT_REFERENCE_DIR,
            reference_path=PRODUCT_REFERENCE_PATH,
        )
        if product_references:
            vlog(
                f"  Product references ready: count={len(product_references)} "
                f"dir={PRODUCT_REFERENCE_DIR}"
            )
            for ref in product_references:
                vlog(
                    f"    ref label={ref['label']} "
                    f"path={ref.get('path', '?')}"
                )
        else:
            vlog(
                f"  Product reference warning: cannot load references from "
                f"{PRODUCT_REFERENCE_DIR}"
            )
    if (not fast_qwen_evidence_mode) and GROUND_DINO_ENABLE:
        product_text_prompt = GROUND_DINO_TEXT_PROMPT
        vlog(f"  Grounding DINO prompt: {product_text_prompt}")
    if not fast_qwen_evidence_mode:
        vlog(
            "  PRODUCT thresholds: "
            f"det_conf>={PRODUCT_CONF:.2f}, "
            f"min_track_frames>={PRODUCT_MIN_TRACK_FRAMES}, "
            f"new_item_streak>={ITEM_NEW_MIN_STREAK}, "
            f"new_item_conf>={ITEM_NEW_MIN_CONF:.2f}, "
            f"new_item_far_dist>={ITEM_NEW_FORCE_FAR_DIST}, "
            f"match_iou>={ITEM_MATCH_IOU_TH:.2f}, "
            f"match_dist<={ITEM_MATCH_MAX_DIST}, "
            f"dino_min_sim>={DINO_MIN_SIM_FOR_MATCH:.2f}, "
            f"label_disagree_min_sim>={ITEM_LABEL_DISAGREE_MIN_SIM:.2f}, "
            f"lock_min_sim>={ITEM_LOCK_MIN_SIM:.2f}, "
            f"dino_weight={DINO_MATCH_WEIGHT:.2f}, "
            f"color_weight={COLOR_MATCH_WEIGHT:.2f}, "
            f"color_min_sim>={COLOR_MIN_SIM_FOR_MATCH:.2f}, "
            f"upd_sim>={ITEM_UPDATE_MIN_SIM:.2f}, "
            f"upd_csim>={ITEM_UPDATE_MIN_CSIM:.2f}, "
            f"anchor_sim>={ITEM_ANCHOR_MIN_SIM:.2f}, "
            f"anchor_csim>={ITEM_ANCHOR_MIN_CSIM:.2f}"
        )

    track_memory = {}
    id_map = {}
    lost_gallery = {}
    pending_lost_relinks = {}
    pending_new_trackers = {}
    pending_cross_video_relinks = {}
    events = []
    product_label_history = {}
    product_track_state = {}
    scanned_count_by_gid = {}
    item_type_count_by_gid = {}
    next_item_id_by_gid = {}
    item_tracks_by_gid = {}
    customer_state = {}
    item_trackid_map_by_gid = {}
    pending_new_item_by_gid = {}
    display_item_tracks_by_gid = {}
    hold_state = {}
    holding_display = {}
    group_members_by_gid = {}
    group_evidence_records = {}
    group_evidence_tag_counts = {}
    person_hand_export_state = {}
    group_saved_frames = {}
    group_person_item_state = {}
    person_item_event_counter = {}
    recent_full_frames = []
    fallback_tracks = {}
    fallback_next_id = 1
    model_name_l = os.path.basename(PRODUCT_MODEL_PATH).lower()
    product_track_supported = PRODUCT_USE_TRACK and not ("yoloe" in model_name_l and "seg" in model_name_l)
    product_detect_failed_once = False
    scanned_items_dir = None
    scanned_frames_dir = None
    reference_hits_dir = None
    hold_fails_dir = None
    group_evidence_dir = os.path.join(output_dir, GROUP_EVIDENCE_DIRNAME)
    if GROUP_EVIDENCE_ENABLE:
        os.makedirs(group_evidence_dir, exist_ok=True)
    person_crops_dir = None
    if SAVE_PERSON_CROPS:
        person_crops_dir = os.path.join(output_dir, PERSON_CROPS_DIRNAME)
        os.makedirs(person_crops_dir, exist_ok=True)
        vlog(f"  Person crops output: {person_crops_dir}")

    def resolve_evidence_person_id(primary_box, fallback_person_id=None):
        if primary_box is None or not persistent_gallery:
            return fallback_person_id, -1.0
        cache_key = (
            "evidence_person",
            int(primary_box[0]),
            int(primary_box[1]),
            int(primary_box[2]),
            int(primary_box[3]),
        )
        curr_fu, curr_fl = extract_customer_library_fashion_pair(
            clean,
            primary_box,
            cache_key=cache_key,
        )
        best_gid = None
        best_fc = -1.0
        for gid, entry in persistent_gallery.items():
            ref_u = entry.get("fashion_upper_init")
            ref_l = entry.get("fashion_lower_init")
            sim_fc, _sim_fc_upper, _sim_fc_lower = customer_library_fashion_pair_similarity_breakdown(
                curr_fu,
                curr_fl,
                ref_u,
                ref_l,
            )
            if sim_fc > best_fc:
                best_fc = sim_fc
                best_gid = int(gid)
        if best_gid is None:
            return fallback_person_id, best_fc
        return best_gid, best_fc

    def record_group_evidence(group_id, tag, frame_idx, note, primary_box=None, secondary_box=None, **extra_fields):
        if not GROUP_EVIDENCE_ENABLE:
            return None
        limits = {
            "exit_view": 3,
            "person_exit": 6,
            "pre_exit_carry": 999999,
            "hand_hold": 999999,
            "hand_hold_weak": 999999,
            "body_carry": 999999,
            "bag_pocket": 999999,
            "item_held": 4,
            "hidden": 2,
            "carried_out": 2,
        }
        gid = int(group_id)
        frame_idx = int(frame_idx)
        resolved_kiosk = extra_fields.get("kiosk")
        if resolved_kiosk is None:
            resolved_kiosk = detect_kiosk_label(person_box=primary_box, item_box=secondary_box)
        saved_frames = group_saved_frames.setdefault(gid, set())
        allow_same_frame = str(tag) == "pre_exit_carry"
        if frame_idx in saved_frames and not allow_same_frame:
            return None
        tag_key = (gid, str(tag))
        count = int(group_evidence_tag_counts.get(tag_key, 0))
        if tag in ("hand_hold", "hand_hold_weak", "bag_pocket", "pre_exit_carry") and FAST_HAND_EVENT_SAVE_ALL:
            count = 0
        if count >= int(limits.get(tag, 2)):
            return None
        suffix = ""
        if str(tag) == "pre_exit_carry":
            pre_exit_offset = extra_fields.get("pre_exit_offset")
            if pre_exit_offset is not None:
                suffix = f"_offset_{int(pre_exit_offset)}"
        resolved_person_id, resolved_person_score = resolve_evidence_person_id(
            primary_box,
            fallback_person_id=extra_fields.get("person_id"),
        )
        path = save_group_evidence_frame(
            clean,
            group_evidence_dir,
            gid,
            frame_idx,
            str(tag),
            str(note),
            primary_box=primary_box,
            secondary_box=secondary_box,
            person_id=resolved_person_id,
            kiosk_label=resolved_kiosk,
            suffix=suffix,
        )
        group_evidence_tag_counts[tag_key] = count + 1
        if not allow_same_frame:
            saved_frames.add(frame_idx)
        group_evidence_records.setdefault(gid, []).append({
            "frame": frame_idx,
            "tag": str(tag),
            "note": str(note),
            "image": path,
            "person_id": make_json_safe(resolved_person_id),
            "person_match_score": float(resolved_person_score),
            "kiosk": make_json_safe(resolved_kiosk),
            **{str(k): make_json_safe(v) for k, v in extra_fields.items()},
        })
        return path

    def record_group_evidence_from_frame(frame_bgr, group_id, tag, frame_idx, note, primary_box=None, secondary_box=None, **extra_fields):
        if not GROUP_EVIDENCE_ENABLE or frame_bgr is None:
            return None
        limits = {
            "exit_view": 3,
            "person_exit": 6,
            "pre_exit_carry": 999999,
            "hand_hold": 999999,
            "hand_hold_weak": 999999,
            "body_carry": 999999,
            "bag_pocket": 999999,
            "item_held": 4,
            "hidden": 2,
            "carried_out": 2,
        }
        gid = int(group_id)
        frame_idx = int(frame_idx)
        resolved_kiosk = extra_fields.get("kiosk")
        if resolved_kiosk is None:
            resolved_kiosk = detect_kiosk_label(person_box=primary_box, item_box=secondary_box)
        saved_frames = group_saved_frames.setdefault(gid, set())
        allow_same_frame = str(tag) == "pre_exit_carry"
        if frame_idx in saved_frames and not allow_same_frame:
            return None
        tag_key = (gid, str(tag))
        count = int(group_evidence_tag_counts.get(tag_key, 0))
        if tag in ("hand_hold", "hand_hold_weak", "bag_pocket", "pre_exit_carry") and FAST_HAND_EVENT_SAVE_ALL:
            count = 0
        if count >= int(limits.get(tag, 2)):
            return None
        suffix = ""
        if str(tag) == "pre_exit_carry":
            pre_exit_offset = extra_fields.get("pre_exit_offset")
            if pre_exit_offset is not None:
                suffix = f"_offset_{int(pre_exit_offset)}"
        resolved_person_id, resolved_person_score = resolve_evidence_person_id(
            primary_box,
            fallback_person_id=extra_fields.get("person_id"),
        )
        path = save_group_evidence_frame(
            frame_bgr,
            group_evidence_dir,
            gid,
            frame_idx,
            str(tag),
            str(note),
            primary_box=primary_box,
            secondary_box=secondary_box,
            person_id=resolved_person_id,
            kiosk_label=resolved_kiosk,
            suffix=suffix,
        )
        group_evidence_tag_counts[tag_key] = count + 1
        if not allow_same_frame:
            saved_frames.add(frame_idx)
        group_evidence_records.setdefault(gid, []).append({
            "frame": frame_idx,
            "tag": str(tag),
            "note": str(note),
            "image": path,
            "person_id": make_json_safe(resolved_person_id),
            "person_match_score": float(resolved_person_score),
            "kiosk": make_json_safe(resolved_kiosk),
            **{str(k): make_json_safe(v) for k, v in extra_fields.items()},
        })
        return path

    def match_persistent_gallery(emb, box, cx, cy, tracker_id, claimed_gids, frame_idx, sim_floor=None):
        if not CROSS_VIDEO_REID or emb is None or not persistent_gallery:
            return None
        vis_ratio = get_visible_ratio_in_frame(box, new_w, new_h)
        if vis_ratio < CROSS_VIDEO_MIN_VISIBLE_RATIO:
            return None

        curr_fu, curr_fl = extract_customer_library_fashion_pair(
            clean, box, cache_key=int(tracker_id)
        )
        candidates = []
        for gid, entry in persistent_gallery.items():
            if gid in claimed_gids or gid in track_memory:
                continue
            views = entry.get("views") or []
            if not views:
                continue
            ref_u = entry.get("fashion_upper_init")
            ref_l = entry.get("fashion_lower_init")
            sim_os = avg_sim_against_views_tensor(emb, views)
            sim_fc, sim_fc_upper, sim_fc_lower = customer_library_fashion_pair_similarity_breakdown(
                curr_fu,
                curr_fl,
                ref_u,
                ref_l,
            )
            curr_u_flag, curr_l_flag, ref_u_flag, ref_l_flag = fashion_pair_presence_flags(
                curr_fu, curr_fl, ref_u, ref_l
            )
            sim = sim_os
            if sim_fc >= 0.0:
                a = customer_library_fashion_cross_alpha()
                sim = (1.0 - a) * sim_os + a * sim_fc
            candidates.append((
                sim, int(gid), sim_os, sim_fc, sim_fc_upper, sim_fc_lower,
                curr_u_flag, curr_l_flag, ref_u_flag, ref_l_flag
            ))
        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda x: x[0])
        (
            best_sim, best_gid, best_os, best_fc, best_fc_upper, best_fc_lower,
            best_curr_u_flag, best_curr_l_flag, best_ref_u_flag, best_ref_l_flag
        ) = candidates[0]
        best_os_entry = max(candidates, key=lambda x: x[2]) if candidates else None
        best_fc_entry = None
        fc_candidates = [cand for cand in candidates if cand[3] >= 0.0]
        if fc_candidates:
            best_fc_entry = max(fc_candidates, key=lambda x: x[3])
        best_os_gid = int(best_os_entry[1]) if best_os_entry is not None else None
        best_fc_gid = int(best_fc_entry[1]) if best_fc_entry is not None else None
        contested_modalities = (
            best_fc_gid is not None
            and best_os_gid is not None
            and best_os_gid != best_fc_gid
        )
        required_streak = max(
            1,
            KIOSK_CROSS_VIDEO_CONTESTED_MIN_STREAK
            if contested_modalities
            else KIOSK_CROSS_VIDEO_ASSIGN_MIN_STREAK,
        )
        effective_ambiguity_gap = max(
            KIOSK_CROSS_VIDEO_AMBIGUITY_GAP,
            KIOSK_CROSS_VIDEO_CONTESTED_GAP if contested_modalities else KIOSK_CROSS_VIDEO_AMBIGUITY_GAP,
        )
        floor = CROSS_VIDEO_REID_SIM_THRESHOLD if sim_floor is None else sim_floor
        top_candidates = ", ".join(
            f"gid{gid}=fused:{sim:.3f}/os:{sim_os:.3f}/fc:{sim_fc:.3f}/u:{sim_u:.3f}/l:{sim_l:.3f}/"
            f"curr({cu}{cl})/ref({ru}{rl})"
            for sim, gid, sim_os, sim_fc, sim_u, sim_l, cu, cl, ru, rl in candidates[:5]
        )
        if best_sim < floor:
            pending_cross_video_relinks.pop(tracker_id, None)
            slot = pending_new_trackers.setdefault(tracker_id, {"embs": [], "ambiguous_hold": False})
            slot["ambiguous_hold"] = False
            vlog(
                f"  [F{frame_idx}] Cross-video reject: tracker {tracker_id} "
                f"best=gid{best_gid}@{best_sim:.3f} os={best_os:.3f} fc={best_fc:.3f} "
                f"fc_upper={best_fc_upper:.3f} fc_lower={best_fc_lower:.3f} "
                f"curr_u={best_curr_u_flag} curr_l={best_curr_l_flag} "
                f"ref_u={best_ref_u_flag} ref_l={best_ref_l_flag} below_th={floor:.3f} "
                f"vis={vis_ratio*100:.1f}% candidates=[{top_candidates}]"
            )
            return None

        if (not CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1) and len(candidates) > 1:
            (
                runner_up, runner_gid, runner_os, runner_fc, runner_fc_upper, runner_fc_lower,
                runner_curr_u_flag, runner_curr_l_flag, runner_ref_u_flag, runner_ref_l_flag
            ) = candidates[1]
            if best_sim - runner_up < effective_ambiguity_gap:
                slot = pending_new_trackers.setdefault(tracker_id, {"embs": [], "ambiguous_hold": False})
                if CROSS_VIDEO_DELAY_ON_AMBIGUOUS:
                    slot["ambiguous_hold"] = True
                slot["ambiguous_since"] = int(slot.get("ambiguous_since", frame_idx))
                cv_slot = pending_cross_video_relinks.get(tracker_id)
                if (
                    cv_slot
                    and cv_slot.get("gid") == best_gid
                    and frame_idx - cv_slot.get("last_frame", -1) <= max(1, FRAME_SKIP)
                ):
                    cv_slot["ambiguous_streak"] = int(cv_slot.get("ambiguous_streak", 0)) + 1
                else:
                    cv_slot = {
                        "gid": best_gid,
                        "streak": 0,
                        "ambiguous_streak": 1,
                    }
                cv_slot["last_frame"] = int(frame_idx)
                cv_slot["last_sim"] = float(best_sim)
                cv_slot["last_gap"] = float(best_sim - runner_up)
                pending_cross_video_relinks[tracker_id] = cv_slot
                vlog(
                    f"  [F{frame_idx}] Cross-video ambiguous: tracker {tracker_id} "
                    f"best=gid{best_gid}@{best_sim:.3f}(os={best_os:.3f},fc={best_fc:.3f},u={best_fc_upper:.3f},l={best_fc_lower:.3f},curr={best_curr_u_flag}{best_curr_l_flag},ref={best_ref_u_flag}{best_ref_l_flag}) "
                    f"runner_up=gid{runner_gid}@{runner_up:.3f}(os={runner_os:.3f},fc={runner_fc:.3f},u={runner_fc_upper:.3f},l={runner_fc_lower:.3f},curr={runner_curr_u_flag}{runner_curr_l_flag},ref={runner_ref_u_flag}{runner_ref_l_flag}) "
                    f"gap={best_sim - runner_up:.3f} gap_th={effective_ambiguity_gap:.3f} "
                    f"contested={'Y' if contested_modalities else 'N'} "
                    f"os_winner={best_os_gid} fc_winner={best_fc_gid} vis={vis_ratio*100:.1f}% "
                    f"ambiguous_streak={cv_slot.get('ambiguous_streak', 0)}/{KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_STREAK} "
                    f"candidates=[{top_candidates}]"
                )
                if (
                    CROSS_VIDEO_DELAY_ON_AMBIGUOUS
                    and int(cv_slot.get("ambiguous_streak", 0)) >= max(1, KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_STREAK)
                    and best_sim >= KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_MIN_SIM
                ):
                    seeded_views = (persistent_gallery[best_gid].get("views") or []).copy()
                    track_memory[best_gid] = {
                        "last_cx": int(cx),
                        "last_cy": int(cy),
                        "last_box": list(map(int, box)),
                        "missing": 0,
                        "seen": 1,
                        "views": seeded_views,
                        "fashion_upper_init": persistent_gallery[best_gid].get("fashion_upper_init"),
                        "fashion_lower_init": persistent_gallery[best_gid].get("fashion_lower_init"),
                        "source": "entry_gallery",
                        "assigned_frame": int(frame_idx),
                    }
                    id_map[tracker_id] = best_gid
                    pending_new_trackers.pop(tracker_id, None)
                    pending_cross_video_relinks.pop(tracker_id, None)
                    claimed_gids.add(best_gid)
                    vlog(
                        f"  [F{frame_idx}] Cross-video ambiguous-resolve: tracker {tracker_id} -> gid {best_gid} "
                        f"(fused={best_sim:.3f}, os={best_os:.3f}, fc={best_fc:.3f}, "
                        f"gap={best_sim - runner_up:.3f}, ambiguous_streak={cv_slot.get('ambiguous_streak', 0)}, "
                        f"min_sim={KIOSK_CROSS_VIDEO_AMBIGUOUS_RESOLVE_MIN_SIM:.3f})"
                    )
                    return best_gid
                return None

        slot = pending_new_trackers.setdefault(tracker_id, {"embs": [], "ambiguous_hold": False})
        slot["ambiguous_hold"] = False
        slot.pop("ambiguous_since", None)

        cv_slot = pending_cross_video_relinks.get(tracker_id)
        if cv_slot and cv_slot.get("gid") == best_gid and frame_idx - cv_slot.get("last_frame", -1) <= max(1, FRAME_SKIP):
            cv_slot["streak"] += 1
        else:
            cv_slot = {"gid": best_gid, "streak": 1}
        cv_slot["last_frame"] = int(frame_idx)
        cv_slot["last_sim"] = float(best_sim)
        pending_cross_video_relinks[tracker_id] = cv_slot

        if cv_slot["streak"] < required_streak:
            return None

        seeded_views = (persistent_gallery[best_gid].get("views") or []).copy()
        track_memory[best_gid] = {
            "last_cx": int(cx),
            "last_cy": int(cy),
            "last_box": list(map(int, box)),
            "missing": 0,
            "seen": 1,
            "views": seeded_views,
            "fashion_upper_init": persistent_gallery[best_gid].get("fashion_upper_init"),
            "fashion_lower_init": persistent_gallery[best_gid].get("fashion_lower_init"),
            "source": "entry_gallery",
            "assigned_frame": int(frame_idx),
        }
        id_map[tracker_id] = best_gid
        pending_cross_video_relinks.pop(tracker_id, None)
        pending_new_trackers.pop(tracker_id, None)
        claimed_gids.add(best_gid)
        vlog(
            f"  [F{frame_idx}] Cross-video match: tracker {tracker_id} -> gid {best_gid} "
            f"(fused={best_sim:.3f}, os={best_os:.3f}, fc={best_fc:.3f}, "
            f"fc_upper={best_fc_upper:.3f}, fc_lower={best_fc_lower:.3f}, "
            f"curr_u={best_curr_u_flag} curr_l={best_curr_l_flag} "
            f"ref_u={best_ref_u_flag} ref_l={best_ref_l_flag}, vis={vis_ratio*100:.1f}%, "
            f"streak={cv_slot['streak']}/{required_streak}, "
            f"contested={'Y' if contested_modalities else 'N'} "
            f"os_winner={best_os_gid} fc_winner={best_fc_gid}, "
            f"candidates=[{top_candidates}])"
        )
        return best_gid

    def assign_or_relink_person(frame_bgr, box, cx, cy, tracker_id, claimed_gids, frame_idx):
        nonlocal next_gid
        if not persistent_gallery:
            vlog(
                f"  [F{frame_idx}] KIOSK skip tracker={tracker_id} "
                f"reason=entry_persistent_gallery_empty"
            )
            return None
        if person_reid_model is None:
            vlog(
                f"  [F{frame_idx}] KIOSK skip tracker={tracker_id} "
                f"reason=customer_reid_unavailable"
            )
            return None

        if tracker_id in id_map:
            gid = id_map[tracker_id]
            if gid in claimed_gids:
                del id_map[tracker_id]
            elif gid in track_memory:
                mem = track_memory[gid]
                emb = extract_customer_library_embedding(frame_bgr, box, cache_key=tracker_id)
                curr_fu, curr_fl = extract_customer_library_fashion_pair(
                    frame_bgr, box, cache_key=tracker_id
                )
                ref_u = mem.get("fashion_upper_init")
                ref_l = mem.get("fashion_lower_init")
                sim_os = avg_sim_against_views_tensor(emb, mem.get("views", []))
                sim_fc, sim_fc_upper, sim_fc_lower = customer_library_fashion_pair_similarity_breakdown(
                    curr_fu,
                    curr_fl,
                    ref_u,
                    ref_l,
                )
                curr_u_flag, curr_l_flag, ref_u_flag, ref_l_flag = fashion_pair_presence_flags(
                    curr_fu, curr_fl, ref_u, ref_l
                )
                sim = sim_os
                if sim_fc >= 0.0:
                    a = customer_library_fashion_within_alpha()
                    sim = (1.0 - a) * sim_os + a * sim_fc
                if int(mem.get("missing", 0)) > 0:
                    if sim >= REID_SIM_THRESHOLD:
                        if emb is not None:
                            mem["views"] = ema_update_views(mem.get("views", []), emb)
                        pending_lost_relinks.pop(tracker_id, None)
                        claimed_gids.add(gid)
                        return gid
                    del id_map[tracker_id]
                else:
                    if sim < REID_SIM_THRESHOLD:
                        sticky_floor = max(0.0, REID_SIM_THRESHOLD - KIOSK_STICKY_REID_MARGIN)
                        if sim >= sticky_floor:
                            claimed_gids.add(gid)
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_idx}] KIOSK keep sticky tracker map tracker={tracker_id} "
                                    f"gid={gid} fused={sim:.3f} os={sim_os:.3f} fc={sim_fc:.3f} "
                                    f"fc_upper={sim_fc_upper:.3f} fc_lower={sim_fc_lower:.3f} "
                                    f"curr_u={curr_u_flag} curr_l={curr_l_flag} "
                                    f"ref_u={ref_u_flag} ref_l={ref_l_flag} "
                                    f"sticky_floor={sticky_floor:.3f}"
                            )
                            return gid
                        vlog(
                            f"  [F{frame_idx}] KIOSK drop stale tracker map tracker={tracker_id} "
                            f"gid={gid} fused={sim:.3f} os={sim_os:.3f} fc={sim_fc:.3f} "
                            f"fc_upper={sim_fc_upper:.3f} fc_lower={sim_fc_lower:.3f} "
                            f"curr_u={curr_u_flag} curr_l={curr_l_flag} "
                            f"ref_u={ref_u_flag} ref_l={ref_l_flag} th={REID_SIM_THRESHOLD:.3f}"
                        )
                        del id_map[tracker_id]
                        pending_lost_relinks.pop(tracker_id, None)
                    else:
                        if emb is not None:
                            mem["views"] = ema_update_views(mem.get("views", []), emb)
                        claimed_gids.add(gid)
                        return gid
            else:
                del id_map[tracker_id]

        emb = extract_customer_library_embedding(frame_bgr, box, cache_key=tracker_id)
        curr_fu, curr_fl = extract_customer_library_fashion_pair(
            frame_bgr, box, cache_key=tracker_id
        )
        best_gid = None
        best_score = REID_SIM_THRESHOLD
        if emb is not None:
            for gid, lost in lost_gallery.items():
                if gid in claimed_gids:
                    continue
                dist = np.linalg.norm(np.array(lost["center"]) - np.array([cx, cy]))
                if dist > SPATIAL_GATE:
                    continue
                sim_os = avg_sim_against_views_tensor(emb, lost.get("views", []))
                snap = lost.get("snapshot", {})
                sim_fc, sim_fc_upper, sim_fc_lower = customer_library_fashion_pair_similarity_breakdown(
                    curr_fu,
                    curr_fl,
                    snap.get("fashion_upper_init"),
                    snap.get("fashion_lower_init"),
                )
                sim = sim_os
                if sim_fc >= 0.0:
                    a = customer_library_fashion_within_alpha()
                    sim = (1.0 - a) * sim_os + a * sim_fc
                if sim > best_score:
                    best_score = sim
                    best_gid = gid

        if best_gid is not None:
            slot = pending_lost_relinks.get(tracker_id)
            if slot and slot.get("gid") == best_gid and frame_idx - slot.get("last_frame", -1) <= max(1, FRAME_SKIP):
                slot["streak"] += 1
            else:
                slot = {"gid": best_gid, "streak": 1}
            slot["last_frame"] = int(frame_idx)
            pending_lost_relinks[tracker_id] = slot
            if slot["streak"] >= max(1, KIOSK_CROSS_VIDEO_ASSIGN_MIN_STREAK):
                id_map[tracker_id] = best_gid
                lost_gallery[best_gid]["views"] = ema_update_views(lost_gallery[best_gid].get("views", []), emb)
                restored = lost_gallery[best_gid].get("snapshot", {}).copy()
                restored["missing"] = 0
                restored["views"] = lost_gallery[best_gid].get("views", [])
                restored["last_cx"] = int(cx)
                restored["last_cy"] = int(cy)
                restored["last_box"] = list(map(int, box))
                restored["assigned_frame"] = int(frame_idx)
                track_memory[best_gid] = restored
                del lost_gallery[best_gid]
                claimed_gids.add(best_gid)
                return best_gid
            return None
        pending_lost_relinks.pop(tracker_id, None)
        gid = match_persistent_gallery(emb, box, cx, cy, tracker_id, claimed_gids, frame_idx)
        if gid is not None:
            return gid
        cv_slot = pending_cross_video_relinks.get(tracker_id)
        if cv_slot and cv_slot.get("streak", 0) < max(1, KIOSK_CROSS_VIDEO_ASSIGN_MIN_STREAK):
            return None
        slot = pending_new_trackers.get(tracker_id)
        if slot and slot.get("ambiguous_hold", False):
            return None

        avg_emb, agg_count = get_pending_avg_embedding(pending_new_trackers, tracker_id, emb)
        if avg_emb is not None and agg_count >= max(1, NEW_ID_AGG_FRAMES):
            gid = match_persistent_gallery(
                avg_emb,
                box,
                cx,
                cy,
                tracker_id,
                claimed_gids,
                frame_idx,
                sim_floor=NEW_ID_PERSIST_SIM_THRESHOLD,
            )
            if gid is not None:
                return gid
            cv_slot = pending_cross_video_relinks.get(tracker_id)
            if cv_slot and cv_slot.get("streak", 0) < max(1, KIOSK_CROSS_VIDEO_ASSIGN_MIN_STREAK):
                return None
            slot = pending_new_trackers.get(tracker_id)
            if slot and slot.get("ambiguous_hold", False):
                return None
        elif emb is not None:
            return None

        slot = pending_new_trackers.get(tracker_id)
        if persistent_gallery and slot is not None:
            sample_count = int(slot.get("samples", 0))
            if sample_count < max(1, KIOSK_ENTRY_MATCH_WAIT_FRAMES):
                return None

        pending_new_trackers.pop(tracker_id, None)
        pending_cross_video_relinks.pop(tracker_id, None)
        vlog(
            f"  [F{frame_idx}] KIOSK skip tracker={tracker_id} "
            f"reason=no_entry_library_match"
        )
        return None

    kiosk1_raw = parse_box(KIOSK_1_BOX_STR)
    kiosk2_raw = parse_box(KIOSK_2_BOX_STR)
    exit_raw = parse_box(EXIT_BOX_STR)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        vlog(f"Could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    new_w = RESIZE_WIDTH
    new_h = int(h * RESIZE_WIDTH / w)

    kiosk1 = expand_box(kiosk1_raw, INTERACT_ZONE_EXPAND, new_w, new_h)
    kiosk2 = expand_box(kiosk2_raw, INTERACT_ZONE_EXPAND, new_w, new_h)
    exit_zone = expand_box(exit_raw, INTERACT_ZONE_EXPAND, new_w, new_h)
    kiosk_union = union_boxes([kiosk1, kiosk2])

    def detect_kiosk_label(person_box=None, item_box=None):
        if person_box is not None:
            if box_center_in_box(person_box, kiosk1):
                return "KIOSK-1"
            if box_center_in_box(person_box, kiosk2):
                return "KIOSK-2"
        if item_box is not None:
            if box_center_in_box(item_box, kiosk1):
                return "KIOSK-1"
            if box_center_in_box(item_box, kiosk2):
                return "KIOSK-2"
        return None

    writer = cv2.VideoWriter(
        tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / FRAME_SKIP), (new_w, new_h)
    )

    frame_count = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        frame = cv2.resize(frame, (new_w, new_h))
        clean = frame.copy()
        recent_full_frames.append({
            "frame_idx": int(frame_count),
            "frame": clean.copy(),
        })
        if len(recent_full_frames) > 64:
            recent_full_frames = recent_full_frames[-64:]

        if frame_count % 200 == 0:
            vlog(
                f"  [{frame_count/max(total,1)*100:5.1f}%] frame {frame_count}/{total} "
                f"events={len(events)} elapsed={time.time()-t0:.0f}s"
            )

        if use_kiosk_zone_logic:
            # Draw kiosk zones only for 3_Entrance and 4_Entrance.
            cv2.rectangle(frame, (kiosk1[0], kiosk1[1]), (kiosk1[2], kiosk1[3]), (255, 80, 80), 2)
            cv2.putText(frame, "KIOSK-1", (kiosk1[0] + 6, kiosk1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 80), 2)
            cv2.rectangle(frame, (kiosk2[0], kiosk2[1]), (kiosk2[2], kiosk2[3]), (80, 200, 255), 2)
            cv2.putText(frame, "KIOSK-2", (kiosk2[0] + 6, kiosk2[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 2)
            cv2.rectangle(frame, (exit_zone[0], exit_zone[1]), (exit_zone[2], exit_zone[3]), (220, 180, 80), 2)
            cv2.putText(frame, "EXIT", (exit_zone[0] + 6, exit_zone[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 180, 80), 2)

        results = person_model.track(
            frame, persist=True, tracker=TRACKER_PATH, classes=[0], conf=0.3, iou=0.5
        )
        raw_person_count = 0
        id_ready_count = 0
        pass_h_count = 0
        pass_area_count = 0
        pass_vis_count = 0
        accepted_count = 0
        valid = []
        pre_person_hands_map = {}
        hand_rois = []

        if results and results[0].boxes is not None:
            raw_person_count = int(len(results[0].boxes))

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            raw_boxes = results[0].boxes.xyxy.cpu().numpy()
            raw_ids = None
            if results[0].boxes.id is not None:
                raw_ids = results[0].boxes.id.cpu().numpy().astype(int)
                id_ready_count = int(len(raw_ids))
            else:
                assigned = []
                used_prev = set()
                for box in raw_boxes:
                    cx = int((box[0] + box[2]) * 0.5)
                    cy = int((box[1] + box[3]) * 0.5)
                    best_k = None
                    best_d = 1e9
                    for fid, fmem in fallback_tracks.items():
                        if fid in used_prev:
                            continue
                        if int(fmem.get("missing", 0)) > KIOSK_FALLBACK_ID_MAX_MISSING:
                            continue
                        fx, fy = int(fmem["cx"]), int(fmem["cy"])
                        d = float(((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5)
                        if d < best_d:
                            best_d = d
                            best_k = int(fid)
                    if best_k is not None and best_d <= KIOSK_FALLBACK_ID_MAX_DIST:
                        fid = int(best_k)
                        used_prev.add(fid)
                    else:
                        fid = int(fallback_next_id)
                        fallback_next_id += 1
                    assigned.append(fid)
                raw_ids = np.asarray(assigned, dtype=int)
                id_ready_count = int(len(raw_ids))

            for box, tid in zip(raw_boxes, raw_ids):
                h_box = box[3] - box[1]
                a_box = (box[2] - box[0]) * h_box
                if not KIOSK_DISABLE_PERSON_SIZE_FILTER:
                    if h_box < KIOSK_PERSON_MIN_BOX_HEIGHT:
                        continue
                    pass_h_count += 1
                    if a_box < KIOSK_PERSON_MIN_BOX_AREA:
                        continue
                    pass_area_count += 1
                else:
                    pass_h_count += 1
                    pass_area_count += 1
                vis = get_visible_ratio_in_frame(box, new_w, new_h)
                if vis < KIOSK_PERSON_MIN_VISIBLE_RATIO:
                    continue
                pass_vis_count += 1
                valid.append((box, tid))
                wrists = []
                if (not fast_qwen_evidence_mode) or FAST_HAND_EVENT_EXPORT_ENABLE:
                    wrists = get_person_hand_points(clean, list(map(int, box)))
                pre_person_hands_map[int(tid)] = wrists
                if wrists:
                    xs = [w[0] for w in wrists]
                    ys = [w[1] for w in wrists]
                    hand_rois.append([
                        max(0, min(xs) - HAND_ROI_PAD),
                        max(0, min(ys) - HAND_ROI_PAD),
                        min(new_w, max(xs) + HAND_ROI_PAD),
                        min(new_h, max(ys) + HAND_ROI_PAD),
                    ])
            accepted_count = int(len(valid))
            hand_rois = merge_rois(
                [roi for roi in hand_rois if roi[2] > roi[0] and roi[3] > roi[1]]
            )

        accepted_person_present = bool(valid)
        hand_activity_present = bool(hand_rois) if PRODUCT_REQUIRE_HAND_ACTIVITY else accepted_person_present

        stable_products = []
        reference_hit_boxes = []
        phone_boxes = []
        # Product rectangles for kiosk videos.
        if (
            (not fast_qwen_evidence_mode)
            and
            use_kiosk_zone_logic
            and PRODUCT_DETECT_ENABLE
            and accepted_person_present
            and hand_activity_present
        ):
            try:
                detections = []

                def remap_detections(local_dets, roi_box):
                    rx1, ry1, _, _ = roi_box
                    remapped = []
                    for det in local_dets:
                        bx1, by1, bx2, by2 = det["box"]
                        remapped.append({
                            "box": [int(bx1 + rx1), int(by1 + ry1), int(bx2 + rx1), int(by2 + ry1)],
                            "cls_name": det["cls_name"],
                            "conf": float(det["conf"]),
                            "track_id": int(det.get("track_id", -1)),
                        })
                    return remapped

                for roi in hand_rois:
                    rx1, ry1, rx2, ry2 = roi
                    roi_frame = clean[ry1:ry2, rx1:rx2]
                    if roi_frame.size == 0:
                        continue
                    roi_dets = []
                    if GROUND_DINO_ENABLE and gdino_model is not None and gdino_processor is not None:
                        gdino_raw = detect_products_ground_dino(roi_frame, product_text_prompt, PRODUCT_CONF)
                        for det in gdino_raw:
                            x1, y1, x2, y2 = map(int, det["box"])
                            size_ok, bw, bh, barea = product_box_size_ok([x1, y1, x2, y2])
                            if not size_ok:
                                if PRODUCT_DEBUG_LOG:
                                    vlog(
                                        f"    label_reject raw_box={[x1, y1, x2, y2]} "
                                        f"reason=small_box w={bw} h={bh} area={barea}"
                                    )
                                continue
                            keep_ok, keep_reason = product_det_keepable(det)
                            if not keep_ok:
                                if PRODUCT_DEBUG_LOG:
                                    vlog(
                                        f"    label_reject raw_box={[x1, y1, x2, y2]} "
                                        f"reason={keep_reason}"
                                    )
                                continue
                            roi_dets.append(det)
                        if PRODUCT_DEBUG_LOG:
                            vlog(
                                f"  [F{frame_count}] PRODUCT source=grounding_dino "
                                f"roi={roi} count={len(roi_dets)}"
                            )
                    if not roi_dets:
                        # YOLO path: use plain per-frame detection with tracker fallback.
                        if product_model is None:
                            raise RuntimeError(
                                "Product model is not loaded. Set KIOSK_PRODUCT_MODEL_PATH to a valid YOLO .pt"
                            )
                        detect_kwargs = {
                            "conf": PRODUCT_CONF,
                            "iou": 0.5,
                            "verbose": False,
                        }
                        prod_res = None
                        if product_track_supported:
                            try:
                                prod_res = product_model.track(
                                    roi_frame,
                                    persist=True,
                                    tracker=TRACKER_PATH,
                                    **detect_kwargs,
                                )
                            except Exception as e_track:
                                msg = str(e_track)
                                if PRODUCT_DEBUG_LOG:
                                    vlog(f"  [F{frame_count}] Product track warning: {msg}")
                                if "shape '[1, -1, 6, 42]' is invalid" in msg:
                                    product_track_supported = False
                                    if PRODUCT_DEBUG_LOG:
                                        vlog(f"  [F{frame_count}] Product track disabled for this video (YOLOE shape issue)")
                        if prod_res is None:
                            try:
                                prod_res = product_model(roi_frame, **detect_kwargs)
                            except Exception as e_det:
                                if PRODUCT_DEBUG_LOG and not product_detect_failed_once:
                                    vlog(f"  [F{frame_count}] Product detect fallback warning: {e_det}")
                                    product_detect_failed_once = True
                                prod_res = []
                        if prod_res and prod_res[0].boxes is not None and len(prod_res[0].boxes) > 0:
                            pxy = prod_res[0].boxes.xyxy.cpu().numpy()
                            pcl = prod_res[0].boxes.cls.cpu().numpy().astype(int)
                            pcf = prod_res[0].boxes.conf.cpu().numpy()
                            if prod_res[0].boxes.id is not None:
                                pids = prod_res[0].boxes.id.cpu().numpy().astype(int)
                            else:
                                pids = np.full((len(pxy),), -1, dtype=int)
                            names_map = product_model.names if hasattr(product_model, "names") else {}
                            for pb, cid, cscore, ptid in zip(pxy, pcl, pcf, pids):
                                x1, y1, x2, y2 = map(int, pb)
                                size_ok, bw, bh, barea = product_box_size_ok([x1, y1, x2, y2])
                                if not size_ok:
                                    if PRODUCT_DEBUG_LOG:
                                        vlog(
                                            f"    label_reject raw_box={[x1, y1, x2, y2]} "
                                            f"reason=small_box w={bw} h={bh} area={barea}"
                                        )
                                    continue
                                name = names_map.get(int(cid), "product")
                                lname = str(name).lower()
                                has_include = any(k in lname for k in PRODUCT_INCLUDE_LABEL_KEYWORDS)
                                if PRODUCT_DEBUG_LOG:
                                    vlog(f"    label_check raw='{name}' include={has_include}")
                                if not has_include:
                                    if PRODUCT_DEBUG_LOG:
                                        vlog(f"    label_reject raw='{name}' reason=no_include_keyword")
                                    continue
                                roi_dets.append({
                                    "box": [x1, y1, x2, y2],
                                    "cls_name": str(name),
                                    "conf": float(cscore),
                                    "track_id": int(ptid),
                                })
                    if (not roi_dets) and PRODUCT_FALLBACK_TO_PERSON_MODEL:
                        try:
                            fb = person_model(roi_frame, classes=PRODUCT_CLASS_IDS, conf=PRODUCT_CONF, iou=0.5, verbose=False)
                            if fb and fb[0].boxes is not None and len(fb[0].boxes) > 0:
                                fxy = fb[0].boxes.xyxy.cpu().numpy()
                                fcl = fb[0].boxes.cls.cpu().numpy().astype(int)
                                fcf = fb[0].boxes.conf.cpu().numpy()
                                names_map = person_model.names if hasattr(person_model, "names") else {}
                                for pb, cid, cscore in zip(fxy, fcl, fcf):
                                    x1, y1, x2, y2 = map(int, pb)
                                    size_ok, bw, bh, barea = product_box_size_ok([x1, y1, x2, y2])
                                    if not size_ok:
                                        if PRODUCT_DEBUG_LOG:
                                            vlog(
                                                f"    label_reject raw_box={[x1, y1, x2, y2]} "
                                                f"reason=small_box w={bw} h={bh} area={barea}"
                                            )
                                        continue
                                    raw_name = str(names_map.get(int(cid), "product"))
                                    lname = raw_name.lower()
                                    has_include = any(k in lname for k in PRODUCT_INCLUDE_LABEL_KEYWORDS)
                                    if PRODUCT_DEBUG_LOG:
                                        vlog(f"    label_check raw='{raw_name}' include={has_include}")
                                    if not has_include:
                                        if PRODUCT_DEBUG_LOG:
                                            vlog(f"    label_reject raw='{raw_name}' reason=no_include_keyword")
                                        continue
                                    roi_dets.append({
                                        "box": [x1, y1, x2, y2],
                                        "cls_name": raw_name,
                                        "conf": float(cscore),
                                        "track_id": -1,
                                    })
                        except Exception as e_fb:
                            if PRODUCT_DEBUG_LOG:
                                vlog(f"  [F{frame_count}] PRODUCT fallback warning: {e_fb}")

                    if PRODUCT_ROI_TOP_K > 0 and len(roi_dets) > PRODUCT_ROI_TOP_K:
                        roi_dets = sorted(roi_dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True)[:PRODUCT_ROI_TOP_K]
                        if PRODUCT_DEBUG_LOG:
                            vlog(
                                f"  [F{frame_count}] PRODUCT roi_topk roi={roi} "
                                f"kept={len(roi_dets)} top_k={PRODUCT_ROI_TOP_K}"
                            )

                    detections.extend(remap_detections(roi_dets, roi))

                if PRODUCT_DEBUG_LOG:
                    vlog(f"  [F{frame_count}] PRODUCT raw_detected={len(detections)}")
                    for d in detections:
                        vlog(
                            f"    raw track={d['track_id']} cls={d['cls_name']} "
                            f"conf={d['conf']:.3f} box={d['box']}"
                        )

                if not detections:
                    # YOLO path: use plain per-frame detection with tracker fallback.
                    pass

                # Merge overlapping detections so impossible stacked boxes collapse.
                clusters = []
                for det in detections:
                    placed = False
                    for cl in clusters:
                        if any(box_iou(det["box"], d["box"]) >= PRODUCT_CLUSTER_IOU for d in cl):
                            cl.append(det)
                            placed = True
                            break
                    if not placed:
                        clusters.append([det])

                for cl in clusters:
                        # One rectangle per cluster (union box).
                        ux1 = min(d["box"][0] for d in cl)
                        uy1 = min(d["box"][1] for d in cl)
                        ux2 = max(d["box"][2] for d in cl)
                        uy2 = max(d["box"][3] for d in cl)

                        # Pick base class by confidence + class priority.
                        best = max(
                            cl,
                            key=lambda d: (
                                d["conf"] + 0.05 * PRODUCT_CLASS_PRIORITY.get(d["cls_name"], 0)
                            ),
                        )
                        chosen_name = best["cls_name"]

                        # Temporal voting by cluster center grid cell.
                        cx = (ux1 + ux2) // 2
                        cy = (uy1 + uy2) // 2
                        key = (cx // 24, cy // 24)
                        hist = product_label_history.setdefault(key, [])
                        hist.append(chosen_name)
                        if len(hist) > PRODUCT_STABLE_FRAMES:
                            hist[:] = hist[-PRODUCT_STABLE_FRAMES:]
                        voted_name = max(set(hist), key=hist.count)

                        stable_products.append({
                            "box": [ux1, uy1, ux2, uy2],
                            "label": voted_name,
                            "conf": float(best["conf"]),
                            "track_id": int(best.get("track_id", -1)),
                        })
                if PRODUCT_DEBUG_LOG:
                    vlog(f"  [F{frame_count}] PRODUCT stable={len(stable_products)}")
                    for sp in stable_products:
                        vlog(
                            f"    stable track={sp['track_id']} cls={sp['label']} "
                            f"conf={sp['conf']:.3f} box={sp['box']}"
                        )
            except Exception as e:
                vlog(f"  [F{frame_count}] Product detect warning: {e}")
        elif (not fast_qwen_evidence_mode) and use_kiosk_zone_logic and PRODUCT_DETECT_ENABLE and PRODUCT_DEBUG_LOG:
            vlog(
                f"  [F{frame_count}] PRODUCT skipped_all reason="
                f"{'no_accepted_person' if not accepted_person_present else ('no_hand_activity' if PRODUCT_REQUIRE_HAND_ACTIVITY and not hand_activity_present else 'disabled')}"
            )

        # Optional phone suppression pass using COCO cell phone class (67).
        if (not fast_qwen_evidence_mode) and use_kiosk_zone_logic and PHONE_SUPPRESS_ENABLE:
            try:
                ph_res = person_model(clean, classes=[67], conf=0.20, iou=0.5, verbose=False)
                if ph_res and ph_res[0].boxes is not None and len(ph_res[0].boxes) > 0:
                    phone_boxes = [list(map(int, b)) for b in ph_res[0].boxes.xyxy.cpu().numpy()]
                    if PRODUCT_DEBUG_LOG:
                        vlog(f"  [F{frame_count}] PHONE detected={len(phone_boxes)}")
            except Exception as e:
                if PRODUCT_DEBUG_LOG:
                    vlog(f"  [F{frame_count}] PHONE detect warning: {e}")

        current_gids = set()
        claimed_gids = set()
        person_boxes = []
        person_box_map = {}
        person_hands_map = {}

        seen_fallback_ids = set()
        for box, tid in valid:
            cx = int((box[0] + box[2]) * 0.5)
            cy = int((box[1] + box[3]) * 0.5)
            fallback_tracks[int(tid)] = {"cx": cx, "cy": cy, "missing": 0}
            seen_fallback_ids.add(int(tid))
        for fid in list(fallback_tracks.keys()):
            if fid in seen_fallback_ids:
                continue
            fallback_tracks[fid]["missing"] = int(fallback_tracks[fid].get("missing", 0)) + 1
            if fallback_tracks[fid]["missing"] > KIOSK_FALLBACK_ID_MAX_MISSING:
                del fallback_tracks[fid]

        if valid:
            assignments = []
            for box, tid in valid:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if USE_ENTRY_REID_TRACKING:
                    gid = assign_or_relink_person(clean, [x1, y1, x2, y2], cx, cy, int(tid), claimed_gids, frame_count)
                    if gid is None:
                        continue
                else:
                    gid = int(tid)
                    if gid in claimed_gids:
                        continue
                    claimed_gids.add(gid)
                assignments.append((gid, [x1, y1, x2, y2], int(tid), cx, cy))

            for gid, box_xyxy, tid, cx, cy in assignments:
                x1, y1, x2, y2 = box_xyxy
                mem = track_memory.setdefault(
                    int(gid),
                    {
                        "last_cx": int(cx),
                        "last_cy": int(cy),
                        "last_box": list(box_xyxy),
                        "missing": 0,
                        "seen": 0,
                        "views": [],
                    },
                )
                gap_since_seen = int(mem.get("missing", 0))
                mem["last_cx"] = int(cx)
                mem["last_cy"] = int(cy)
                mem["last_box"] = list(box_xyxy)
                mem["missing"] = 0
                mem["seen"] = int(mem.get("seen", 0)) + 1
                mem["last_visible_frame"] = clean.copy()
                mem["last_visible_frame_idx"] = int(frame_count)
                assigned_age = max(0, frame_count - int(mem.get("assigned_frame", frame_count)))
                should_try_periodic_update = (
                    person_reid_model is not None
                    and mem["seen"] % max(1, PERSON_REID_UPDATE_EVERY) == 0
                    and assigned_age >= max(0, KIOSK_REID_UPDATE_MIN_ASSIGNED_FRAMES)
                )
                should_try_gap_update = (
                    person_reid_model is not None
                    and gap_since_seen >= max(1, PERSON_REID_NEW_VIEW_GAP)
                    and assigned_age >= max(0, KIOSK_REID_UPDATE_MIN_ASSIGNED_FRAMES)
                )
                if should_try_periodic_update or should_try_gap_update:
                    emb = extract_customer_library_embedding(clean, box_xyxy, cache_key=int(tid))
                    if emb is not None:
                        update_reason = None
                        if should_try_periodic_update:
                            update_reason = f"periodic_{max(1, PERSON_REID_UPDATE_EVERY)}"
                        if should_try_gap_update and is_new_view_candidate(mem.get("views", []), emb):
                            update_reason = (
                                f"new_view_after_gap_{gap_since_seen}"
                                if update_reason is None
                                else f"{update_reason}+new_view_after_gap_{gap_since_seen}"
                            )
                        if update_reason is not None:
                            sim_os_verify = avg_sim_against_views_tensor(emb, mem.get("views", []))
                            curr_fu_verify, curr_fl_verify = extract_customer_library_fashion_pair(
                                clean, box_xyxy, cache_key=int(tid)
                            )
                            sim_fc_verify, sim_fc_upper_verify, sim_fc_lower_verify = customer_library_fashion_pair_similarity_breakdown(
                                curr_fu_verify,
                                curr_fl_verify,
                                mem.get("fashion_upper_init"),
                                mem.get("fashion_lower_init"),
                            )
                            fused_verify = sim_os_verify
                            if sim_fc_verify >= 0.0:
                                a_verify = customer_library_fashion_cross_alpha()
                                fused_verify = (1.0 - a_verify) * sim_os_verify + a_verify * sim_fc_verify
                            is_entry_gallery_gid = str(mem.get("source", "")) == "entry_gallery"
                            verify_min_os = (
                                KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_OS
                                if is_entry_gallery_gid
                                else KIOSK_REID_UPDATE_VERIFY_MIN_OS
                            )
                            verify_min_fc = (
                                KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_FC
                                if is_entry_gallery_gid
                                else KIOSK_REID_UPDATE_VERIFY_MIN_FC
                            )
                            verify_min_fused = (
                                KIOSK_ENTRY_REID_UPDATE_VERIFY_MIN_FUSED
                                if is_entry_gallery_gid
                                else KIOSK_REID_UPDATE_VERIFY_MIN_FUSED
                            )
                            verify_ok = (
                                sim_os_verify >= verify_min_os
                                and fused_verify >= verify_min_fused
                                and (
                                    sim_fc_verify < 0.0
                                    or sim_fc_verify >= verify_min_fc
                                )
                            )
                            scene_ok = (
                                (not is_entry_gallery_gid)
                                or accepted_count <= max(1, KIOSK_ENTRY_REID_UPDATE_MAX_PEOPLE)
                            )
                            compete_ok = True
                            best_comp_gid = None
                            best_comp_os = -1.0
                            best_comp_fc = -1.0
                            best_comp_fused = -1.0
                            if verify_ok and scene_ok and persistent_gallery:
                                for other_gid, entry in persistent_gallery.items():
                                    other_gid = int(other_gid)
                                    if other_gid == int(gid):
                                        continue
                                    other_views = entry.get("views") or []
                                    if not other_views:
                                        continue
                                    other_sim_os = avg_sim_against_views_tensor(emb, other_views)
                                    other_sim_fc, _other_fc_upper, _other_fc_lower = customer_library_fashion_pair_similarity_breakdown(
                                        curr_fu_verify,
                                        curr_fl_verify,
                                        entry.get("fashion_upper_init"),
                                        entry.get("fashion_lower_init"),
                                    )
                                    other_fused = other_sim_os
                                    if other_sim_fc >= 0.0:
                                        a_verify = customer_library_fashion_cross_alpha()
                                        other_fused = (1.0 - a_verify) * other_sim_os + a_verify * other_sim_fc
                                    if other_fused > best_comp_fused:
                                        best_comp_gid = other_gid
                                        best_comp_os = other_sim_os
                                        best_comp_fc = other_sim_fc
                                        best_comp_fused = other_fused
                                if (
                                    best_comp_gid is not None
                                    and best_comp_fused >= (fused_verify + KIOSK_REID_UPDATE_COMPETE_MARGIN)
                                ):
                                    compete_ok = False
                            if verify_ok and scene_ok and compete_ok:
                                mem["views"] = ema_update_views(mem.get("views", []), emb)
                                save_reid_fashion_debug_crops(
                                    clean,
                                    box_xyxy,
                                    gid,
                                    vname,
                                    frame_count,
                                    "kiosk_update_view",
                                )
                                if PRODUCT_DEBUG_LOG:
                                    vlog(
                                        f"  [F{frame_count}] KIOSK reid_view_update gid={gid} "
                                        f"tracker={tid} reason={update_reason} seen={mem['seen']} "
                                        f"gap_since_seen={gap_since_seen} views={len(mem.get('views', []))} "
                                        f"os={sim_os_verify:.3f} fc={sim_fc_verify:.3f} "
                                        f"fc_upper={sim_fc_upper_verify:.3f} fc_lower={sim_fc_lower_verify:.3f} "
                                        f"fused={fused_verify:.3f} "
                                        f"scene_people={accepted_count} "
                                        f"comp_gid={best_comp_gid} comp_os={best_comp_os:.3f} "
                                        f"comp_fc={best_comp_fc:.3f} comp_fused={best_comp_fused:.3f}"
                                    )
                            elif PRODUCT_DEBUG_LOG:
                                if verify_ok and not scene_ok:
                                    vlog(
                                        f"  [F{frame_count}] KIOSK reid_view_skip gid={gid} "
                                        f"tracker={tid} reason=crowded_scene_guard "
                                        f"update_reason={update_reason} seen={mem['seen']} "
                                        f"gap_since_seen={gap_since_seen} "
                                        f"scene_people={accepted_count} max_people={KIOSK_ENTRY_REID_UPDATE_MAX_PEOPLE} "
                                        f"os={sim_os_verify:.3f}/{verify_min_os:.3f} "
                                        f"fc={sim_fc_verify:.3f}/{verify_min_fc:.3f} "
                                        f"fc_upper={sim_fc_upper_verify:.3f} fc_lower={sim_fc_lower_verify:.3f} "
                                        f"fused={fused_verify:.3f}/{verify_min_fused:.3f}"
                                    )
                                elif verify_ok and not compete_ok:
                                    vlog(
                                        f"  [F{frame_count}] KIOSK reid_view_skip gid={gid} "
                                        f"tracker={tid} reason=competing_id_stronger "
                                        f"update_reason={update_reason} seen={mem['seen']} "
                                        f"gap_since_seen={gap_since_seen} "
                                        f"os={sim_os_verify:.3f} fc={sim_fc_verify:.3f} "
                                        f"fc_upper={sim_fc_upper_verify:.3f} fc_lower={sim_fc_lower_verify:.3f} "
                                        f"fused={fused_verify:.3f} "
                                        f"comp_gid={best_comp_gid} comp_os={best_comp_os:.3f} "
                                        f"comp_fc={best_comp_fc:.3f} comp_fused={best_comp_fused:.3f} "
                                        f"margin={KIOSK_REID_UPDATE_COMPETE_MARGIN:.3f}"
                                    )
                                else:
                                    vlog(
                                        f"  [F{frame_count}] KIOSK reid_view_skip gid={gid} "
                                        f"tracker={tid} reason=identity_drift_guard "
                                        f"update_reason={update_reason} seen={mem['seen']} "
                                        f"gap_since_seen={gap_since_seen} "
                                        f"os={sim_os_verify:.3f}/{verify_min_os:.3f} "
                                        f"fc={sim_fc_verify:.3f}/{verify_min_fc:.3f} "
                                        f"fc_upper={sim_fc_upper_verify:.3f} fc_lower={sim_fc_lower_verify:.3f} "
                                        f"fused={fused_verify:.3f}/{verify_min_fused:.3f}"
                                    )
                        elif should_try_gap_update and PRODUCT_DEBUG_LOG:
                            vlog(
                                f"  [F{frame_count}] KIOSK reid_view_skip gid={gid} "
                                f"tracker={tid} reason=gap_but_not_new_view "
                                f"gap_since_seen={gap_since_seen} seen={mem['seen']}"
                            )

                color = (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                person_text = f"person_id={int(gid)}"
                cv2.putText(
                    frame,
                    person_text,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                if gid in holding_display:
                    hold_text = holding_display[gid]["text"]
                    cv2.putText(
                        frame,
                        hold_text,
                        (x1, max(36, y1 - 26)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                person_boxes.append({
                    "gid": int(gid),
                    "box": list(box_xyxy),
                    "cx": int(cx),
                    "cy": int(cy),
                })
                person_box_map[int(gid)] = list(box_xyxy)
                person_hands_map[int(gid)] = pre_person_hands_map.get(int(tid), [])
                current_gids.add(int(gid))
                if SAVE_PERSON_CROPS:
                    crop = clean[max(0, y1):min(new_h, y2), max(0, x1):min(new_w, x2)]
                else:
                    crop = None
                if SAVE_PERSON_CROPS and crop is not None and crop.size > 0:
                    gid_dir = os.path.join(person_crops_dir, f"ID{gid}")
                    os.makedirs(gid_dir, exist_ok=True)
                    crop_path = os.path.join(gid_dir, f"{vname}_F{frame_count:06d}.jpg")
                    try:
                        cv2.imwrite(crop_path, crop)
                    except Exception:
                        pass
        if PERSON_DEBUG_LOG:
            vlog(
                f"  [F{frame_count}] PERSON raw={raw_person_count} id_ready={id_ready_count} "
                f"pass_h={pass_h_count} pass_area={pass_area_count} pass_vis={pass_vis_count} "
                f"accepted={accepted_count}"
            )

        # Customer entry events (first time seen in this video).
        for gid in sorted(current_gids):
            cs = customer_state.setdefault(gid, {"entered": False, "exited": False})
            cs["source"] = str(track_memory.get(gid, {}).get("source", "kiosk_continuous"))
            group_id = group_id_for_person(gid)
            group_members_by_gid.setdefault(int(group_id), set()).add(int(gid))
            if not cs.get("entered", False):
                cs["entered"] = True
                cs["entry_frame"] = int(frame_count)
                events.append({
                    "person_id": int(gid),
                    "group_id": int(group_id),
                    "event": "Entry",
                    "frame": int(frame_count),
                })
                vlog(f"  [F{frame_count}] EVENT gid={gid} Entry")
            cs["missing"] = 0
            if gid in person_box_map:
                cs["last_box"] = list(person_box_map[gid])
                cs["last_seen_frame"] = int(frame_count)

        if fast_qwen_evidence_mode and current_gids:
            groups_seen_now = {}
            for gid in sorted(current_gids):
                group_id = group_id_for_person(gid)
                groups_seen_now.setdefault(int(group_id), []).append(person_box_map.get(gid))
            for group_id, member_boxes in groups_seen_now.items():
                group_box = union_boxes(member_boxes)
                if group_box is None:
                    continue
                if box_center_in_box(group_box, exit_zone):
                    last_exit = int(group_evidence_tag_counts.get((int(group_id), "exit_last_frame"), -10_000))
                    if frame_count - last_exit >= FAST_EVIDENCE_EXIT_GAP:
                        record_group_evidence(
                            group_id,
                            "exit_view",
                            frame_count,
                            "group visible near exit",
                            primary_box=group_box,
                            event_key=f"exit_minute_{int(frame_count // max(1, int(round(float(max(1.0, fps)) * 60.0))))}",
                        )
                        group_evidence_tag_counts[(int(group_id), "exit_last_frame")] = int(frame_count)
                        events.append({
                            "group_id": int(group_id),
                            "event": "GroupExitView",
                            "frame": int(frame_count),
                        })
            if FAST_HAND_EVENT_EXPORT_ENABLE:
                for gid in sorted(current_gids):
                    wrists = person_hands_map.get(gid, [])
                    person_box = person_box_map.get(gid)
                    if not wrists or person_box is None:
                        continue
                    group_id = group_id_for_person(gid)
                    hand_box = points_to_box(wrists, FAST_HAND_EVENT_PAD, new_w, new_h)
                    weak_kiosk_label = detect_kiosk_label(person_box=person_box, item_box=hand_box)
                    weak_saved_frames_key = int(group_id)
                    if hand_box is None:
                        if FAST_HAND_EVENT_LOG:
                            vlog(
                                f"  [F{frame_count}] HAND_HOLD skip gid={gid} group={group_id} "
                                f"reason=no_hand_box wrists={len(wrists)}"
                            )
                        continue
                    confirm_box = detect_handheld_object_box(clean, hand_box) if FAST_HOLD_CONFIRM_ENABLE else hand_box
                    if confirm_box is None:
                        body_carry_saved = False
                        body_zone = torso_bag_zone(person_box) if FAST_BODY_CARRY_ENABLE else None
                        body_confirm_box = (
                            detect_handheld_object_box(clean, body_zone)
                            if body_zone is not None and FAST_HOLD_CONFIRM_ENABLE
                            else body_zone
                        )
                        if body_confirm_box is not None and frame_count not in group_saved_frames.setdefault(int(group_id), set()):
                            body_sig = extract_box_signature(clean, body_confirm_box)
                            prev_item_state = group_person_item_state.get(int(gid))
                            same_item = False
                            if prev_item_state is not None:
                                prev_item_box = prev_item_state.get("box")
                                prev_item_sig = prev_item_state.get("sig")
                                same_item = is_same_hand_item(body_confirm_box, body_sig, prev_item_box, prev_item_sig)
                                last_saved = int(prev_item_state.get("last_saved_frame", -10_000))
                                if same_item and (frame_count - last_saved) < FAST_HAND_ITEM_RESAVE_GAP:
                                    body_confirm_box = None
                            prev_hand = person_hand_export_state.get(int(gid))
                            if body_confirm_box is not None and prev_hand is not None:
                                prev_box = prev_hand.get("box")
                                prev_sig = prev_hand.get("sig")
                                box_similar = prev_box is not None and box_iou(body_confirm_box, prev_box) >= FAST_HAND_EVENT_DEDUP_IOU
                                sig_similar = False
                                if body_sig is not None and prev_sig is not None:
                                    sig_similar = emb_cos_sim(body_sig, prev_sig) >= FAST_HAND_EVENT_DEDUP_SIM
                                if box_similar and sig_similar:
                                    body_confirm_box = None
                            if body_confirm_box is not None:
                                if prev_item_state is not None and same_item:
                                    item_event_id = prev_item_state.get("item_event_id")
                                else:
                                    person_item_event_counter[int(gid)] = int(person_item_event_counter.get(int(gid), 0)) + 1
                                    item_event_id = int(person_item_event_counter[int(gid)])
                                body_path = record_group_evidence(
                                    group_id,
                                    "body_carry",
                                    frame_count,
                                    f"person_{int(gid)} carrying object against body",
                                    primary_box=person_box,
                                    secondary_box=body_confirm_box,
                                    person_id=int(gid),
                                    item_event_id=item_event_id,
                                    kiosk=detect_kiosk_label(person_box=person_box, item_box=body_confirm_box),
                                )
                                if body_path is not None:
                                    group_evidence_tag_counts[(int(group_id), "body_carry_last_frame")] = int(frame_count)
                                    person_hand_export_state[int(gid)] = {
                                        "box": list(body_confirm_box),
                                        "sig": body_sig,
                                    }
                                    group_saved_frames.setdefault(int(group_id), set()).add(int(frame_count))
                                    group_person_item_state[int(gid)] = {
                                        "last_saved_frame": int(frame_count),
                                        "box": list(body_confirm_box),
                                        "sig": body_sig,
                                        "tag": "body_carry",
                                        "item_event_id": item_event_id,
                                    }
                                    group_evidence_tag_counts[(int(group_id), "hand_confirm_last_frame")] = int(frame_count)
                                    events.append({
                                        "person_id": int(gid),
                                        "group_id": int(group_id),
                                        "event": "BodyCarryView",
                                        "kiosk": detect_kiosk_label(person_box=person_box, item_box=body_confirm_box),
                                        "frame": int(frame_count),
                                    })
                                    body_carry_saved = True
                                    if FAST_HAND_EVENT_LOG:
                                        vlog(
                                            f"  [F{frame_count}] BODY_CARRY save gid={gid} group={group_id} "
                                            f"box={body_confirm_box} image={body_path}"
                                        )
                        weak_saved = False
                        if FAST_HAND_EVENT_WEAK_ENABLE:
                            prev_hand = person_hand_export_state.get(int(gid))
                            last_confirm = int(group_evidence_tag_counts.get((int(group_id), "hand_confirm_last_frame"), -10_000))
                            last_weak = int(group_evidence_tag_counts.get((int(group_id), "hand_weak_last_frame"), -10_000))
                            prev_item_state = group_person_item_state.get(int(gid))
                            if (
                                prev_hand is not None
                                and prev_item_state is not None
                                and last_confirm >= 0
                                and frame_count - last_confirm <= FAST_HAND_EVENT_WEAK_MAX_GAP
                                and frame_count - last_weak >= FAST_HAND_EVENT_WEAK_GAP
                                and frame_count - int(prev_item_state.get("last_saved_frame", -10_000)) >= FAST_HAND_ITEM_RESAVE_GAP
                                and frame_count not in group_saved_frames.setdefault(weak_saved_frames_key, set())
                            ):
                                weak_path = record_group_evidence(
                                    group_id,
                                    "hand_hold_weak",
                                    frame_count,
                                    f"person_{int(gid)} likely still holding object",
                                    primary_box=person_box,
                                    secondary_box=hand_box,
                                    person_id=int(gid),
                                    item_event_id=prev_item_state.get("item_event_id"),
                                    kiosk=weak_kiosk_label,
                                )
                                if weak_path is not None:
                                    group_evidence_tag_counts[(int(group_id), "hand_weak_last_frame")] = int(frame_count)
                                    group_saved_frames.setdefault(weak_saved_frames_key, set()).add(int(frame_count))
                                    group_person_item_state[int(gid)] = {
                                        "last_saved_frame": int(frame_count),
                                        "box": list(hand_box),
                                        "sig": prev_item_state.get("sig"),
                                        "tag": "hand_hold_weak",
                                    }
                                    events.append({
                                        "person_id": int(gid),
                                        "group_id": int(group_id),
                                        "event": "HandHoldWeakView",
                                        "kiosk": detect_kiosk_label(person_box=person_box, item_box=hand_box),
                                        "frame": int(frame_count),
                                    })
                                    weak_saved = True
                                    if FAST_HAND_EVENT_LOG:
                                        vlog(
                                            f"  [F{frame_count}] HAND_HOLD_WEAK save gid={gid} group={group_id} "
                                            f"hand_box={hand_box} image={weak_path} "
                                            f"last_confirm_gap={frame_count - last_confirm}"
                                        )
                        if FAST_HAND_EVENT_LOG:
                            if not weak_saved and not body_carry_saved:
                                vlog(
                                    f"  [F{frame_count}] HAND_HOLD skip gid={gid} group={group_id} "
                                    f"reason=no_object_confirm hand_box={hand_box}"
                                )
                        if body_carry_saved:
                            continue
                        continue
                    hand_sig = extract_box_signature(clean, confirm_box)
                    hand_kiosk_label = detect_kiosk_label(person_box=person_box, item_box=confirm_box)
                    hand_saved_frames_key = int(group_id)
                    prev_item_state = group_person_item_state.get(int(gid))
                    if frame_count in group_saved_frames.setdefault(hand_saved_frames_key, set()):
                        if FAST_HAND_EVENT_LOG:
                            vlog(
                                f"  [F{frame_count}] HAND_HOLD skip gid={gid} group={group_id} "
                                f"reason=frame_already_saved"
                            )
                        continue
                    if prev_item_state is not None:
                        prev_item_box = prev_item_state.get("box")
                        prev_item_sig = prev_item_state.get("sig")
                        same_item = is_same_hand_item(confirm_box, hand_sig, prev_item_box, prev_item_sig)
                        last_saved = int(prev_item_state.get("last_saved_frame", -10_000))
                        if same_item and (frame_count - last_saved) < FAST_HAND_ITEM_RESAVE_GAP:
                            if FAST_HAND_EVENT_LOG:
                                vlog(
                                    f"  [F{frame_count}] HAND_HOLD skip gid={gid} group={group_id} "
                                    f"reason=same_item_gap last={last_saved} gap={frame_count - last_saved} "
                                    f"min={FAST_HAND_ITEM_RESAVE_GAP}"
                                )
                            continue
                    else:
                        same_item = False
                    prev_hand = person_hand_export_state.get(int(gid))
                    if prev_hand is not None:
                        prev_box = prev_hand.get("box")
                        prev_sig = prev_hand.get("sig")
                        box_similar = prev_box is not None and box_iou(confirm_box, prev_box) >= FAST_HAND_EVENT_DEDUP_IOU
                        sig_similar = False
                        if hand_sig is not None and prev_sig is not None:
                            sig_similar = emb_cos_sim(hand_sig, prev_sig) >= FAST_HAND_EVENT_DEDUP_SIM
                        if box_similar and sig_similar:
                            if FAST_HAND_EVENT_LOG:
                                vlog(
                                    f"  [F{frame_count}] HAND_HOLD skip gid={gid} group={group_id} "
                                    f"reason=dedup iou={box_iou(confirm_box, prev_box):.3f} "
                                    f"sim={emb_cos_sim(hand_sig, prev_sig):.3f}"
                                )
                            continue
                    if prev_item_state is not None and same_item:
                        item_event_id = prev_item_state.get("item_event_id")
                    else:
                        person_item_event_counter[int(gid)] = int(person_item_event_counter.get(int(gid), 0)) + 1
                        item_event_id = int(person_item_event_counter[int(gid)])
                    hand_path = record_group_evidence(
                        group_id,
                        "hand_hold",
                        frame_count,
                        f"person_{int(gid)} confirmed object near hand",
                        primary_box=person_box,
                        secondary_box=confirm_box,
                        person_id=int(gid),
                        item_event_id=item_event_id,
                        kiosk=hand_kiosk_label,
                    )
                    if hand_path is None:
                        if FAST_HAND_EVENT_LOG:
                            vlog(
                                f"  [F{frame_count}] HAND_HOLD skip gid={gid} group={group_id} "
                                f"reason=tag_limit confirm_box={confirm_box}"
                            )
                        continue
                    group_evidence_tag_counts[(int(group_id), "hand_last_frame")] = int(frame_count)
                    person_hand_export_state[int(gid)] = {
                        "box": list(confirm_box),
                        "sig": hand_sig,
                    }
                    group_saved_frames.setdefault(hand_saved_frames_key, set()).add(int(frame_count))
                    group_person_item_state[int(gid)] = {
                        "last_saved_frame": int(frame_count),
                        "box": list(confirm_box),
                        "sig": hand_sig,
                        "tag": "hand_hold",
                        "item_event_id": item_event_id,
                    }
                    group_evidence_tag_counts[(int(group_id), "hand_confirm_last_frame")] = int(frame_count)
                    events.append({
                        "person_id": int(gid),
                        "group_id": int(group_id),
                        "event": "HandHoldView",
                        "kiosk": detect_kiosk_label(person_box=person_box, item_box=confirm_box),
                        "frame": int(frame_count),
                    })
                    if FAST_HAND_EVENT_LOG:
                        vlog(
                            f"  [F{frame_count}] HAND_HOLD save gid={gid} group={group_id} "
                            f"box={confirm_box} image={hand_path}"
                        )
                    if FAST_BAG_EVENT_EXPORT_ENABLE:
                        bag_zone = torso_bag_zone(person_box)
                        if (
                            bag_zone is not None
                            and box_center_in_box(confirm_box, bag_zone)
                            and any(point_in_box(wx, wy, bag_zone) for wx, wy in wrists)
                        ):
                            last_bag = int(group_evidence_tag_counts.get((int(group_id), "bag_last_frame"), -10_000))
                            if frame_count - last_bag >= FAST_BAG_EVENT_GAP and frame_count not in group_saved_frames.setdefault(hand_saved_frames_key, set()):
                                record_group_evidence(
                                    group_id,
                                    "bag_pocket",
                                    frame_count,
                                    f"person_{int(gid)} putting item into bag/pocket",
                                    primary_box=person_box,
                                    secondary_box=confirm_box,
                                    person_id=int(gid),
                                    item_event_id=item_event_id,
                                    kiosk=hand_kiosk_label,
                                )
                                group_evidence_tag_counts[(int(group_id), "bag_last_frame")] = int(frame_count)
                                events.append({
                                    "person_id": int(gid),
                                    "group_id": int(group_id),
                                    "event": "BagPocketView",
                                    "kiosk": detect_kiosk_label(person_box=person_box, item_box=confirm_box),
                                    "frame": int(frame_count),
                                })

        # Update missing counters and emit Exit when customer disappears long enough.
        for gid, cs in customer_state.items():
            if gid in current_gids:
                continue
            cs["missing"] = int(cs.get("missing", 0)) + 1
            if cs["missing"] == 1:
                group_id = group_id_for_person(gid)
                mem = track_memory.get(int(gid), {})
                prev_item_state = group_person_item_state.get(int(gid))
                last_saved_frame = int(prev_item_state.get("last_saved_frame", -10_000)) if prev_item_state is not None else -10_000
                last_visible_frame_idx = int(mem.get("last_visible_frame_idx", -10_000))
                recent_carry = prev_item_state is not None and (last_visible_frame_idx - last_saved_frame) <= FAST_PRE_EXIT_CARRY_GAP
                if recent_carry:
                    saved_pre_exit = False
                    for offset in FAST_PRE_EXIT_CARRY_OFFSETS:
                        target_frame_idx = int(last_visible_frame_idx - int(offset))
                        candidate = None
                        for rec in reversed(recent_full_frames):
                            if int(rec.get("frame_idx", -10_000)) == target_frame_idx:
                                candidate = {
                                    "frame_idx": int(rec.get("frame_idx", target_frame_idx)),
                                    "frame": rec.get("frame"),
                                    "box": mem.get("last_box"),
                                }
                                break
                        if candidate is None:
                            nearest = None
                            for rec in reversed(recent_full_frames):
                                rec_idx = int(rec.get("frame_idx", -10_000))
                                if rec_idx <= target_frame_idx:
                                    nearest = rec
                                    break
                            if nearest is not None:
                                candidate = {
                                    "frame_idx": int(nearest.get("frame_idx", target_frame_idx)),
                                    "frame": nearest.get("frame"),
                                    "box": mem.get("last_box"),
                                }
                        if candidate is None and int(offset) == 0 and mem.get("last_visible_frame") is not None:
                            candidate = {
                                "frame_idx": int(last_visible_frame_idx),
                                "frame": mem.get("last_visible_frame"),
                                "box": mem.get("last_box"),
                            }
                        if candidate is None:
                            continue
                        exit_anchor_path = record_group_evidence_from_frame(
                            candidate.get("frame"),
                            group_id,
                            "pre_exit_carry",
                            int(candidate.get("frame_idx", last_visible_frame_idx)),
                            f"person_{int(gid)} pre-exit carry anchor",
                            primary_box=candidate.get("box"),
                            person_id=int(gid),
                            item_event_id=prev_item_state.get("item_event_id"),
                            pre_exit_offset=int(offset),
                            kiosk=detect_kiosk_label(person_box=candidate.get("box"), item_box=None),
                        )
                        if exit_anchor_path is not None:
                            saved_pre_exit = True
                            events.append({
                                "person_id": int(gid),
                                "group_id": int(group_id),
                                "event": "PreExitCarryView",
                                "frame": int(candidate.get("frame_idx", last_visible_frame_idx)),
                            })
                            vlog(
                                f"  [F{frame_count}] PRE_EXIT_CARRY save gid={gid} group={group_id} "
                                f"source_frame={int(candidate.get('frame_idx', last_visible_frame_idx))} "
                                f"offset={int(offset)} image={exit_anchor_path}"
                            )
                    if saved_pre_exit:
                        group_evidence_tag_counts[(int(group_id), f"pre_exit_last_frame_{int(gid)}")] = int(frame_count)
            if cs.get("entered", False) and (not cs.get("exited", False)) and cs["missing"] >= CUSTOMER_EXIT_MISSING_FRAMES:
                cs["exited"] = True
                cs["exit_frame"] = int(frame_count)
                group_id = group_id_for_person(gid)
                carried_out = sum(
                    1 for it in item_tracks_by_gid.get(group_id, [])
                    if item_counts_as_with_customer(it.get("state"))
                )
                interacted = len(item_tracks_by_gid.get(group_id, []))
                events.append({
                    "person_id": int(gid),
                    "group_id": int(group_id),
                    "event": "Exit",
                    "frame": int(frame_count),
                    "items_interacted": int(interacted),
                    "items_carried_out": int(carried_out),
                })
                vlog(
                    f"  [F{frame_count}] EVENT gid={gid} group={group_id} Exit "
                    f"items_interacted={interacted} items_carried_out={carried_out}"
                )

        # Associate products to nearest customer and count scan-like entries.
        item_seen_with_customer_frame_by_gid = {}
        if (not fast_qwen_evidence_mode) and use_kiosk_zone_logic and stable_products:
            # Process high-confidence first and enforce one item_id per frame.
            stable_products = sorted(stable_products, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
            item_ids_used_this_frame_by_gid = {}
            for p in stable_products:
                px1, py1, px2, py2 = p["box"]
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2
                p_ref_emb = None
                p_ref_cemb = None

                # Build/update a short-lived product streak before expensive matching.
                if p.get("track_id", -1) >= 0:
                    pkey = ("track", int(p["track_id"]))
                else:
                    pkey = ("grid", pcx // PRODUCT_KEY_GRID, pcy // PRODUCT_KEY_GRID)
                st = product_track_state.setdefault(
                    pkey, {"last_frame": -10_000, "streak": 0, "last_saved_frame": -10_000}
                )
                if frame_count - st["last_frame"] <= max(1, FRAME_SKIP):
                    st["streak"] += 1
                else:
                    st["streak"] = 1
                st["last_frame"] = frame_count
                match_ready = st["streak"] >= PRODUCT_MATCH_MIN_STREAK

                # Exclude handheld scanner-like object using reference image similarity.
                if (
                    match_ready
                    and SCANNER_FILTER_ENABLE
                    and (scanner_ref_emb is not None or scanner_ref_cemb is not None)
                ):
                    p_emb_scan = extract_product_embedding(clean, p["box"])
                    p_cemb_scan = extract_color_hist_embedding(clean, p["box"])
                    s_sim = emb_cos_sim(p_emb_scan, scanner_ref_emb)
                    s_csim = emb_cos_sim(p_cemb_scan, scanner_ref_cemb)
                    if (s_sim >= SCANNER_DINO_SIM_TH) or (s_csim >= SCANNER_COLOR_SIM_TH):
                        if PRODUCT_DEBUG_LOG:
                            vlog(
                                f"  [F{frame_count}] PRODUCT scanner_excluded "
                                f"track={p.get('track_id',-1)} cls={p['label']} "
                                f"sim={s_sim:.3f} csim={s_csim:.3f} "
                                f"th_sim={SCANNER_DINO_SIM_TH:.2f} th_csim={SCANNER_COLOR_SIM_TH:.2f}"
                            )
                        continue
                    p_ref_emb = p_emb_scan
                    p_ref_cemb = p_cemb_scan

                if match_ready and PRODUCT_REFERENCE_ENABLE and product_references:
                    if p_ref_emb is None:
                        p_ref_emb = extract_product_embedding(clean, p["box"])
                    if p_ref_cemb is None:
                        p_ref_cemb = extract_color_hist_embedding(clean, p["box"])
                    best_ref = None
                    best_ref_sim = -1.0
                    best_ref_csim = -1.0
                    best_ref_score = -1.0
                    for ref in product_references:
                        ref_sim = emb_cos_sim(p_ref_emb, ref.get("dino_emb"))
                        ref_csim = emb_cos_sim(p_ref_cemb, ref.get("color_emb"))
                        ref_score = max(ref_sim, ref_csim)
                        if ref_score > best_ref_score:
                            best_ref = ref
                            best_ref_sim = ref_sim
                            best_ref_csim = ref_csim
                            best_ref_score = ref_score
                    if best_ref is not None and (
                        best_ref_sim >= PRODUCT_REFERENCE_DINO_SIM_TH
                        or best_ref_csim >= PRODUCT_REFERENCE_COLOR_SIM_TH
                    ):
                        p["label"] = best_ref["label"]
                        p["reference_hit"] = True
                        p["reference_sim"] = best_ref_sim
                        p["reference_csim"] = best_ref_csim
                        p["reference_path"] = best_ref.get("path")
                        reference_hit_boxes.append(
                            {
                                "box": list(p["box"]),
                                "label": p["label"],
                                "sim": best_ref_sim,
                                "csim": best_ref_csim,
                            }
                        )
                        ref_crop_path = save_product_crop_unknown(
                            clean,
                            p["box"],
                            reference_hits_dir,
                            vname,
                            frame_count,
                            p["label"],
                            track_id=p.get("track_id", None),
                        )
                        if PRODUCT_DEBUG_LOG:
                            vlog(
                                f"  [F{frame_count}] PRODUCT reference_match "
                                f"track={p.get('track_id',-1)} label={p['label']} "
                                f"sim={best_ref_sim:.3f} csim={best_ref_csim:.3f} "
                                f"ref={best_ref.get('path', '?')} "
                                f"image={ref_crop_path}"
                            )
                elif PRODUCT_DEBUG_LOG and (SCANNER_FILTER_ENABLE or PRODUCT_REFERENCE_ENABLE):
                    vlog(
                        f"  [F{frame_count}] PRODUCT match_wait track={p.get('track_id',-1)} "
                        f"cls={p['label']} streak={st['streak']}/{PRODUCT_MATCH_MIN_STREAK}"
                    )

                best_gid = None
                best_dist = 1e9
                assoc_candidates = list(person_boxes)
                if not assoc_candidates:
                    for gid, mem in track_memory.items():
                        if mem.get("missing", 9999) <= KIOSK_ASSOC_MAX_MISSING:
                            assoc_candidates.append({
                                "gid": gid,
                                "cx": int(mem.get("last_cx", 0)),
                                "cy": int(mem.get("last_cy", 0)),
                            })
                for pb in assoc_candidates:
                    dx = pcx - pb["cx"]
                    dy = pcy - pb["cy"]
                    dist = float((dx * dx + dy * dy) ** 0.5)
                    if dist < best_dist:
                        best_dist = dist
                        best_gid = pb["gid"]

                dist_ok = PRODUCT_ASSOC_DISABLE_DIST_CAP or (best_dist <= PRODUCT_ASSOC_MAX_DIST)
                assoc_ok = (best_gid is not None and dist_ok)
                hold_fail_reason = None
                hold_fail_extra = ""
                if PRODUCT_DEBUG_LOG and PRODUCT_SKIP_LOG and not assoc_ok:
                    if best_gid is None:
                        reason = "no_person_assoc"
                        extra = ""
                    else:
                        reason = "assoc_too_far"
                        extra = f" dist={best_dist:.1f} max={PRODUCT_ASSOC_MAX_DIST}"
                    vlog(
                        f"  [F{frame_count}] PRODUCT skipped track={p.get('track_id',-1)} "
                        f"cls={p['label']} reason={reason}{extra}"
                    )
                if not assoc_ok:
                    hold_fail_reason = "assoc_fail"
                    hold_fail_extra = f" best_gid={best_gid} dist={best_dist:.1f}"

                if PRODUCT_DEBUG_LOG and st["streak"] < PRODUCT_MIN_TRACK_FRAMES:
                    vlog(
                        f"  [F{frame_count}] PRODUCT wait track={p.get('track_id',-1)} "
                        f"cls={p['label']} streak={st['streak']}/{PRODUCT_MIN_TRACK_FRAMES}"
                    )

                interacting_with_customer = False
                used_hands = False
                if assoc_ok and best_gid in person_box_map:
                    wrists = person_hands_map.get(best_gid, [])
                    if wrists:
                        used_hands = True
                        dmin = min(point_to_box_distance(wx, wy, p["box"]) for wx, wy in wrists)
                        interacting_with_customer = dmin <= POSE_HAND_NEAR_PX
                        if PRODUCT_DEBUG_LOG and PRODUCT_SKIP_LOG and not interacting_with_customer:
                            vlog(
                                f"  [F{frame_count}] PRODUCT skipped track={p.get('track_id',-1)} "
                                f"cls={p['label']} reason=not_near_hands gid={best_gid} box_dmin={dmin:.1f}"
                            )
                        if not interacting_with_customer:
                            hold_fail_reason = "not_near_hands"
                            hold_fail_extra = f" gid={best_gid} box_dmin={dmin:.1f} th={POSE_HAND_NEAR_PX}"
                    else:
                        # Fallback when wrists are not available.
                        if not HOLDING_REQUIRE_HANDS:
                            ex = expand_xyxy(person_box_map[best_gid], PRODUCT_PERSON_PAD, new_w, new_h)
                            interacting_with_customer = point_in_box(pcx, pcy, ex)
                            if PRODUCT_DEBUG_LOG and PRODUCT_SKIP_LOG and not interacting_with_customer:
                                vlog(
                                    f"  [F{frame_count}] PRODUCT skipped track={p.get('track_id',-1)} "
                                    f"cls={p['label']} reason=not_near_customer_body gid={best_gid}"
                                )
                            if not interacting_with_customer:
                                hold_fail_reason = "not_near_body"
                                hold_fail_extra = f" gid={best_gid}"
                        elif PRODUCT_DEBUG_LOG and PRODUCT_SKIP_LOG:
                            vlog(
                                f"  [F{frame_count}] PRODUCT skipped track={p.get('track_id',-1)} "
                                f"cls={p['label']} reason=no_hands_detected gid={best_gid}"
                            )
                        if not HOLDING_REQUIRE_HANDS and not interacting_with_customer:
                            hold_fail_reason = "not_near_body"
                            hold_fail_extra = f" gid={best_gid}"
                        elif HOLDING_REQUIRE_HANDS:
                            hold_fail_reason = "no_hands_detected"
                            hold_fail_extra = f" gid={best_gid}"

                # Suppress phone-like held objects.
                if interacting_with_customer and PHONE_SUPPRESS_ENABLE and phone_boxes:
                    pbox = [px1, py1, px2, py2]
                    if any(box_iou(pbox, ph) >= PHONE_IOU_SUPPRESS_TH for ph in phone_boxes):
                        interacting_with_customer = False
                        if PRODUCT_DEBUG_LOG and PRODUCT_SKIP_LOG:
                            vlog(
                                f"  [F{frame_count}] PRODUCT skipped track={p.get('track_id',-1)} "
                                f"cls={p['label']} reason=overlap_phone"
                            )
                        hold_fail_reason = "overlap_phone"
                        hold_fail_extra = f" gid={best_gid} iou_th={PHONE_IOU_SUPPRESS_TH:.2f}"

                if HOLD_FAIL_SUMMARY_LOG and not interacting_with_customer:
                    reason = hold_fail_reason or "unknown"
                    hold_fail_image = None
                    if SAVE_HOLD_FAIL_CROPS:
                        hold_fail_image = save_product_crop_unknown(
                            clean,
                            p["box"],
                            hold_fails_dir,
                            vname,
                            frame_count,
                            f"{p['label']}_{reason}",
                            track_id=p.get("track_id", None),
                        )
                    vlog(
                        f"  [F{frame_count}] HOLD_FAIL track={p.get('track_id',-1)} cls={p['label']} "
                        f"reason={reason}{hold_fail_extra} "
                        f"box={p['box']} image={hold_fail_image}"
                    )

                # Show product label/box only when it is interacting with customer.
                hs = None
                if interacting_with_customer and assoc_ok:
                    if p.get("track_id", -1) >= 0:
                        hold_key = ("track", int(best_gid), int(p.get("track_id", -1)))
                    else:
                        # Coarser fallback key to reduce jitter misses.
                        hold_key = ("grid", int(best_gid), pcx // 40, pcy // 40)
                    hs = hold_state.setdefault(
                        hold_key, {"last_frame": -10_000, "streak": 0, "active": False}
                    )
                    if frame_count - hs["last_frame"] <= max(2, FRAME_SKIP + 1):
                        hs["streak"] += 1
                    else:
                        hs["streak"] = 1
                    hs["last_frame"] = frame_count
                    if (not hs["active"]) and hs["streak"] >= HOLD_DETECT_MIN_STREAK:
                        hs["active"] = True
                        holding_display[int(best_gid)] = {
                            "ttl": HOLDING_DISPLAY_TTL,
                            "text": f"HOLDING {p['label']}",
                        }
                        events.append({
                            "person_id": int(best_gid),
                            "event": "HoldingDetected",
                            "label": p["label"],
                            "track_id": int(p.get("track_id", -1)),
                            "frame": int(frame_count),
                        })
                        vlog(
                            f"  [F{frame_count}] HOLDING gid={best_gid} "
                            f"label={p['label']} track={p.get('track_id',-1)} streak={hs['streak']}"
                        )
                # Update persistent display track for held items (prevents stacked rectangles).
                if hs is not None and hs.get("active", False) and assoc_ok:
                    gid_disp = int(best_gid)
                    dtracks = display_item_tracks_by_gid.setdefault(gid_disp, [])
                    pb = [px1, py1, px2, py2]
                    pcx2 = (pb[0] + pb[2]) // 2
                    pcy2 = (pb[1] + pb[3]) // 2
                    best_dt = None
                    best_dt_score = -1e9
                    for dt in dtracks:
                        gap = frame_count - int(dt.get("last_frame", -10_000))
                        if gap > ITEM_MATCH_MAX_GAP:
                            continue
                        lbox = dt.get("box", pb)
                        iou = box_iou(pb, lbox)
                        lcx = (lbox[0] + lbox[2]) // 2
                        lcy = (lbox[1] + lbox[3]) // 2
                        dist = float(((pcx2 - lcx) ** 2 + (pcy2 - lcy) ** 2) ** 0.5)
                        if iou >= ITEM_MATCH_IOU_TH or dist <= ITEM_MATCH_MAX_DIST:
                            same_label_bonus = 0.10 if dt.get("label") == p["label"] else 0.0
                            score = iou - 0.002 * dist + same_label_bonus
                            if score > best_dt_score:
                                best_dt_score = score
                                best_dt = dt
                    if best_dt is None:
                        dtracks.append({
                            "label": p["label"],
                            "item_id": None,
                            "box": list(pb),
                            "last_frame": int(frame_count),
                            "ttl": int(ITEM_DISPLAY_MEMORY_FRAMES),
                            "conf": float(p.get("conf", 0.0)),
                        })
                    else:
                        best_dt["label"] = p["label"]
                        best_dt["box"] = list(pb)
                        best_dt["last_frame"] = int(frame_count)
                        best_dt["ttl"] = int(ITEM_DISPLAY_MEMORY_FRAMES)
                        best_dt["conf"] = max(float(best_dt.get("conf", 0.0)), float(p.get("conf", 0.0)))
                # TEMP: disable product track/stability streak gating.
                if interacting_with_customer and (
                    frame_count - st["last_saved_frame"] >= PRODUCT_RECOUNT_GAP_FRAMES
                ):
                    owner_gid = int(best_gid)
                    gid_for_save = group_id_for_person(best_gid)
                    new_item_created = False
                    quality_ok, quality_sharp, quality_tg, quality_reason = assess_item_crop_quality(clean, p["box"])
                    p_emb = extract_product_embedding(clean, p["box"])
                    p_cemb = extract_color_hist_embedding(clean, p["box"])
                    # Per-customer item identity using temporal continuity (IoU/center distance/time gap).
                    gtracks = item_tracks_by_gid.setdefault(gid_for_save, [])
                    trackid_map = item_trackid_map_by_gid.setdefault(gid_for_save, {})
                    pending = pending_new_item_by_gid.setdefault(gid_for_save, {})
                    pb = p["box"]
                    pcx2 = (pb[0] + pb[2]) // 2
                    pcy2 = (pb[1] + pb[3]) // 2
                    best_track = None
                    best_track_score = -1e9
                    best_track_sim = -1.0
                    best_track_csim = -1.0
                    nearest_iou = -1.0
                    nearest_dist = -1.0
                    nearest_sim = -1.0
                    nearest_csim = -1.0
                    # Hard lock by detector track id when available to avoid item ID switching/splitting.
                    det_tid = int(p.get("track_id", -1))
                    if det_tid >= 0 and det_tid in trackid_map:
                        locked_item_id = int(trackid_map[det_tid])
                        for it in gtracks:
                            if int(it.get("item_id", -1)) == locked_item_id:
                                lock_sim = emb_cos_sim(p_emb, it.get("emb"))
                                lock_csim = emb_cos_sim(p_cemb, it.get("color_emb"))
                                lock_mix = lock_sim
                                if lock_sim >= 0.0 and lock_csim >= 0.0:
                                    lock_mix = (1.0 - COLOR_MATCH_WEIGHT) * lock_sim + COLOR_MATCH_WEIGHT * lock_csim
                                elif lock_csim >= 0.0:
                                    lock_mix = lock_csim
                                # Anti-switch guard: if appearance is too different, break lock
                                # instead of forcing this detection into the old item id.
                                if lock_mix >= 0.0 and lock_mix < ITEM_LOCK_MIN_SIM:
                                    if PRODUCT_DEBUG_LOG:
                                        vlog(
                                            f"  [F{frame_count}] ITEM lock_break gid={gid_for_save} "
                                            f"track={det_tid} item={locked_item_id:03d} "
                                            f"sim={lock_sim:.3f} csim={lock_csim:.3f} mix={lock_mix:.3f} "
                                            f"< {ITEM_LOCK_MIN_SIM:.2f}"
                                        )
                                    trackid_map.pop(det_tid, None)
                                else:
                                    best_track = it
                                    best_track_sim = lock_sim
                                    best_track_csim = lock_csim
                                break

                    for it in gtracks:
                        gap_limit = ITEM_MATCH_MAX_GAP_HOLDING if owner_gid in holding_display else ITEM_MATCH_MAX_GAP
                        if best_track is not None:
                            break
                        gap = frame_count - int(it.get("last_frame", -10_000))
                        if gap > gap_limit:
                            continue
                        lbox = it.get("last_box", pb)
                        iou = box_iou(pb, lbox)
                        lcx = (lbox[0] + lbox[2]) // 2
                        lcy = (lbox[1] + lbox[3]) // 2
                        dist = float(((pcx2 - lcx) ** 2 + (pcy2 - lcy) ** 2) ** 0.5)
                        nearest_iou = max(nearest_iou, iou)
                        if nearest_dist < 0 or dist < nearest_dist:
                            nearest_dist = dist

                        sim = emb_cos_sim(p_emb, it.get("emb"))
                        csim = emb_cos_sim(p_cemb, it.get("color_emb"))
                        nearest_sim = max(nearest_sim, sim)
                        nearest_csim = max(nearest_csim, csim)

                        hidden_relink_ok = False
                        if it.get("state") == "with_customer_hidden":
                            hidden_gap = frame_count - int(it.get("hidden_frame", it.get("last_frame", -10_000)))
                            hidden_relink_ok = (
                                hidden_gap <= ITEM_HIDDEN_RELINK_MAX_GAP
                                and (iou >= ITEM_HIDDEN_RELINK_MIN_IOU or dist <= ITEM_HIDDEN_RELINK_MAX_DIST)
                                and csim >= ITEM_HIDDEN_RELINK_MIN_CSIM
                                and sim >= ITEM_HIDDEN_RELINK_MIN_SIM
                            )

                        soft_relink_ok = (
                            gap <= ITEM_SOFT_RELINK_MAX_GAP
                            and (iou >= ITEM_SOFT_RELINK_MIN_IOU or dist <= ITEM_SOFT_RELINK_MAX_DIST)
                            and csim >= ITEM_SOFT_RELINK_MIN_CSIM
                            and sim >= ITEM_SOFT_RELINK_MIN_SIM
                        )

                        # Appearance-only matching (geometry does not decide identity).
                        if (
                            (sim < DINO_MIN_SIM_FOR_MATCH or csim < COLOR_MIN_SIM_FOR_MATCH)
                            and not hidden_relink_ok
                            and not soft_relink_ok
                        ):
                            continue

                        if (
                            it.get("label") is not None
                            and p.get("label") is not None
                            and it.get("label") != p.get("label")
                            and sim < ITEM_LABEL_DISAGREE_MIN_SIM
                        ):
                            continue

                        score = DINO_MATCH_WEIGHT * sim + COLOR_MATCH_WEIGHT * csim
                        if hidden_relink_ok:
                            score += 0.08
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_count}] ITEM hidden_relink gid={gid_for_save} "
                                    f"item={int(it.get('item_id', -1)):03d} "
                                    f"iou={iou:.3f} dist={dist:.1f} sim={sim:.3f} csim={csim:.3f}"
                                )
                        elif soft_relink_ok:
                            score += 0.04
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_count}] ITEM soft_relink gid={gid_for_save} "
                                    f"item={int(it.get('item_id', -1)):03d} "
                                    f"iou={iou:.3f} dist={dist:.1f} sim={sim:.3f} csim={csim:.3f}"
                                )
                        if score > best_track_score:
                            best_track_score = score
                            best_track = it
                            best_track_sim = sim
                            best_track_csim = csim

                    # Label-switch sanity: avoid overwriting an existing item with a different label
                    # when appearance evidence is weak.
                    if best_track is not None:
                        prev_label = best_track.get("label")
                        cur_label = p.get("label")
                        if (
                            prev_label is not None
                            and cur_label is not None
                            and prev_label != cur_label
                            and (best_track_sim < ITEM_LABEL_SWITCH_MIN_SIM or best_track_csim < ITEM_LABEL_SWITCH_MIN_CSIM)
                        ):
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_count}] ITEM switch_guard gid={gid_for_save} "
                                    f"item={int(best_track.get('item_id', -1)):03d} "
                                    f"{prev_label}->{cur_label} sim={best_track_sim:.3f} csim={best_track_csim:.3f} blocked"
                                )
                            best_track = None
                    if best_track is None:
                        if float(p.get("conf", 0.0)) < ITEM_NEW_MIN_CONF:
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_count}] ITEM pending_new skipped gid={gid_for_save} "
                                    f"label={p['label']} conf={p.get('conf', 0.0):.3f} "
                                    f"< {ITEM_NEW_MIN_CONF:.2f}"
                                )
                            continue
                        if not quality_ok:
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_count}] ITEM pending_new skipped gid={gid_for_save} "
                                    f"label={p['label']} reason=low_quality({quality_reason}) "
                                    f"lap={quality_sharp:.1f} tg={quality_tg:.1f}"
                                )
                            continue
                        # Force-create a new item when this held object is clearly far from
                        # all known item tracks for this customer.
                        force_new_far = (nearest_dist >= 0.0 and nearest_dist >= ITEM_NEW_FORCE_FAR_DIST)

                        # Debounce new item creation to avoid over-splitting from jitter.
                        pkey2 = (pcx2 // ITEM_PENDING_GRID, pcy2 // ITEM_PENDING_GRID)
                        pst = pending.setdefault(pkey2, {"last_frame": -10_000, "streak": 0, "box": list(pb)})
                        if frame_count - pst["last_frame"] <= max(2, FRAME_SKIP + 1):
                            pst["streak"] += 1
                        else:
                            pst["streak"] = 1
                        pst["last_frame"] = int(frame_count)
                        pst["box"] = list(pb)

                        if pst["streak"] >= ITEM_NEW_MIN_STREAK or force_new_far:
                            next_item_id_by_gid[gid_for_save] = next_item_id_by_gid.get(gid_for_save, 0) + 1
                            best_track = {
                                "item_id": int(next_item_id_by_gid[gid_for_save]),
                                "label": p["label"],
                                "label_votes": {p["label"]: 1},
                                "emb": p_emb,
                                "color_emb": p_cemb,
                                "anchor_emb": p_emb,
                                "anchor_color_emb": p_cemb,
                                "last_box": list(pb),
                                "last_frame": int(frame_count),
                            }
                            gtracks.append(best_track)
                            if det_tid >= 0:
                                trackid_map[det_tid] = int(best_track["item_id"])
                            new_item_created = True
                            pending.pop(pkey2, None)
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_count}] ITEM_NEW gid={gid_for_save} "
                                    f"item={best_track['item_id']:03d} "
                                    f"label={p['label']} conf={p.get('conf', 0.0):.3f} "
                                    f"reason={'far_from_existing' if force_new_far else 'no_match_or_low_similarity'} "
                                    f"(near_iou={nearest_iou:.3f}, near_dist={nearest_dist:.1f}, near_sim={nearest_sim:.3f}, near_csim={nearest_csim:.3f}) "
                                    f"th(iou={ITEM_MATCH_IOU_TH:.2f}, dist={ITEM_MATCH_MAX_DIST}, far_dist={ITEM_NEW_FORCE_FAR_DIST}, sim={DINO_MIN_SIM_FOR_MATCH:.2f}, csim={COLOR_MIN_SIM_FOR_MATCH:.2f})"
                                )
                        else:
                            if PRODUCT_DEBUG_LOG:
                                vlog(
                                    f"  [F{frame_count}] ITEM pending_new gid={gid_for_save} "
                                    f"label={p['label']} streak={pst['streak']}/{ITEM_NEW_MIN_STREAK}"
                                )
                            continue
                    else:
                        best_track["last_box"] = list(pb)
                        best_track["last_frame"] = int(frame_count)
                        anchor_sim = emb_cos_sim(p_emb, best_track.get("anchor_emb"))
                        anchor_csim = emb_cos_sim(p_cemb, best_track.get("anchor_color_emb"))
                        update_allowed = (
                            quality_ok
                            and best_track_sim >= ITEM_UPDATE_MIN_SIM
                            and best_track_csim >= ITEM_UPDATE_MIN_CSIM
                            and (anchor_sim < 0.0 or anchor_sim >= ITEM_ANCHOR_MIN_SIM)
                            and (anchor_csim < 0.0 or anchor_csim >= ITEM_ANCHOR_MIN_CSIM)
                        )
                        if PRODUCT_DEBUG_LOG:
                            vlog(
                                f"  [F{frame_count}] ITEM update_check gid={gid_for_save} "
                                f"item={int(best_track.get('item_id', -1)):03d} "
                                f"quality={quality_reason}:lap={quality_sharp:.1f}/tg={quality_tg:.1f} "
                                f"sim={best_track_sim:.3f} csim={best_track_csim:.3f} "
                                f"anchor_sim={anchor_sim:.3f} anchor_csim={anchor_csim:.3f} "
                                f"allowed={'Y' if update_allowed else 'N'}"
                            )
                        if update_allowed and p_emb is not None:
                            if best_track.get("emb") is None:
                                best_track["emb"] = p_emb
                            else:
                                best_track["emb"] = (0.85 * np.asarray(best_track["emb"]) + 0.15 * np.asarray(p_emb))
                        if update_allowed and p_cemb is not None:
                            if best_track.get("color_emb") is None:
                                best_track["color_emb"] = p_cemb
                            else:
                                best_track["color_emb"] = (0.85 * np.asarray(best_track["color_emb"]) + 0.15 * np.asarray(p_cemb))
                        votes = best_track.setdefault("label_votes", {})
                        votes[p["label"]] = votes.get(p["label"], 0) + 1
                        best_track["label"] = max(votes.items(), key=lambda kv: kv[1])[0]
                        if det_tid >= 0:
                            trackid_map[det_tid] = int(best_track["item_id"])
                    item_id = int(best_track["item_id"])
                    item_seen_with_customer_frame_by_gid.setdefault(gid_for_save, set()).add(item_id)

                    prev_state = best_track.get("state", "unknown")
                    if prev_state in ("unknown", "placed"):
                        ev_name = "ItemPicked" if prev_state == "unknown" else "ItemRepicked"
                        events.append({
                            "person_id": int(owner_gid),
                            "group_id": int(gid_for_save),
                            "event": ev_name,
                            "item_id": int(item_id),
                            "label": best_track.get("label", p.get("label", "item")),
                            "frame": int(frame_count),
                        })
                        vlog(f"  [F{frame_count}] EVENT gid={owner_gid} group={gid_for_save} {ev_name} item={item_id:03d}")
                    best_track["state"] = "with_customer"
                    best_track["last_with_frame"] = int(frame_count)
                    best_track["owner_gid"] = int(owner_gid)
                    best_track["exit_streak"] = 0
                    best_track["away_from_kiosk_streak"] = 0

                    # One item_id cannot be assigned to multiple product boxes in same frame.
                    used_ids = item_ids_used_this_frame_by_gid.setdefault(gid_for_save, set())
                    if item_id in used_ids:
                        continue
                    used_ids.add(item_id)
                    canonical_label = best_track.get("label", p["label"])
                    # Attach stable item_id to on-screen display memory tracks.
                    gid_disp = int(gid_for_save)
                    dtracks = display_item_tracks_by_gid.get(gid_disp, [])
                    for dt in dtracks:
                        if box_iou(dt.get("box", pb), pb) >= 0.4:
                            dt["item_id"] = item_id
                            dt["label"] = canonical_label
                    full_frame_path = None
                    crop_path = None
                    save_ok = quality_ok and float(p.get("conf", 0.0)) >= ITEM_SAVE_MIN_CONF
                    if new_item_created and save_ok:
                        full_frame_path = save_item_full_frame(
                            clean, p["box"], scanned_frames_dir, gid_for_save, item_id, vname, frame_count, canonical_label
                        )
                    if save_ok:
                        crop_path = save_product_crop(
                            clean, p["box"], scanned_items_dir, gid_for_save, item_id, vname, frame_count, canonical_label,
                            track_id=p.get("track_id", None)
                        )
                        scanned_count_by_gid[gid_for_save] = scanned_count_by_gid.get(gid_for_save, 0) + 1
                        type_counter = item_type_count_by_gid.setdefault(gid_for_save, {})
                        type_counter[canonical_label] = type_counter.get(canonical_label, 0) + 1
                    elif PRODUCT_DEBUG_LOG:
                        vlog(
                            f"  [F{frame_count}] ITEM save_skipped gid={gid_for_save} item={item_id:03d} "
                            f"reason=quality_or_conf quality={quality_reason} lap={quality_sharp:.1f} tg={quality_tg:.1f} "
                            f"conf={p.get('conf', 0.0):.3f} min_conf={ITEM_SAVE_MIN_CONF:.2f}"
                        )

                    if crop_path is not None:
                        best_track["last_image"] = crop_path
                    if full_frame_path is not None:
                        best_track["last_full_frame_image"] = full_frame_path

                    events.append({
                        "person_id": int(owner_gid),
                        "group_id": int(gid_for_save),
                        "event": "ItemHeld",
                        "kiosk": detect_kiosk_label(person_box=person_box_map.get(owner_gid), item_box=p["box"]),
                        "label": canonical_label,
                        "item_id": int(item_id),
                        "track_id": int(p.get("track_id", -1)),
                        "frame": int(frame_count),
                        "image": crop_path,
                        "full_frame_image": full_frame_path,
                    })
                    vlog(
                        f"  [F{frame_count}] ITEM_HELD gid={owner_gid} group={gid_for_save} item={item_id:03d} label={canonical_label} "
                        f"conf={p.get('conf', 0.0):.3f} streak={st['streak']} image={crop_path}"
                    )
                    record_group_evidence(
                        gid_for_save,
                        "item_held",
                        frame_count,
                        f"item_{item_id:03d} {canonical_label}",
                        primary_box=person_box_map.get(owner_gid),
                        secondary_box=p["box"],
                        person_id=int(owner_gid) if owner_gid is not None else None,
                        item_id=int(item_id),
                        label=str(canonical_label),
                    )
                    if new_item_created and full_frame_path is not None:
                        vlog(
                            f"  [F{frame_count}] ITEM_FRAME gid={gid_for_save} item={item_id:03d} image={full_frame_path}"
                        )
                    if PRODUCT_DEBUG_LOG and crop_path is None:
                        vlog(
                            f"  [F{frame_count}] PRODUCT save_failed "
                            f"track={p.get('track_id',-1)} cls={p['label']} reason=empty_crop"
                        )
                    st["last_saved_frame"] = frame_count

        # Item placement detection: if item was with customer but not seen as held
        # for a while, consider it placed.
        if not fast_qwen_evidence_mode:
            for gid, gtracks in item_tracks_by_gid.items():
                for it in gtracks:
                    state = str(it.get("state", "unknown"))
                    if state not in ("with_customer", "with_customer_hidden"):
                        continue
                    owner_gid = it.get("owner_gid")
                    owner_state = customer_state.get(int(owner_gid)) if owner_gid is not None else None
                    owner_box = owner_state.get("last_box") if owner_state is not None else None
                    owner_missing = int(owner_state.get("missing", 9999)) if owner_state is not None else 9999
                    owner_in_exit = (
                        owner_box is not None
                        and owner_missing <= ITEM_HIDDEN_MAX_OWNER_MISSING
                        and box_center_in_box(owner_box, exit_zone)
                    )
                    if owner_in_exit:
                        it["exit_streak"] = int(it.get("exit_streak", 0)) + 1
                    else:
                        it["exit_streak"] = 0
                    if int(it.get("exit_streak", 0)) >= ITEM_EXIT_CONFIRM_FRAMES:
                        it["state"] = "carried_out"
                        events.append({
                            "person_id": int(owner_gid) if owner_gid is not None else None,
                            "group_id": int(gid),
                            "event": "ItemCarriedOut",
                            "kiosk": detect_kiosk_label(person_box=owner_box, item_box=it.get("last_box")),
                            "item_id": int(it.get("item_id", -1)),
                            "label": it.get("label", "item"),
                            "frame": int(frame_count),
                        })
                        vlog(
                            f"  [F{frame_count}] EVENT gid={owner_gid} group={gid} "
                            f"ItemCarriedOut item={int(it.get('item_id', -1)):03d}"
                        )
                        record_group_evidence(
                            gid,
                            "carried_out",
                            frame_count,
                            f"item_{int(it.get('item_id', -1)):03d} {it.get('label', 'item')}",
                            primary_box=owner_box,
                            secondary_box=it.get("last_box"),
                            person_id=int(owner_gid) if owner_gid is not None else None,
                            item_id=int(it.get("item_id", -1)),
                            label=str(it.get("label", "item")),
                        )

            for gid, gtracks in item_tracks_by_gid.items():
                seen_ids = item_seen_with_customer_frame_by_gid.get(gid, set())
                for it in gtracks:
                    iid = int(it.get("item_id", -1))
                    if iid in seen_ids:
                        continue
                    if it.get("state") != "with_customer":
                        continue
                    last_with = int(it.get("last_with_frame", -10_000))
                    if frame_count - last_with >= ITEM_PLACE_GAP_FRAMES:
                        owner_gid = it.get("owner_gid")
                        owner_state = customer_state.get(int(owner_gid)) if owner_gid is not None else None
                        item_box = it.get("last_box")
                        hidden_on_customer = False
                        occluded_at_kiosk = False
                        if owner_state is not None and item_box is not None:
                            owner_missing = int(owner_state.get("missing", 9999))
                            owner_box = owner_state.get("last_box")
                            if owner_box is not None and owner_missing <= ITEM_HIDDEN_MAX_OWNER_MISSING:
                                hidden_zone = expand_xyxy(owner_box, ITEM_HIDDEN_BODY_PAD, new_w, new_h)
                                icx = int((item_box[0] + item_box[2]) * 0.5)
                                icy = int((item_box[1] + item_box[3]) * 0.5)
                                if kiosk_union is not None and box_center_in_box(item_box, kiosk_union) and box_center_in_box(owner_box, kiosk_union):
                                    occluded_at_kiosk = True
                                hidden_on_customer = point_in_box(icx, icy, hidden_zone)
                        if occluded_at_kiosk:
                            it["state"] = "occluded_at_kiosk"
                            it["occluded_frame"] = int(frame_count)
                            it["away_from_kiosk_streak"] = 0
                            events.append({
                                "person_id": int(owner_gid) if owner_gid is not None else None,
                                "group_id": int(gid),
                                "event": "ItemOccludedAtKiosk",
                                "kiosk": detect_kiosk_label(person_box=owner_box, item_box=item_box),
                                "item_id": int(iid),
                                "label": it.get("label", "item"),
                                "frame": int(frame_count),
                            })
                            vlog(
                                f"  [F{frame_count}] EVENT gid={owner_gid} group={gid} "
                                f"ItemOccludedAtKiosk item={iid:03d}"
                            )
                        elif hidden_on_customer:
                            it["state"] = "with_customer_hidden"
                            it["hidden_frame"] = int(frame_count)
                            it["exit_streak"] = 0
                            events.append({
                                "person_id": int(owner_gid) if owner_gid is not None else None,
                                "group_id": int(gid),
                                "event": "ItemHiddenOnCustomer",
                                "kiosk": detect_kiosk_label(person_box=owner_box, item_box=item_box),
                                "item_id": int(iid),
                                "label": it.get("label", "item"),
                                "frame": int(frame_count),
                            })
                            vlog(
                                f"  [F{frame_count}] EVENT gid={owner_gid} group={gid} "
                                f"ItemHiddenOnCustomer item={iid:03d}"
                            )
                            record_group_evidence(
                                gid,
                                "hidden",
                                frame_count,
                                f"item_{iid:03d} {it.get('label', 'item')} hidden",
                                primary_box=owner_box,
                                secondary_box=item_box,
                                person_id=int(owner_gid) if owner_gid is not None else None,
                                item_id=int(iid),
                                label=str(it.get("label", "item")),
                            )
                        else:
                            it["state"] = "placed"
                            events.append({
                                "person_id": None if GROUP_TRACKING_ENABLE else int(gid),
                                "group_id": int(gid),
                                "event": "ItemPlaced",
                                "item_id": int(iid),
                                "label": it.get("label", "item"),
                                "frame": int(frame_count),
                            })
                            vlog(f"  [F{frame_count}] EVENT group={gid} ItemPlaced item={iid:03d}")

            for gid, gtracks in item_tracks_by_gid.items():
                for it in gtracks:
                    state = str(it.get("state", "unknown"))
                    if state not in ("with_customer_hidden", "occluded_at_kiosk"):
                        continue
                    owner_gid = it.get("owner_gid")
                    owner_state = customer_state.get(int(owner_gid)) if owner_gid is not None else None
                    owner_box = owner_state.get("last_box") if owner_state is not None else None
                    owner_missing = int(owner_state.get("missing", 9999)) if owner_state is not None else 9999
                    owner_seen_recent = owner_box is not None and owner_missing <= ITEM_HIDDEN_MAX_OWNER_MISSING

                    owner_in_kiosk = owner_seen_recent and kiosk_union is not None and box_center_in_box(owner_box, kiosk_union)

                    if state == "occluded_at_kiosk":
                        if owner_in_kiosk:
                            it["away_from_kiosk_streak"] = 0
                        else:
                            it["away_from_kiosk_streak"] = int(it.get("away_from_kiosk_streak", 0)) + 1
                        if int(it.get("away_from_kiosk_streak", 0)) >= ITEM_OCCLUDED_PLACE_GAP_FRAMES:
                            it["state"] = "placed"
                            events.append({
                                "person_id": None if GROUP_TRACKING_ENABLE else int(gid),
                                "group_id": int(gid),
                                "event": "ItemPlacedAtKiosk",
                                "item_id": int(it.get("item_id", -1)),
                                "label": it.get("label", "item"),
                                "frame": int(frame_count),
                            })
                            vlog(
                                f"  [F{frame_count}] EVENT group={gid} "
                                f"ItemPlacedAtKiosk item={int(it.get('item_id', -1)):03d}"
                            )

        for ref_hit in reference_hit_boxes:
            bx1, by1, bx2, by2 = [int(v) for v in ref_hit["box"]]
            color = (0, 220, 120)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            if PRODUCT_OVERLAY_ENABLE:
                text = f"{ref_hit['label']} ref"
                cv2.putText(
                    frame,
                    text,
                    (bx1, max(18, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Draw remembered held-item rectangles (single persistent box, no stack).
        for gid, dtracks in list(display_item_tracks_by_gid.items()):
            kept = []
            for dt in dtracks:
                # Keep confirmed item boxes persistent at placed location.
                # They should only move when we detect the same item being held again.
                if dt.get("item_id") is not None:
                    dt["ttl"] = max(int(dt.get("ttl", 0)), ITEM_DISPLAY_MEMORY_FRAMES)
                elif gid in holding_display:
                    dt["ttl"] = max(int(dt.get("ttl", 0)), ITEM_DISPLAY_MEMORY_FRAMES)
                else:
                    dt["ttl"] = int(dt.get("ttl", 0)) - 1
                if dt["ttl"] <= 0:
                    continue
                # Simple de-dup: skip boxes that heavily overlap stronger ones.
                duplicate = False
                for k in kept:
                    if box_iou(dt["box"], k["box"]) >= 0.6:
                        if float(dt.get("conf", 0.0)) <= float(k.get("conf", 0.0)):
                            duplicate = True
                            break
                if not duplicate:
                    kept.append(dt)
            # Hard dedupe by item_id: at most one display track per (gid, item_id).
            by_item = {}
            for dt in kept:
                iid = dt.get("item_id", None)
                if iid is None:
                    key = ("none", id(dt))
                else:
                    key = ("item", int(iid))
                prev = by_item.get(key)
                if prev is None:
                    by_item[key] = dt
                else:
                    prev_score = (float(prev.get("conf", 0.0)), int(prev.get("last_frame", -1)))
                    curr_score = (float(dt.get("conf", 0.0)), int(dt.get("last_frame", -1)))
                    if curr_score > prev_score:
                        by_item[key] = dt
            kept = list(by_item.values())
            display_item_tracks_by_gid[gid] = kept
            for dt in kept:
                if dt.get("item_id") is None:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in dt["box"]]
                color = (0, 255, 255)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                if PRODUCT_OVERLAY_ENABLE:
                    item_txt = f" item_{int(dt['item_id']):03d}"
                    owner_text = f"Group_{gid}" if GROUP_TRACKING_ENABLE else f"ID{gid}"
                    text = f"{dt.get('label','item')}{item_txt} {owner_text}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    tx = max(0, bx2 - tw)
                    ty = max(th + 2, by1 - 6)
                    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Fade holding overlays.
        for gid in list(holding_display.keys()):
            holding_display[gid]["ttl"] -= 1
            if holding_display[gid]["ttl"] <= 0:
                del holding_display[gid]

        # Handle disappeared tracks and interaction end.
        for gid, mem in list(track_memory.items()):
            if gid in current_gids:
                continue
            mem["missing"] += 1

            if use_kiosk_zone_logic and mem.get("active_kiosk"):
                mem["missing_interact"] = mem.get("missing_interact", 0) + 1
                if mem["missing_interact"] >= INTERACT_END_MISSING_FRAMES:
                    events.append({
                        "person_id": int(gid),
                        "event": "InteractEnd",
                        "kiosk": mem["active_kiosk"],
                        "frame": int(frame_count),
                    })
                    vlog(f"  [F{frame_count}] EVENT gid={gid} InteractEnd {mem['active_kiosk']}")
                    mem["active_kiosk"] = None
                    mem["missing_interact"] = 0

            if mem["missing"] == 1 and mem.get("views"):
                lost_gallery[gid] = {
                    "center": (mem["last_cx"], mem["last_cy"]),
                    "views": mem["views"].copy(),
                    "snapshot": mem.copy(),
                }

        writer.write(frame)

    frame_loop_elapsed = time.time() - t0
    cap.release()
    writer.release()
    shutil.copy2(tmp_vid, out_vid)
    os.remove(tmp_vid)

    # Entry videos seed the persistent gallery. Kiosk videos may refine only
    # already-known Entry IDs so later re-matches use fresher views.
    for gid, mem in track_memory.items():
        if mem.get("seen", 0) < MIN_SEEN_FOR_GALLERY:
            continue
        if use_kiosk_zone_logic and gid not in persistent_gallery:
            continue
        if gid not in persistent_gallery:
            persistent_gallery[gid] = {"views": []}
        for v in mem.get("views", []):
            persistent_gallery[gid]["views"] = ema_update_persistent_gallery_views(
                persistent_gallery[gid]["views"], v
            )

    group_summary_payloads = []
    kiosk_output_events = []
    qwen_summary_elapsed = 0.0
    if use_kiosk_zone_logic:
        for gid in sorted(item_tracks_by_gid):
            carried_items = [
                it for it in item_tracks_by_gid.get(gid, [])
                if item_counts_as_with_customer(it.get("state"))
            ]
            if not carried_items:
                continue
            carried_items = sorted(
                carried_items,
                key=lambda it: (int(it.get("item_id", 10**9)), str(it.get("label", ""))),
            )
            item_list = []
            for it in carried_items:
                item_list.append({
                    "item_id": int(it.get("item_id", -1)),
                    "label": str(it.get("label", "item")),
                    "state": str(it.get("state", "unknown")),
                    "last_frame": int(it.get("last_frame", frame_count)),
                    "image": it.get("last_image"),
                    "full_frame_image": it.get("last_full_frame_image"),
                })
            events.append({
                "person_id": None if GROUP_TRACKING_ENABLE else int(gid),
                "group_id": int(gid),
                "event": "VideoEndCarryOutSummary",
                "frame": int(frame_count),
                "item_count": int(len(item_list)),
                "items": item_list,
            })
            vlog(
                f"  [F{frame_count}] VIDEO_END group={gid} "
                f"assume_exit items_carried_out={len(item_list)}"
            )
            for item in item_list:
                vlog(
                    f"    carry_out item={item['item_id']:03d} "
                    f"label={item['label']} "
                    f"state={item['state']} "
                    f"last_frame={item['last_frame']} "
                    f"image={item['image']}"
                )

        all_group_ids = sorted(
            set(group_members_by_gid.keys())
            | set(item_tracks_by_gid.keys())
            | set(group_evidence_records.keys())
        )
        for gid in all_group_ids:
            member_ids = sorted(group_members_by_gid.get(gid, set()))
            evidence_records = sorted(
                group_evidence_records.get(gid, []),
                key=lambda rec: (int(rec.get("frame", -1)), str(rec.get("tag", ""))),
            )
            selected_records = select_diverse_evidence_records(
                evidence_records,
                fps,
                GROUP_EVIDENCE_MAX_IMAGES if GROUP_EVIDENCE_MAX_IMAGES > 0 else None,
            )
            evidence_paths = [
                rec.get("image")
                for rec in selected_records
                if rec.get("image") and os.path.isfile(rec.get("image"))
            ]
            llm_input_paths = []
            if GROUP_EVIDENCE_ENABLE:
                group_dir = os.path.join(group_evidence_dir, f"group_{int(gid):03d}")
                os.makedirs(group_dir, exist_ok=True)
                llm_input_dir = os.path.join(group_dir, GROUP_LLM_INPUT_DIRNAME)
                os.makedirs(llm_input_dir, exist_ok=True)
                for name in os.listdir(llm_input_dir):
                    path = os.path.join(llm_input_dir, name)
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except FileNotFoundError:
                            pass
                for idx, src_path in enumerate(evidence_paths, start=1):
                    if not src_path or not os.path.isfile(src_path):
                        continue
                    dst_name = f"{idx:02d}_{os.path.basename(src_path)}"
                    dst_path = os.path.join(llm_input_dir, dst_name)
                    shutil.copy2(src_path, dst_path)
                    llm_input_paths.append(dst_path)
            timeline = [
                ev for ev in events
                if int(ev.get("group_id", -999999)) == int(gid)
            ]
            prompt = build_group_vlm_prompt(gid, member_ids, timeline)
            if not llm_input_paths and not evidence_paths:
                summary = build_empty_group_summary(gid)
                raw_or_status = build_vlm_response_meta(
                    *current_vlm_provider_and_model(),
                    raw_usage=None,
                    raw_text="no_evidence_images",
                    status="no_evidence_images",
                )
                vlog(
                    f"  GROUP_SUMMARY group={gid} members={member_ids or ['unknown']} "
                    f"evidence=0/{len(evidence_records)} vlm_status=no_evidence_images"
                )
            else:
                images_for_vlm = llm_input_paths if llm_input_paths else evidence_paths
                qwen_t0 = time.time()
                summary, raw_or_status = run_group_vlm_summary(
                    images_for_vlm,
                    prompt,
                )
                qwen_summary_elapsed += time.time() - qwen_t0
            kiosk_group_event = build_kiosk_group_event(
                vname,
                gid,
                summary,
                member_ids,
                event_timeline=timeline,
                fps=fps,
            )
            payload = {
                "video": vname,
                "group_id": int(gid),
                "kiosk_event_summary": make_json_safe(kiosk_group_event),
                "vlm_result": make_json_safe(summary),
                "vlm_meta": make_json_safe(raw_or_status),
            }
            group_summary_payloads.append(payload)
            kiosk_output_events.append(kiosk_group_event)
            if GROUP_EVIDENCE_ENABLE:
                payload["llm_input_images"] = llm_input_paths
                raw_text_path = write_vlm_raw_text(group_dir, "", raw_or_status)
                if raw_text_path:
                    payload["vlm_raw_text_path"] = raw_text_path
                with open(os.path.join(group_dir, "summary.json"), "w") as gf:
                    json.dump(payload, gf, indent=2)
            else:
                payload["llm_input_images"] = llm_input_paths
            if llm_input_paths or evidence_paths:
                vlog(
                    f"  GROUP_SUMMARY group={gid} members={member_ids or ['unknown']} "
                    f"evidence={len(evidence_paths)}/{len(evidence_records)} vlm_status=ok"
                )

    cross_state["next_gid"] = next_gid

    output_events = kiosk_output_events if use_kiosk_zone_logic else events
    with open(out_json, "w") as f:
        json.dump(make_json_safe(output_events), f, indent=2)
    tracking_summary = {
        "video": vname,
        "persistent_gallery_ids": sorted(int(gid) for gid in persistent_gallery.keys()),
        "customers": [
            {
                "person_id": int(gid),
                "source": str(cs.get("source", track_memory.get(gid, {}).get("source", "unknown"))),
                "entered": bool(cs.get("entered", False)),
                "exited": bool(cs.get("exited", False)),
                "entry_frame": cs.get("entry_frame"),
                "exit_frame": cs.get("exit_frame"),
                "last_seen_frame": cs.get("last_seen_frame"),
                "group_id": int(group_id_for_person(gid)),
            }
            for gid, cs in sorted(customer_state.items(), key=lambda kv: int(kv[0]))
        ],
    }
    with open(out_tracking_json, "w") as f:
        json.dump(make_json_safe(tracking_summary), f, indent=2)
    if group_summary_payloads:
        out_group_json = os.path.join(output_dir, f"{vname}_group_summaries.json")
        with open(out_group_json, "w") as gf:
            json.dump(group_summary_payloads, gf, indent=2)
    if scanned_count_by_gid:
        for gid in sorted(scanned_count_by_gid):
            vlog(f"  SUMMARY scanned_items gid={gid} total={scanned_count_by_gid[gid]}")
            if gid in item_type_count_by_gid:
                parts = [f"{k}:{v}" for k, v in sorted(item_type_count_by_gid[gid].items())]
                vlog(f"  SUMMARY item_types gid={gid} " + ", ".join(parts))
    total_elapsed = time.time() - process_t0
    post_elapsed = max(0.0, total_elapsed - frame_loop_elapsed)
    vlog(
        f"Done — {len(events)} events, {frame_count} frames, {total_elapsed:.1f}s "
        f"(frame_loop={frame_loop_elapsed:.1f}s, qwen={qwen_summary_elapsed:.1f}s, post={post_elapsed:.1f}s)"
    )
    for cust in tracking_summary["customers"]:
        vlog(
            f"  TRACKING gid={cust['person_id']} source={cust['source']} "
            f"entered={cust['entered']} exited={cust['exited']} "
            f"entry_frame={cust['entry_frame']} exit_frame={cust['exit_frame']}"
        )
    return events


def main():
    def _process_one_session(entry_input_dir, kiosk_input_dir, runtime_output_base):
        print(f"[Detect] entry_input={entry_input_dir}")
        print(f"[Detect] kiosk_input={kiosk_input_dir}")
        print(f"[Detect] output_root={runtime_output_base}")
        print(f"[Detect] clear_output={'yes' if DETECT_CLEAR_OUTPUT else 'no'}")

        os.makedirs(runtime_output_base, exist_ok=True)
        logs_output_dir = os.path.join(runtime_output_base, LOGS_OUTPUT_DIRNAME)
        video_output_dir = os.path.join(runtime_output_base, VIDEO_OUTPUT_DIRNAME)
        reid_views_output_dir = os.path.join(logs_output_dir, "reid_views")
        reid_fashion_views_output_dir = os.path.join(logs_output_dir, "reid_fashion_views")
        os.makedirs(logs_output_dir, exist_ok=True)
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(reid_views_output_dir, exist_ok=True)
        os.makedirs(reid_fashion_views_output_dir, exist_ok=True)

        all_session_videos = find_videos(kiosk_input_dir)
        entry_candidates = IntegratedEntry.find_videos(entry_input_dir)

        ordered_videos_map = {}
        for vp in entry_candidates + all_session_videos:
            ordered_videos_map[os.path.abspath(vp)] = vp
        ordered_videos = sorted(
            ordered_videos_map.values(),
            key=lambda vp: (_leading_number(vp), os.path.basename(vp).upper()),
        )

        if not ordered_videos:
            print(
                f" No videos found. entry={entry_input_dir} kiosk={kiosk_input_dir}"
            )
            return {}

        if should_rerun_kiosk_llm_only(runtime_output_base, ordered_videos):
            print(
                f"[Detect] {current_vlm_result_filename()} already exists, rerunning all kiosk videos "
                "LLM input to refresh total item result."
            )
            session_events = rerun_existing_kiosk_llm_only(runtime_output_base, ordered_videos)
            if session_events is not None:
                run_additional_gemini_model_comparisons(
                    runtime_output_base,
                    ordered_videos,
                    session_events,
                    GROUP_GEMINI_MODEL,
                )
                return session_events
            print(
                "[Detect] kiosk-llm-only aborted. Falling back to full processing "
                "because some kiosk videos have no existing group_001 evidence."
            )

        cross_state = {
            "next_gid": 1,
            "persistent_gallery": {},
            "persistent_gallery_view_paths": {},
        }
        all_events = {}
        global REID_FASHION_DEBUG_DIR
        REID_FASHION_DEBUG_DIR = reid_fashion_views_output_dir

        for vp in ordered_videos:
            vname = os.path.splitext(os.path.basename(vp))[0]
            video_kind = _video_kind(vp)
            if video_kind == "kiosk":
                kiosk_output_dir = os.path.join(logs_output_dir, vname)
                os.makedirs(kiosk_output_dir, exist_ok=True)
                out_log = os.path.join(
                    kiosk_output_dir,
                    f"{vname}_log.txt",
                )
                try:
                    if os.path.basename(vp).lower().startswith("sample"):
                        print(f"[Sample-Kiosk] {vp}")
                        ev = process_kiosk_video(vp, kiosk_output_dir, cross_state)
                    else:
                        print(f"[Kiosk] {vp}")
                        ev = process_kiosk_video(vp, kiosk_output_dir, cross_state)
                except Exception as e:
                    err_text = (
                        f"\n[FATAL] video={vp}\n"
                        f"error={e}\n"
                        f"{traceback.format_exc()}"
                    )
                    print(err_text)
                    with open(out_log, "a") as lf:
                        lf.write(err_text)
                        if not err_text.endswith("\n"):
                            lf.write("\n")
                    raise
                move_finished_video(kiosk_output_dir, video_output_dir, vp)
                if ev:
                    all_events[os.path.basename(vp)] = ev
                continue

            entry_output_dir = os.path.join(logs_output_dir, vname)
            os.makedirs(entry_output_dir, exist_ok=True)
            IntegratedEntry.OUTPUT_BASE = entry_output_dir
            IntegratedEntry.CROPS_DIR = None
            IntegratedEntry.REID_DEBUG_DIR = reid_views_output_dir
            IntegratedEntry.REID_FASHION_DEBUG_DIR = reid_fashion_views_output_dir
            os.makedirs(IntegratedEntry.REID_DEBUG_DIR, exist_ok=True)
            print(f"[Entry] {vp}")
            ev = IntegratedEntry.process_video(vp, entry_output_dir, cross_state)
            move_finished_video(entry_output_dir, video_output_dir, vp)
            if ev:
                all_events[os.path.basename(vp)] = ev

        with open(os.path.join(runtime_output_base, "all_events_summary.json"), "w") as f:
            json.dump(all_events, f, indent=2)
        session_result = build_session_result(runtime_output_base, ordered_videos, all_events)
        result_payload = make_json_safe(session_result)
        with open(current_vlm_result_path(runtime_output_base), "w") as f:
            json.dump(result_payload, f, indent=2)
        with open(os.path.join(logs_output_dir, "result.json"), "w") as f:
            json.dump(result_payload, f, indent=2)
        run_additional_gemini_model_comparisons(
            runtime_output_base,
            ordered_videos,
            all_events,
            GROUP_GEMINI_MODEL,
        )
        print(
            f"Detect processing complete. videos={len(ordered_videos)} output={runtime_output_base}"
        )
        return all_events

    explicit_entry = "DETECT_ENTRY_VIDEO_FOLDER" in os.environ
    explicit_kiosk = "DETECT_KIOSK_VIDEO_FOLDER" in os.environ
    explicit_output = "DETECT_OUTPUT_BASE" in os.environ

    if explicit_entry or explicit_kiosk:
        runtime_output_base = OUTPUT_BASE
        if not explicit_output:
            entry_dir_abs = os.path.abspath(ENTRY_VIDEO_FOLDER)
            kiosk_dir_abs = os.path.abspath(VIDEO_FOLDER)
            if os.path.isdir(entry_dir_abs) and os.path.basename(entry_dir_abs).lower().startswith("session_"):
                runtime_output_base = os.path.join(entry_dir_abs, "output")
            elif os.path.isdir(kiosk_dir_abs) and os.path.basename(kiosk_dir_abs).lower().startswith("session_"):
                runtime_output_base = os.path.join(kiosk_dir_abs, "output")
        if DETECT_CLEAR_OUTPUT and not session_dirs:
            print(f"[Detect] upfront clear_output=yes target={runtime_output_base}")
            os.makedirs(runtime_output_base, exist_ok=True)
            if should_rerun_kiosk_llm_only(runtime_output_base):
                print(
                    f"[Detect] skip clear because {current_vlm_result_filename()} exists "
                    f"target={runtime_output_base}"
                )
            else:
                clear_output_folder(runtime_output_base)
        _process_one_session(ENTRY_VIDEO_FOLDER, VIDEO_FOLDER, runtime_output_base)
        return

    session_dirs = find_session_dirs(DEFAULT_SESSIONS_ROOT)

    if DETECT_CLEAR_OUTPUT and session_dirs:
        print(f"[Detect] sessions_root={DEFAULT_SESSIONS_ROOT}")
        print(f"[Detect] upfront clear all session outputs count={len(session_dirs)}")
        for session_dir in session_dirs:
            runtime_output_base = os.path.join(session_dir, "output")
            print(f"[Detect] upfront clear_output=yes target={runtime_output_base}")
            os.makedirs(runtime_output_base, exist_ok=True)
            if should_rerun_kiosk_llm_only(runtime_output_base):
                print(
                    f"[Detect] skip clear because {current_vlm_result_filename()} exists "
                    f"target={runtime_output_base}"
                )
            else:
                clear_output_folder(runtime_output_base)

    if not session_dirs:
        runtime_output_base = OUTPUT_BASE
        if not explicit_output:
            entry_dir_abs = os.path.abspath(ENTRY_VIDEO_FOLDER)
            kiosk_dir_abs = os.path.abspath(VIDEO_FOLDER)
            if os.path.isdir(entry_dir_abs) and os.path.basename(entry_dir_abs).lower().startswith("session_"):
                runtime_output_base = os.path.join(entry_dir_abs, "output")
            elif os.path.isdir(kiosk_dir_abs) and os.path.basename(kiosk_dir_abs).lower().startswith("session_"):
                runtime_output_base = os.path.join(kiosk_dir_abs, "output")
        if DETECT_CLEAR_OUTPUT:
            print(f"[Detect] upfront clear_output=yes target={runtime_output_base}")
            os.makedirs(runtime_output_base, exist_ok=True)
            if should_rerun_kiosk_llm_only(runtime_output_base):
                print(
                    f"[Detect] skip clear because {current_vlm_result_filename()} exists "
                    f"target={runtime_output_base}"
                )
            else:
                clear_output_folder(runtime_output_base)
        _process_one_session(ENTRY_VIDEO_FOLDER, VIDEO_FOLDER, runtime_output_base)
        return

    all_sessions_summary = {}
    print(f"[Detect] sessions_root={DEFAULT_SESSIONS_ROOT}")
    print(f"[Detect] found_sessions={len(session_dirs)}")
    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        runtime_output_base = os.path.join(session_dir, "output")
        session_events = _process_one_session(session_dir, session_dir, runtime_output_base)
        result_payload = load_json_file(current_vlm_result_path(runtime_output_base), default={}) or {}
        all_sessions_summary[session_name] = {
            "output": runtime_output_base,
            "videos_with_events": sorted(session_events.keys()),
            "event_count": int(sum(len(v) for v in session_events.values())),
            "result_file": current_vlm_result_filename(),
            "vlm_usage_summary": result_payload.get("vlm_usage_summary"),
        }
        if DETECT_INTER_SESSION_SLEEP_SEC > 0 and session_dir != session_dirs[-1]:
            print(
                f"[Detect] inter-session pause {DETECT_INTER_SESSION_SLEEP_SEC}s "
                f"after {session_name}"
            )
            time.sleep(DETECT_INTER_SESSION_SLEEP_SEC)

    summary_path = os.path.join(DEFAULT_SESSIONS_ROOT, "all_sessions_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_sessions_summary, f, indent=2)
    print(f"[Detect] all sessions complete. summary={summary_path}")
    overall_answer_compare = build_overall_session_answer_comparison(
        DEFAULT_SESSIONS_ROOT,
        session_dirs,
        os.path.join(DEFAULT_SESSIONS_ROOT, "answer.txt"),
    )
    if overall_answer_compare is None:
        print(
            f"[Detect] overall answer comparison skipped. "
            f"No valid answer file at {os.path.join(DEFAULT_SESSIONS_ROOT, 'answer.txt')}"
        )
    else:
        print(
            f"[Detect] overall answer comparison complete -> "
            f"{overall_answer_compare['output_path']}"
        )


if __name__ == "__main__":
    main()
