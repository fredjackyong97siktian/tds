import cv2
import numpy as np
from ultralytics import YOLO
import json, os, time, shutil, glob
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchreid
from PIL import Image
try:
    import timm
except ImportError:
    timm = None
try:
    from fashion_clip.fashion_clip import FashionCLIP
except ImportError:
    FashionCLIP = None
##ISSUE NOW - video 
# [F366] EVENT gid=4 type=Exit image=/Users/fredjackyong/Documents/kebunapp/theft_detection/Output/crops/ID4/5_F000365.jpg shouldn't be Exit
# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_FOLDER = os.environ.get("VIDEO_FOLDER", os.path.join(BASE_DIR, "new_video"))
OUTPUT_BASE  = os.environ.get("OUTPUT_BASE", os.path.join(BASE_DIR, "Output"))
CROPS_DIR    = os.environ.get("CROPS_DIR", os.path.join(OUTPUT_BASE, "crops"))
REID_DEBUG_DIR = os.environ.get("REID_DEBUG_DIR", os.path.join(OUTPUT_BASE, "reid_views"))
MODEL_PATH   = os.environ.get("MODEL_PATH", (os.path.join(BASE_DIR, "models", "detection", "yolo26s.pt") if os.path.exists(os.path.join(BASE_DIR, "models", "detection", "yolo26s.pt")) else os.path.join(BASE_DIR, "yolo26s.pt")))
TRACKER_PATH = os.environ.get("TRACKER_PATH", (os.path.join(BASE_DIR, "models", "tracker", "custom_tracker.yaml") if os.path.exists(os.path.join(BASE_DIR, "models", "tracker", "custom_tracker.yaml")) else os.path.join(BASE_DIR, "custom_tracker.yaml")))
REID_MASK_MODEL_PATH = os.environ.get("REID_MASK_MODEL_PATH", (os.path.join(BASE_DIR, "models", "segmentation", "yoloe-11l-seg.pt") if os.path.exists(os.path.join(BASE_DIR, "models", "segmentation", "yoloe-11l-seg.pt")) else os.path.join(BASE_DIR, "yoloe-11l-seg.pt")))
REID_USE_SEG_MASK = os.environ.get("REID_USE_SEG_MASK", "1") == "1"
REID_SEG_STRICT = os.environ.get("REID_SEG_STRICT", "1") == "1"
MAX_FRAMES_PER_VIDEO = int(os.environ.get("MAX_FRAMES_PER_VIDEO", "0"))

FRAME_SKIP   = 1
RESIZE_WIDTH = 960
MIN_BOX_HEIGHT = 150
MIN_BOX_AREA   = 2200
MIN_VISIBLE_RATIO = float(os.environ.get("MIN_VISIBLE_RATIO", "0.70"))

ZONE_LEFT        = 580
ZONE_RIGHT       = 750
ZONE_TOP_OFFSET  = 450
ZONE_BOTTOM_OFFSET = 100

# ===== REID =====
REID_SIM_THRESHOLD = 0.40   # within-video match if ANY gallery view hits this
REID_MAX_GAP       = 300    # frames in gallery (~13s at 22fps)
SPATIAL_GATE       = 200    # pixels
GALLERY_EMA_ALPHA  = 0.15   # slow update — preserve original views
MIN_SEEN_FOR_GALLERY = 8    # clean frames before saving to gallery

# Multi-view gallery settings
MAX_VIEWS_PER_PERSON = 6    # max distinct embeddings stored per person
VIEW_DIVERSITY_THRESH = 0.75 # only add a new view if it's dissimilar enough
                              # from existing views (lower = more views stored)

OCCLUSION_IOU_THRESHOLD = 0.15
EVENT_DISPLAY_TTL = 20
DISAPPEAR_EDGE_MARGIN = int(os.environ.get("DISAPPEAR_EDGE_MARGIN", "35"))
EXIT_LEFT_MARGIN = int(os.environ.get("EXIT_LEFT_MARGIN", "20"))
EVENT_BOX_AREA_RATIO_MAX = float(os.environ.get("EVENT_BOX_AREA_RATIO_MAX", "1.80"))
EVENT_BOX_AREA_RATIO_MIN = float(os.environ.get("EVENT_BOX_AREA_RATIO_MIN", "0.55"))
EVENT_MAX_FOOT_JUMP = int(os.environ.get("EVENT_MAX_FOOT_JUMP", "90"))
EVENT_HISTORY_LEN = int(os.environ.get("EVENT_HISTORY_LEN", "8"))
ZONE_OVERLAP_MIN_RATIO = float(os.environ.get("ZONE_OVERLAP_MIN_RATIO", "0.15"))
EVENT_DEBUG = os.environ.get("EVENT_DEBUG", "0") == "1"
MIN_EVENT_DET_CONF = float(os.environ.get("MIN_EVENT_DET_CONF", "0.5"))
MIN_EXIT_ZONE_SAMPLES = int(os.environ.get("MIN_EXIT_ZONE_SAMPLES", "4"))
MIN_EXIT_VISIBLE_RATIO = float(os.environ.get("MIN_EXIT_VISIBLE_RATIO", "0.8"))
MIN_EXIT_BOX_WIDTH = int(os.environ.get("MIN_EXIT_BOX_WIDTH", "55"))
MIN_EXIT_BOX_HEIGHT = int(os.environ.get("MIN_EXIT_BOX_HEIGHT", "110"))
EXIT_CENTER_MARGIN_X = int(os.environ.get("EXIT_CENTER_MARGIN_X", "20"))
EXIT_CENTER_MARGIN_Y = int(os.environ.get("EXIT_CENTER_MARGIN_Y", "12"))
MIN_CARRYOVER_SEEN_FRAMES = int(os.environ.get("MIN_CARRYOVER_SEEN_FRAMES", "4"))
REID_SCORE_DEBUG = os.environ.get("REID_SCORE_DEBUG", "1") == "1"
REID_WITHIN_VIDEO_DEBUG = os.environ.get("REID_WITHIN_VIDEO_DEBUG", "0") == "1"
WITHIN_VIDEO_VISIBILITY_LOG = os.environ.get("WITHIN_VIDEO_VISIBILITY_LOG", "1") == "1"
REID_UPDATE_VIEW_LOG = os.environ.get("REID_UPDATE_VIEW_LOG", "1") == "1"
REID_UPDATE_VIEW_MIN_SIM = float(os.environ.get("REID_UPDATE_VIEW_MIN_SIM", "0.68"))
REID_UPDATE_VIEW_MIN_OS = float(os.environ.get("REID_UPDATE_VIEW_MIN_OS", "0.70"))
REID_UPDATE_VIEW_MIN_FC = 0.60
LOST_RELINK_MIN_STREAK = int(os.environ.get("LOST_RELINK_MIN_STREAK", "4"))
SAVE_REID_DEBUG_CROPS = os.environ.get("SAVE_REID_DEBUG_CROPS", "1") == "1"
ID_BOOK_RELEASE_MISSING = int(os.environ.get("ID_BOOK_RELEASE_MISSING", "12"))

# Cross-video Re-ID. Same person across multiple videos should keep the same
# global ID. Stricter than within-video threshold and uses an ambiguity gap.
CROSS_VIDEO_REID = os.environ.get("CROSS_VIDEO_REID", "1") == "1"
CROSS_VIDEO_REID_SIM_THRESHOLD = float(os.environ.get("CROSS_VIDEO_REID_SIM_THRESHOLD", "0.60"))
CROSS_VIDEO_MIN_VISIBLE_RATIO = float(os.environ.get("CROSS_VIDEO_MIN_VISIBLE_RATIO", "0.70"))
CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1 = os.environ.get("CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1", "1") == "0"
CROSS_VIDEO_DELAY_ON_AMBIGUOUS = os.environ.get("CROSS_VIDEO_DELAY_ON_AMBIGUOUS", "1") == "1"
CROSS_VIDEO_MIN_STREAK = int(os.environ.get("CROSS_VIDEO_MIN_STREAK", "4")) #ISSUE that cause the inconsistency of the ID2
NEW_ID_AGG_FRAMES = int(os.environ.get("NEW_ID_AGG_FRAMES", "1"))
NEW_ID_PERSIST_SIM_THRESHOLD = float(
    os.environ.get("NEW_ID_PERSIST_SIM_THRESHOLD", str(CROSS_VIDEO_REID_SIM_THRESHOLD + 0.10))
)

# Re-ID model setup. Default to local OSNet checkpoint.
REID_MODEL_NAME = os.environ.get("REID_MODEL_NAME", "osnet_x1_0")
DEFAULT_REID_WEIGHTS_PATH = (
    (os.path.join(BASE_DIR, "models", "reid", "osnet_x1_0_msmt17.pt") if os.path.exists(os.path.join(BASE_DIR, "models", "reid", "osnet_x1_0_msmt17.pt")) else os.path.join(BASE_DIR, "models", "osnet_x1_0_msmt17.pt"))
    if REID_MODEL_NAME.startswith("osnet")
    else ""
)
REID_WEIGHTS_PATH = os.environ.get(
    "REID_WEIGHTS_PATH",
    DEFAULT_REID_WEIGHTS_PATH,
)
REID_INPUT_HEIGHT = int(os.environ.get("REID_INPUT_HEIGHT", "256"))
REID_INPUT_WIDTH = int(os.environ.get("REID_INPUT_WIDTH", "128"))
REID_TIMM_PRETRAINED = os.environ.get("REID_TIMM_PRETRAINED", "0") == "1"
FASHIONCLIP_ENABLE = os.environ.get("FASHIONCLIP_ENABLE", "1") == "1"
FASHIONCLIP_MODEL_NAME = os.environ.get("FASHIONCLIP_MODEL_NAME", "fashion-clip")
FASHIONCLIP_UPPER_WEIGHT = float(os.environ.get("FASHIONCLIP_UPPER_WEIGHT", "0.8"))
FASHIONCLIP_LOWER_WEIGHT = float(os.environ.get("FASHIONCLIP_LOWER_WEIGHT", "0.2"))
FASHIONCLIP_WITHIN_ALPHA = float(os.environ.get("FASHIONCLIP_WITHIN_ALPHA", "0.20"))
FASHIONCLIP_CROSS_ALPHA = float(os.environ.get("FASHIONCLIP_CROSS_ALPHA", "0.25"))

# ==============================
# DEVICE + MODEL
# ==============================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

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
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(key)
            continue
        filtered[key] = value
    return filtered, skipped


class TimmEmbeddingModel(torch.nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = getattr(self.backbone, "num_features", None)

    def forward(self, x):
        return self.backbone(x)


def load_reid_model():
    print(f"Loading Re-ID model: {REID_MODEL_NAME}")
    using_timm = timm is not None and REID_MODEL_NAME in timm.list_models()

    if using_timm:
        model = TimmEmbeddingModel(
            REID_MODEL_NAME,
            pretrained=REID_TIMM_PRETRAINED and not REID_WEIGHTS_PATH,
        )
    else:
        model = torchreid.models.build_model(
            name=REID_MODEL_NAME,
            num_classes=1000,
            pretrained=False,
        )

    if REID_WEIGHTS_PATH and os.path.exists(REID_WEIGHTS_PATH):
        try:
            checkpoint = torch.load(REID_WEIGHTS_PATH, map_location="cpu", weights_only=False)
        except UnicodeDecodeError:
            # Legacy torchreid checkpoints (e.g. resnet50_msmt17) were pickled
            # under Python 2 and need explicit latin1 encoding to unpickle.
            checkpoint = torch.load(REID_WEIGHTS_PATH, map_location="cpu",
                                    weights_only=False, encoding="latin1")
        state_dict = _extract_state_dict(checkpoint)
        state_dict, skipped = _filter_incompatible_keys(model, state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Re-ID weights from {REID_WEIGHTS_PATH}")
        if skipped:
            print(f"  Skipped incompatible keys: {len(skipped)}")
        if missing:
            print(f"  Missing keys while loading: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys while loading: {len(unexpected)}")
    else:
        print("WARNING: No local Re-ID checkpoint found.")
        if using_timm and REID_TIMM_PRETRAINED:
            print("         timm pretrained weights were requested, but could not be loaded locally.")
        print("         Using randomly initialized weights will make re-identification unreliable.")
        print(f"         Expected checkpoint path: {REID_WEIGHTS_PATH}")

    model = model.to(DEVICE)
    model.eval()
    print(f"Re-ID ready (feature_dim={getattr(model, 'feature_dim', 'unknown')})")
    return model


reid_model = load_reid_model()

REID_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((REID_INPUT_HEIGHT, REID_INPUT_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

os.makedirs(OUTPUT_BASE, exist_ok=True)
yolo_model = YOLO(MODEL_PATH)
reid_mask_model = None
if REID_USE_SEG_MASK:
    if os.path.exists(REID_MASK_MODEL_PATH):
        try:
            reid_mask_model = YOLO(REID_MASK_MODEL_PATH)
            try:
                reid_mask_model.set_classes(["person"])
                print("ReID segmentation classes set: ['person']")
            except Exception as e:
                print(f"Warning: cannot set segmentation classes to person: {e}")
            print(f"ReID segmentation ready: {REID_MASK_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: cannot load ReID segmentation model {REID_MASK_MODEL_PATH}: {e}")
    else:
        print(f"Warning: ReID segmentation model not found: {REID_MASK_MODEL_PATH}")
fashionclip_model = None
if FASHIONCLIP_ENABLE:
    if FashionCLIP is None:
        print("Warning: fashion_clip is not installed. FashionCLIP fusion disabled.")
    else:
        try:
            fashionclip_model = FashionCLIP(FASHIONCLIP_MODEL_NAME)
            print(f"FashionCLIP ready: {FASHIONCLIP_MODEL_NAME}")
        except Exception as e:
            print(f"Warning: cannot load FashionCLIP model {FASHIONCLIP_MODEL_NAME}: {e}")


# ==============================
# EMBEDDING HELPERS
# ==============================
SEG_CROP_CACHE = {}

def _crop_with_person_segmentation(frame, box, return_used=False, cache_key=None):
    if cache_key is not None and cache_key in SEG_CROP_CACHE:
        crop_cached, used_cached, reason_cached = SEG_CROP_CACHE[cache_key]
        if return_used:
            return crop_cached, used_cached, reason_cached
        return crop_cached

    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if y2 <= y1 or x2 <= x1:
        return (None, False, "invalid_box") if return_used else None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (None, False, "empty_crop") if return_used else None
    if reid_mask_model is None:
        return (crop, False, "model_unavailable") if return_used else crop

    try:
        # For YOLOE/open-vocabulary segmentation, fixed COCO class filters
        # (e.g. classes=[0]) can return empty masks. Run unfiltered and keep
        # the largest instance inside this already person-localized crop.
        seg = reid_mask_model.predict(crop, conf=0.10, verbose=False)
        if not seg or seg[0].masks is None or seg[0].boxes is None:
            return (crop, False, "no_mask") if return_used else crop
        masks = seg[0].masks.data
        boxes = seg[0].boxes.xyxy
        if masks is None or len(masks) == 0 or boxes is None or len(boxes) == 0:
            return (crop, False, "no_mask") if return_used else crop
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = int(torch.argmax(areas).item())
        mask = masks[best_idx].detach().cpu().numpy()
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask > 0.5).astype(np.uint8)
        if int(mask_bin.sum()) <= 0:
            return (crop, False, "mask_empty") if return_used else crop
        masked = crop.copy()
        masked[mask_bin == 0] = 0
        result = (masked, True, "ok")
        if cache_key is not None:
            SEG_CROP_CACHE[cache_key] = result
        if return_used:
            return result
        return masked
    except Exception:
        result = (crop, False, "predict_error")
        if cache_key is not None:
            SEG_CROP_CACHE[cache_key] = result
        if return_used:
            return result
        return crop

@torch.no_grad()
def extract_embedding(frame, box, return_seg_used=False, cache_key=None):
    """Returns an L2-normalised embedding tensor, or None if crop too small."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if (y2 - y1) < 32 or (x2 - x1) < 16:
        return (None, False, "small_box") if return_seg_used else None
    crop_bgr, seg_used, seg_reason = _crop_with_person_segmentation(
        frame, box, return_used=True, cache_key=cache_key
    )
    if crop_bgr is None or crop_bgr.size == 0:
        return (None, seg_used, seg_reason) if return_seg_used else None
    crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = REID_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
    feat = reid_model(t)
    emb = F.normalize(feat, p=2, dim=1).cpu().squeeze(0)
    if return_seg_used:
        return emb, seg_used, seg_reason
    return emb


def cosine_sim(a, b):
    return float(torch.dot(a, b).clamp(0, 1))


def best_sim_against_views(query_emb, views):
    """
    Match query against a list of view embeddings.
    Returns the highest cosine similarity found.
    """
    if not views:
        return 0.0
    return max(cosine_sim(query_emb, v) for v in views)


def avg_sim_against_views(query_emb, views):
    """
    Match query against a list of view embeddings.
    Returns the average cosine similarity across all views.
    """
    if not views:
        return 0.0
    sims = [cosine_sim(query_emb, v) for v in views]
    return float(sum(sims) / max(1, len(sims)))


def maybe_add_view(views, new_emb):
    """
    Add new_emb to the view list only if it's sufficiently different
    from all existing views (ensures diversity, avoids redundant copies).
    Caps at MAX_VIEWS_PER_PERSON.
    """
    if new_emb is None:
        return views
    if len(views) >= MAX_VIEWS_PER_PERSON:
        return views
    # Check diversity — only add if dissimilar enough from all stored views
    for v in views:
        if cosine_sim(new_emb, v) > VIEW_DIVERSITY_THRESH:
            return views   # too similar to an existing view, skip
    return views + [new_emb]


@torch.no_grad()
def extract_fashionclip_embedding(crop_bgr):
    if fashionclip_model is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        vec = fashionclip_model.encode_images([pil], batch_size=1)[0]
        if isinstance(vec, np.ndarray):
            t = torch.from_numpy(vec).float()
        else:
            t = torch.tensor(vec, dtype=torch.float32)
        if t.ndim != 1:
            t = t.view(-1)
        return F.normalize(t.unsqueeze(0), p=2, dim=1).squeeze(0)
    except Exception:
        return None


def extract_fashion_pair(frame, box, return_seg_used=False, cache_key=None):
    crop, seg_used, seg_reason = _crop_with_person_segmentation(
        frame, box, return_used=True, cache_key=cache_key
    )
    if crop is None or crop.size == 0:
        return (None, None, seg_used, seg_reason) if return_seg_used else (None, None)
    h = crop.shape[0]
    if h < 24:
        return (None, None, seg_used, seg_reason) if return_seg_used else (None, None)
    split = int(h * 0.55)
    upper = crop[:split, :]
    lower = crop[split:, :]
    upper_emb = extract_fashionclip_embedding(upper)
    lower_emb = extract_fashionclip_embedding(lower)
    if return_seg_used:
        return upper_emb, lower_emb, seg_used, seg_reason
    return upper_emb, lower_emb


def fashion_pair_similarity(curr_u, curr_l, ref_u, ref_l):
    weighted_num = 0.0
    weighted_den = 0.0
    if curr_u is not None and ref_u is not None:
        weighted_num += FASHIONCLIP_UPPER_WEIGHT * cosine_sim(curr_u, ref_u)
        weighted_den += FASHIONCLIP_UPPER_WEIGHT
    if curr_l is not None and ref_l is not None:
        weighted_num += FASHIONCLIP_LOWER_WEIGHT * cosine_sim(curr_l, ref_l)
        weighted_den += FASHIONCLIP_LOWER_WEIGHT
    if weighted_den <= 0.0:
        return -1.0
    return weighted_num / weighted_den


def ema_update_views(views, new_emb):
    """
    Update the closest existing view with EMA, or add as new view.
    This keeps views fresh without destroying diversity.
    """
    if new_emb is None or not views:
        return views
    sims = [cosine_sim(new_emb, v) for v in views]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] > VIEW_DIVERSITY_THRESH:
        # Update the closest view with EMA
        updated = (1 - GALLERY_EMA_ALPHA) * views[best_idx] + GALLERY_EMA_ALPHA * new_emb
        updated = F.normalize(updated.unsqueeze(0), p=2, dim=1).squeeze(0)
        new_views = views.copy()
        new_views[best_idx] = updated
        return new_views
    else:
        # New distinct view — add it
        return maybe_add_view(views, new_emb)


# ==============================
# GEOMETRY HELPERS
# ==============================

def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def get_foot(box):
    x1, y1, x2, y2 = box
    return int((x1+x2)/2), int(y2)

def get_box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def get_box_size(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)

def get_visible_ratio_in_frame(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    full_area = max(1.0, (x2 - x1) * (y2 - y1))
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(frame_w), x2), min(float(frame_h), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0
    clipped_area = (cx2 - cx1) * (cy2 - cy1)
    return float(clipped_area / full_area)


def smooth_box(prev, new):
    return new if prev is None else prev * 0.7 + new * 0.3

def is_event_box_stable(mem, box):
    last_box = mem.get("last_event_box")
    if last_box is None:
        return True

    last_area = max(1.0, get_box_area(last_box))
    area_ratio = get_box_area(box) / last_area
    if area_ratio > EVENT_BOX_AREA_RATIO_MAX or area_ratio < EVENT_BOX_AREA_RATIO_MIN:
        return False

    fx, _ = get_foot(box)
    last_fx = mem.get("last_event_foot_x", fx)
    if abs(fx - last_fx) > EVENT_MAX_FOOT_JUMP:
        return False

    return True

def append_event_history(mem, box):
    cx, _ = get_centroid(box)
    bw, bh = get_box_size(box)
    mem.setdefault("zone_center_history", []).append(cx)
    mem.setdefault("zone_width_history", []).append(bw)
    mem.setdefault("zone_height_history", []).append(bh)
    for key in ("zone_center_history", "zone_width_history", "zone_height_history"):
        if len(mem[key]) > EVENT_HISTORY_LEN:
            mem[key] = mem[key][-EVENT_HISTORY_LEN:]

def box_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) +
                    (b2[2]-b2[0])*(b2[3]-b2[1]) - inter + 1e-6)

def foot_zone_status(box, frame_h):
    fx, fy = get_foot(box)
    if not (frame_h - ZONE_TOP_OFFSET <= fy <= frame_h - ZONE_BOTTOM_OFFSET):
        return "OUT_Y"
    if fx < ZONE_LEFT:
        return "LEFT"
    if fx > ZONE_RIGHT:
        return "RIGHT"
    return "INSIDE"

def get_zone_bounds(frame_h):
    return (
        ZONE_LEFT,
        frame_h - ZONE_TOP_OFFSET,
        ZONE_RIGHT,
        frame_h - ZONE_BOTTOM_OFFSET,
    )

def is_in_door_zone_center(box, frame_h):
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = get_zone_bounds(frame_h)
    ox1, oy1 = max(x1, zx1), max(y1, zy1)
    ox2, oy2 = min(x2, zx2), min(y2, zy2)
    if ox2 <= ox1 or oy2 <= oy1:
        return False
    overlap_area = (ox2 - ox1) * (oy2 - oy1)
    box_area = max(1.0, get_box_area(box))
    overlap_ratio = overlap_area / box_area
    return overlap_ratio >= ZONE_OVERLAP_MIN_RATIO

def is_at_disappear_boundary(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    at_frame_edge = (
        x1 <= DISAPPEAR_EDGE_MARGIN or
        y1 <= DISAPPEAR_EDGE_MARGIN or
        x2 >= frame_w - DISAPPEAR_EDGE_MARGIN or
        y2 >= frame_h - DISAPPEAR_EDGE_MARGIN
    )
    # The configured door zone's right edge is also treated as an exit boundary.
    fx, fy = get_foot(box)
    at_door_right_boundary = (
        fx >= ZONE_RIGHT - DISAPPEAR_EDGE_MARGIN and
        frame_h - ZONE_TOP_OFFSET <= fy <= frame_h - ZONE_BOTTOM_OFFSET
    )
    return at_frame_edge or at_door_right_boundary

def save_crop(frame, box, gid, vname, frame_count, prefix=""):
    """Save a person crop to the global per-ID folder. Returns the path
    written, or None if the crop was empty."""
    out_dir = os.path.join(CROPS_DIR, f"ID{gid}")
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if y2 <= y1 or x2 <= x1:
        return None
    fname = f"{prefix}{vname}_F{frame_count:06d}.jpg"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, frame[y1:y2, x1:x2])
    return path


def save_reid_view_crop(frame, box, gid, vname, frame_count, tag):
    """Save crops used to build/update ReID gallery views, grouped by ID."""
    out_dir = os.path.join(REID_DEBUG_DIR, f"ID{gid}")
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if y2 <= y1 or x2 <= x1:
        return None
    fname = f"{vname}_F{frame_count:06d}_{tag}.jpg"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, frame[y1:y2, x1:x2])
    return path


# ==============================
# PER-VIDEO PROCESSOR
# ==============================

def process_video(video_path, output_dir, cross_state=None):
    if cross_state is None:
        cross_state = {"next_gid": 1, "persistent_gallery": {}}
    persistent_gallery = cross_state["persistent_gallery"]

    vname = os.path.splitext(os.path.basename(video_path))[0]
    out_vid   = os.path.join(output_dir, f"{vname}_output.mp4")
    out_json  = os.path.join(output_dir, f"{vname}_events.json")
    out_log   = os.path.join(output_dir, f"{vname}_log.txt")
    tmp_vid   = os.path.join(output_dir, f"{vname}_temp.mp4")

    def vlog(msg):
        msg_str = str(msg)
        prefix = f"[{vname}] "
        log_msg = "\n".join(prefix + line if line else line for line in msg_str.splitlines())
        print(log_msg)
        with open(out_log, "a") as lf:
            lf.write(log_msg + "\n")

    # Start a fresh per-video log file each run.
    with open(out_log, "w") as lf:
        lf.write("")

    vlog(f"\n{'='*60}\nProcessing: {vname}\n{'='*60}")
    if persistent_gallery:
        vlog(f"  Persistent gallery has {len(persistent_gallery)} known ID(s): "
             f"{sorted(persistent_gallery.keys())}")

    # ── Per-video state ───────────────────────────────────────────────
    track_memory  = {}
    lost_gallery  = {}   # gid → {center, frame, views: [tensor, ...]}
    pending_new_trackers = {}  # tracker_id -> {"embs": [tensor]}
    pending_lost_relinks = {}  # tracker_id -> {"gid", "streak", "last_frame", "last_sim"}
    pending_cross_video_relinks = {}  # tracker_id -> {"gid", "streak", "last_frame", "last_sim"}
    pending_tracker_state = {}  # tracker_id -> pre-ID zone/event state
    id_map        = {}   # yolo tracker_id → global_id
    events        = []
    event_display = {}
    next_gid      = cross_state["next_gid"]

    # ------------------------------------------------------------------
    def emit_event(gid, event_name, frame_count, image_path):
        last_event = track_memory[gid].get("last_event")
        if last_event == event_name:
            return False
        # Exit is valid only if this ID has entered before.
        if event_name == "Exit" and not track_memory[gid].get("has_entry_history", False):
            return False
        if event_name == "Exit":
            if track_memory[gid].get("last_det_conf", 0.0) < MIN_EVENT_DET_CONF:
                return False
            if (track_memory[gid].get("zone_samples", 0) < MIN_EXIT_ZONE_SAMPLES
                    and not track_memory[gid].get("allow_carryover_exit", False)):
                return False
            # TEMP: anti-false-exit quality gates disabled
            # event_box = track_memory[gid].get("last_event_box")
            # if event_box is None:
            #     return False
            # vis_ratio = get_visible_ratio_in_frame(event_box, new_w, new_h)
            # bw, bh = get_box_size(event_box)
            # cx, cy = get_centroid(event_box)
            # if vis_ratio < MIN_EXIT_VISIBLE_RATIO:
            #     if EVENT_DEBUG:
            #         vlog(f"  [F{frame_count}] EXIT blocked gid={gid}: vis={vis_ratio:.2f} < {MIN_EXIT_VISIBLE_RATIO:.2f}")
            #     return False
            # if bw < MIN_EXIT_BOX_WIDTH or bh < MIN_EXIT_BOX_HEIGHT:
            #     if EVENT_DEBUG:
            #         vlog(
            #             f"  [F{frame_count}] EXIT blocked gid={gid}: "
            #             f"box=({bw:.1f},{bh:.1f}) < ({MIN_EXIT_BOX_WIDTH},{MIN_EXIT_BOX_HEIGHT})"
            #         )
            #     return False
            # if (cx < EXIT_CENTER_MARGIN_X or cx > (new_w - EXIT_CENTER_MARGIN_X)
            #         or cy < EXIT_CENTER_MARGIN_Y or cy > (new_h - EXIT_CENTER_MARGIN_Y)):
            #     if EVENT_DEBUG:
            #         vlog(
            #             f"  [F{frame_count}] EXIT blocked gid={gid}: "
            #             f"center=({cx:.1f},{cy:.1f}) near edge margins"
            #         )
            #     return False
        events.append({
            "person_id": int(gid),
            "event": event_name,
            "frame": int(frame_count),
            "image": str(image_path),
        })
        # Keep status for overlay, but allow future opposite events.
        track_memory[gid]["state"] = event_name.lower()
        track_memory[gid]["last_event"] = event_name
        if event_name == "Entry":
            track_memory[gid]["has_entry_history"] = True
        if event_name == "Exit":
            track_memory[gid]["allow_carryover_exit"] = False
        # Re-arm zone tracking so next event requires entering zone again.
        track_memory[gid]["was_in_door_zone"] = False
        track_memory[gid]["last_in_door_zone"] = False
        track_memory[gid]["zone_entry_center_x"] = None
        track_memory[gid]["last_zone_center_x"] = None
        track_memory[gid]["zone_entry_foot_x"] = None
        track_memory[gid]["last_zone_foot_x"] = None
        track_memory[gid]["zone_center_history"] = []
        track_memory[gid]["zone_width_history"] = []
        track_memory[gid]["zone_height_history"] = []

        # Cross-video identity should only keep people currently "inside".
        # On Exit, remove them immediately from the persistent gallery.
        if event_name == "Exit":
            persistent_gallery.pop(gid, None)

        vlog(f"  [F{frame_count}] EVENT gid={gid} type={event_name} image={image_path}")

        event_display[gid] = {
            "event": event_name.upper(),
            "ttl": EVENT_DISPLAY_TTL,
        }
        return True

    def classify_door_event(mem, box, frame_h, disappeared=False):
        cx, _ = get_centroid(box)
        zone_start_x = mem.get("zone_entry_center_x")
        last_zone_x = mem.get("last_zone_center_x")
        if zone_start_x is None:
            zone_start_x = mem.get("first_cx", cx)
        if last_zone_x is None:
            last_zone_x = mem.get("last_cx", cx)

        zone_width = max(1.0, float(ZONE_RIGHT - ZONE_LEFT))
        right_half_threshold = ZONE_LEFT + zone_width * 0.55
        moved_right = last_zone_x >= zone_start_x - 2

        # Main rule: if person was in zone and then disappears, classify by
        # last in-zone position (robust when they vanish before crossing line).
        if disappeared:
            if last_zone_x >= right_half_threshold and moved_right:
                return "Entry"
            return "Exit"

        # Visible leave fallback (no disappearance yet).
        if not is_in_door_zone_center(box, frame_h):
            if last_zone_x >= right_half_threshold and cx > ZONE_RIGHT:
                return "Entry"
            if cx < ZONE_LEFT + EXIT_LEFT_MARGIN:
                return "Exit"
            # Left the zone but still near center: treat as exit by default.
            if last_zone_x < right_half_threshold:
                return "Exit"

        return None

    def update_pending_tracker_state(tracker_id, box, frame_h):
        st = pending_tracker_state.get(tracker_id)
        if st is None:
            st = {
                "was_in_door_zone": False,
                "last_in_door_zone": False,
                "zone_entry_center_x": None,
                "last_zone_center_x": None,
                "zone_entry_foot_x": None,
                "last_zone_foot_x": None,
                "last_zone_box": None,
                "last_foot_side": None,
                "zone_center_history": [],
                "zone_width_history": [],
                "zone_height_history": [],
                "samples": 0,
            }
            pending_tracker_state[tracker_id] = st

        in_door = is_in_door_zone_center(box, frame_h)
        st["last_in_door_zone"] = in_door
        cx, _ = get_centroid(box)
        fx, _ = get_foot(box)
        if in_door:
            st["was_in_door_zone"] = True
            if st["zone_entry_center_x"] is None:
                st["zone_entry_center_x"] = cx
            st["last_zone_center_x"] = cx
            if st["zone_entry_foot_x"] is None:
                st["zone_entry_foot_x"] = fx
            st["last_zone_foot_x"] = fx
            st["last_zone_box"] = box.copy()
            st["last_foot_side"] = "INSIDE"
            st["samples"] += 1
            bw, bh = get_box_size(box)
            st["zone_center_history"].append(cx)
            st["zone_width_history"].append(bw)
            st["zone_height_history"].append(bh)
            for k in ("zone_center_history", "zone_width_history", "zone_height_history"):
                if len(st[k]) > EVENT_HISTORY_LEN:
                    st[k] = st[k][-EVENT_HISTORY_LEN:]
        else:
            st["last_foot_side"] = foot_zone_status(box, frame_h)

    def pop_pending_tracker_state(tracker_id):
        return pending_tracker_state.pop(tracker_id, None)

    def is_gid_booked(gid, requester_tid=None):
        """
        Prevent another tracker from taking a gid that is still owned by a
        recently-seen tracker. Release only after enough missing frames.
        """
        # TEMP: lock disabled
        # return False
        if ID_BOOK_RELEASE_MISSING <= 0:
            return False
        for tid, mapped_gid in id_map.items():
            if mapped_gid != gid:
                continue
            if requester_tid is not None and tid == requester_tid:
                return False
            mem = track_memory.get(gid)
            if mem is None:
                continue
            if mem.get("missing", 0) < ID_BOOK_RELEASE_MISSING:
                return True
        return False

    def match_persistent_gallery(
        emb, box, cx, cy, tracker_id, claimed_gids,
        sim_floor=None
    ):
        # Last-resort match against IDs known from previous videos. Bootstraps
        # a fresh track_memory entry seeded with the persistent gallery views,
        # so the same person carries the same global ID across videos.
        if not CROSS_VIDEO_REID or emb is None or not persistent_gallery:
            return None
        vis_ratio = get_visible_ratio_in_frame(box, new_w, new_h)
        if vis_ratio < CROSS_VIDEO_MIN_VISIBLE_RATIO:
            vlog(
                f"  [F{frame_count}] Cross-video MISS (low visibility): "
                f"tracker {tracker_id}, vis={vis_ratio*100:.1f}% "
                f"< {CROSS_VIDEO_MIN_VISIBLE_RATIO*100:.1f}%"
            )
            return None

        _curr_face_emb_local = None
        if FACE_REID_ENABLE:
            x1b, y1b, x2b, y2b = map(int, box)
            x1b, y1b = max(0, x1b), max(0, y1b)
            x2b, y2b = min(frame.shape[1], x2b), min(frame.shape[0], y2b)
            if x2b > x1b and y2b > y1b:
                _curr_face_emb_local = extract_face_embedding_from_crop(frame[y1b:y2b, x1b:x2b])
        curr_face_debug = get_last_face_reid_debug()
        slot = pending_new_trackers.setdefault(
            tracker_id, {"embs": [], "ambiguous_hold": False, "ambiguous_since": None}
        )

        curr_fu, curr_fl = extract_fashion_pair(frame, box, cache_key=tracker_id)
        candidates = []
        for gid, entry in persistent_gallery.items():
            if gid in claimed_gids or gid in track_memory: 
                continue
            # TEMP: lock disabled
            if is_gid_booked(gid, tracker_id):
                continue
            views = entry.get("views") or []
            if not views:
                continue
            sim_os = avg_sim_against_views(emb, views)
            sim_fc = fashion_pair_similarity(
                curr_fu, curr_fl,
                entry.get("fashion_upper_init"),
                entry.get("fashion_lower_init"),
            )
            face_views = entry.get("face_views") or []
            sim_fr = avg_sim_against_views(_curr_face_emb_local, face_views) if _curr_face_emb_local is not None else -1.0
            weights = []
            scores = []
            body_weight = max(0.0, 1.0 - FASHIONCLIP_CROSS_ALPHA - FACE_REID_CROSS_ALPHA)
            if sim_os >= 0.0:
                weights.append(body_weight if body_weight > 0.0 else 1.0)
                scores.append(sim_os)
            if sim_fc >= 0.0:
                weights.append(FASHIONCLIP_CROSS_ALPHA)
                scores.append(sim_fc)
            if sim_fr >= 0.0:
                weights.append(FACE_REID_CROSS_ALPHA)
                scores.append(sim_fr)
            sim = sim_os
            if weights:
                sim = float(sum(w * s for w, s in zip(weights, scores)) / max(1e-9, sum(weights)))
            candidates.append((sim, gid, sim_os, sim_fc, sim_fr))

        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda x: x[0])
        best_sim, best_gid, _best_os, _best_fc, _best_fr = candidates[0]
        if sim_floor is None:
            sim_floor = CROSS_VIDEO_REID_SIM_THRESHOLD

        # DEBUG: log every cross-video match attempt with all candidate sims
        cand_str = ", ".join(
            f"gid{g}=fused:{s:.3f}/os:{so:.3f}/fc:{sf:.3f}/fr:{('NA' if sr < 0.0 else f'{sr:.3f}')}"
            for s, g, so, sf, sr in candidates[:5]
        )
        if best_sim < sim_floor:
            slot["ambiguous_hold"] = False
            pending_cross_video_relinks.pop(tracker_id, None)
            vlog(f"  [F{frame_count}] Cross-video MISS (below floor): "
                 f"tracker {tracker_id}, best={best_sim:.3f} < {sim_floor} "
                 f"| vis={vis_ratio*100:.1f}% "
                 f"| face_status={curr_face_debug.get('status')} "
                 f"face_reason={curr_face_debug.get('reason')} "
                 f"face_bbox={curr_face_debug.get('bbox')} "
                 f"face_dim={curr_face_debug.get('dim')} "
                 f"| candidates: [{cand_str}]")
            return None
        if (not CROSS_VIDEO_ALLOW_AMBIGUOUS_TOP1
                and len(candidates) > 1
                and best_sim - candidates[1][0] < 0.10):
            if CROSS_VIDEO_DELAY_ON_AMBIGUOUS:
                slot["ambiguous_hold"] = True
                if slot.get("ambiguous_since") is None:
                    slot["ambiguous_since"] = frame_count
            else:
                slot["ambiguous_hold"] = False
            pending_cross_video_relinks.pop(tracker_id, None)
            vlog(f"  [F{frame_count}] Cross-video MISS (ambiguous): "
                 f"tracker {tracker_id}, best=gid{best_gid}@{best_sim:.3f} "
                 f"vs runner-up gid{candidates[1][1]}@{candidates[1][0]:.3f} "
                 f"(gap {best_sim - candidates[1][0]:.3f} < 0.05) "
                 f"| vis={vis_ratio*100:.1f}% | candidates: [{cand_str}]")
            return None

        slot["ambiguous_hold"] = False

        # Require a consecutive-frame streak before confirming cross-video ID.
        cv_slot = pending_cross_video_relinks.get(tracker_id)
        if (cv_slot
                and cv_slot.get("gid") == best_gid
                and frame_count - cv_slot.get("last_frame", -1) == 1):
            cv_slot["streak"] += 1
        else:
            cv_slot = {"gid": best_gid, "streak": 1}
        cv_slot["last_frame"] = frame_count
        cv_slot["last_sim"] = best_sim
        pending_cross_video_relinks[tracker_id] = cv_slot

        if cv_slot["streak"] < max(1, CROSS_VIDEO_MIN_STREAK):
            return None

        bw, bh = box[2] - box[0], box[3] - box[1]
        fx, _ = get_foot(box)
        seeded_views = ema_update_views(
            persistent_gallery[best_gid]["views"].copy(), emb)
        track_memory[best_gid] = {
            "first_cx": int(cx), "last_cx": int(cx), "last_cy": int(cy),
            "first_foot_x": int(fx), "last_foot_x": int(fx),
            "zone_entry_foot_x": None, "last_zone_foot_x": None,
            "zone_entry_center_x": None, "last_zone_center_x": None,
            "last_zone_box": None, "last_foot_side": None,
            "last_event_box": box.copy(), "last_event_foot_x": int(fx),
            "zone_center_history": [], "zone_width_history": [], "zone_height_history": [],
            "smooth_box": box, "missing": 0, "seen_frames": 1,
            "was_in_door_zone": False, "last_in_door_zone": False,
            "state": None, "last_event": None, "last_w": bw, "last_h": bh,
            "has_entry_history": True,
            "zone_samples": 0,
            "last_det_conf": 1.0,
            "allow_carryover_exit": True,
            "last_crop_path": None,
            "views": seeded_views,
            "fashion_upper_init": persistent_gallery[best_gid].get("fashion_upper_init"),
            "fashion_lower_init": persistent_gallery[best_gid].get("fashion_lower_init"),
            "pre_occ_views": None,
            "was_occluding": False,
            # Cross-video carry-over: this ID came from persistent gallery
            # (previously inside). If now seen outside zone, mark Exit.
            "pending_carryover_exit_check": True,
        }
        id_map[tracker_id] = best_gid
        pending_cross_video_relinks.pop(tracker_id, None)
        claimed_gids.add(best_gid)
        vlog(f"  [F{frame_count}] Cross-video match: tracker {tracker_id} "
             f"→ gid {best_gid} (sim {best_sim:.2f}, vis {vis_ratio*100:.1f}%)")
        return best_gid

    def get_pending_avg_embedding(tracker_id, emb):
        if emb is None:
            return None, 0
        slot = pending_new_trackers.setdefault(tracker_id, {"embs": []})
        slot["embs"].append(emb)
        if len(slot["embs"]) > NEW_ID_AGG_FRAMES:
            slot["embs"] = slot["embs"][-NEW_ID_AGG_FRAMES:]
        count = len(slot["embs"])
        stacked = torch.stack(slot["embs"], dim=0)
        avg = stacked.mean(dim=0, keepdim=True)
        avg = F.normalize(avg, p=2, dim=1).squeeze(0)
        return avg, count

    def assign_or_relink(frame, box, cx, cy, tracker_id, claimed_gids):
        nonlocal next_gid
        vis_ratio = get_visible_ratio_in_frame(box, new_w, new_h)

        # Step 1 — existing mapping
        if tracker_id in id_map:
            gid = id_map[tracker_id]
            if gid in claimed_gids:
                # This global ID is already used by another detection
                # in the current frame, so this tracker mapping is stale.
                del id_map[tracker_id]
            elif gid in track_memory:
                mem = track_memory[gid]
                if mem["missing"] > 0:
                    # Was absent — verify with multi-view match
                    emb = extract_embedding(frame, box, cache_key=tracker_id)
                    if emb is not None:
                        sim_os = avg_sim_against_views(emb, mem["views"])
                        curr_fu, curr_fl = extract_fashion_pair(frame, box, cache_key=tracker_id)
                        sim_fc = fashion_pair_similarity(
                            curr_fu, curr_fl,
                            mem.get("fashion_upper_init"),
                            mem.get("fashion_lower_init"),
                        )
                        sim = sim_os
                        if sim_fc >= 0.0:
                            a = max(0.0, min(1.0, FASHIONCLIP_WITHIN_ALPHA))
                            sim = (1.0 - a) * sim_os + a * sim_fc
                        if WITHIN_VIDEO_VISIBILITY_LOG:
                            vlog(
                                f"  [F{frame_count}] ReID verify-missing(vis): "
                                f"tracker={tracker_id} gid={gid} sim={sim:.3f} "
                                f"vis={vis_ratio*100:.1f}% th={REID_SIM_THRESHOLD:.3f}"
                            )
                        if REID_WITHIN_VIDEO_DEBUG:
                            vlog(
                                f"  [F{frame_count}] ReID verify-missing: "
                                f"tracker={tracker_id} gid={gid} sim={sim:.3f} "
                                f"th={REID_SIM_THRESHOLD:.3f}"
                            )
                        if sim >= REID_SIM_THRESHOLD:
                            # Confirmed — also add this as a new view
                            mem["views"] = ema_update_views(mem["views"], emb)
                            pending_lost_relinks.pop(tracker_id, None)
                            claimed_gids.add(gid)
                            return gid
                    # Failed verify — tracker recycled this id
                    del id_map[tracker_id]
                else:
                    pending_new_trackers.pop(tracker_id, None)
                    pending_lost_relinks.pop(tracker_id, None)
                    pending_cross_video_relinks.pop(tracker_id, None)
                    claimed_gids.add(gid)
                    return gid   # continuous, trust it
            else:
                del id_map[tracker_id]

        # Step 2 — ReID against lost gallery (multi-view)
        emb = extract_embedding(frame, box, cache_key=tracker_id)
        best_gid, best_score = None, REID_SIM_THRESHOLD

        if emb is not None:
            curr_fu, curr_fl = extract_fashion_pair(frame, box, cache_key=tracker_id)
            local_candidates = []
            for gid, lost in lost_gallery.items():
                if gid in claimed_gids:
                    continue
                dist = np.linalg.norm(
                    np.array(lost["center"]) - np.array([cx, cy]))
                if dist > SPATIAL_GATE:
                    continue
                sim_os = avg_sim_against_views(emb, lost["views"])
                snap = lost.get("track_snapshot", {})
                sim_fc = fashion_pair_similarity(
                    curr_fu, curr_fl,
                    snap.get("fashion_upper_init"),
                    snap.get("fashion_lower_init"),
                )
                sim = sim_os
                if sim_fc >= 0.0:
                    a = max(0.0, min(1.0, FASHIONCLIP_WITHIN_ALPHA))
                    sim = (1.0 - a) * sim_os + a * sim_fc
                local_candidates.append((sim, gid, dist))
                if sim > best_score:
                    best_score, best_gid = sim, gid
            if REID_WITHIN_VIDEO_DEBUG and local_candidates:
                local_candidates.sort(reverse=True, key=lambda x: x[0])
                top = local_candidates[:3]
                cand_str = ", ".join(
                    f"gid{g}:sim={s:.3f},dist={d:.1f}" for s, g, d in top
                )
                vlog(
                    f"  [F{frame_count}] ReID lost-gallery candidates: "
                    f"tracker={tracker_id} [{cand_str}] "
                    f"(th={REID_SIM_THRESHOLD:.3f})"
                )

        if best_gid is not None:
            slot = pending_lost_relinks.get(tracker_id)
            if slot and slot.get("gid") == best_gid and frame_count - slot.get("last_frame", -1) == 1:
                slot["streak"] += 1
            else:
                slot = {"gid": best_gid, "streak": 1}
            slot["last_frame"] = frame_count
            slot["last_sim"] = best_score
            pending_lost_relinks[tracker_id] = slot

            if REID_WITHIN_VIDEO_DEBUG:
                vlog(
                    f"  [F{frame_count}] ReID relink-candidate(lost-gallery): "
                    f"tracker={tracker_id} gid={best_gid} sim={best_score:.3f} "
                    f"streak={slot['streak']}/{LOST_RELINK_MIN_STREAK}"
                )

            if slot["streak"] >= max(1, LOST_RELINK_MIN_STREAK):
                if REID_WITHIN_VIDEO_DEBUG:
                    vlog(
                        f"  [F{frame_count}] ReID relink(lost-gallery): "
                        f"tracker={tracker_id} -> gid={best_gid} sim={best_score:.3f} "
                        f"(streak met)"
                    )
                id_map[tracker_id] = best_gid
                # Merge new view into gallery views
                lost_gallery[best_gid]["views"] = ema_update_views(
                    lost_gallery[best_gid]["views"], emb)
                # Restore track memory from gallery
                track_memory[best_gid] = {
                    **lost_gallery[best_gid]["track_snapshot"],
                    "missing": 0,
                    "views": lost_gallery[best_gid]["views"],
                    "pre_occ_views": None,       # ← restore excluded fields
                    "was_occluding": False,      # ← this was the crash
                }
                del lost_gallery[best_gid]
                pending_lost_relinks.pop(tracker_id, None)
                pending_new_trackers.pop(tracker_id, None)
                claimed_gids.add(best_gid)
                if WITHIN_VIDEO_VISIBILITY_LOG:
                    vlog(
                        f"  [F{frame_count}] ReID relink(lost-gallery,vis): "
                        f"tracker={tracker_id} -> gid={best_gid} "
                        f"sim={best_score:.3f} vis={vis_ratio*100:.1f}%"
                    )
                return best_gid
            return None
        else:
            pending_lost_relinks.pop(tracker_id, None)

        # Step 2.5 — cross-video persistent gallery
        gid = match_persistent_gallery(emb, box, cx, cy, tracker_id, claimed_gids)
        if gid is not None:
            pending_new_trackers.pop(tracker_id, None)
            pending_lost_relinks.pop(tracker_id, None)
            pending_cross_video_relinks.pop(tracker_id, None)
            return gid
        cv_slot = pending_cross_video_relinks.get(tracker_id)
        if cv_slot and cv_slot.get("streak", 0) < max(1, CROSS_VIDEO_MIN_STREAK):
            return None
        slot = pending_new_trackers.get(tracker_id)
        if slot and slot.get("ambiguous_hold", False):
            return None

        # Step 3 — delayed assignment for new tracker:
        # accumulate a few embeddings, try stronger cross-video match using
        # averaged embedding, then fallback to creating a new ID.
        avg_emb, agg_count = get_pending_avg_embedding(tracker_id, emb)
        if avg_emb is not None and agg_count >= NEW_ID_AGG_FRAMES:
            gid = match_persistent_gallery(
                avg_emb, box, cx, cy, tracker_id, claimed_gids,
                sim_floor=NEW_ID_PERSIST_SIM_THRESHOLD,
            )
            if gid is not None:
                pending_new_trackers.pop(tracker_id, None)
                pending_lost_relinks.pop(tracker_id, None)
                pending_cross_video_relinks.pop(tracker_id, None)
                return gid
            cv_slot = pending_cross_video_relinks.get(tracker_id)
            if cv_slot and cv_slot.get("streak", 0) < max(1, CROSS_VIDEO_MIN_STREAK):
                return None
            slot = pending_new_trackers.get(tracker_id)
            if slot and slot.get("ambiguous_hold", False):
                return None
        elif emb is not None:
            # Not enough evidence yet; skip assigning a global ID this frame.
            return None

        # Fallback: create new person ID.
        new_id = next_gid
        next_gid += 1
        id_map[tracker_id] = new_id
        pending_new_trackers.pop(tracker_id, None)
        pending_lost_relinks.pop(tracker_id, None)
        pending_cross_video_relinks.pop(tracker_id, None)
        claimed_gids.add(new_id)
        return new_id

    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        vlog(f"  Could not open {video_path}")
        return []

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    new_w = RESIZE_WIDTH
    new_h = int(h * RESIZE_WIDTH / w)
    out_fps = max(1.0, fps / FRAME_SKIP)

    writer = cv2.VideoWriter(
        tmp_vid, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (new_w, new_h))

    frame_count = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if MAX_FRAMES_PER_VIDEO and frame_count > MAX_FRAMES_PER_VIDEO:
            vlog(f"  Reached MAX_FRAMES_PER_VIDEO={MAX_FRAMES_PER_VIDEO}, stopping early")
            break
        if frame_count % FRAME_SKIP != 0:
            continue

        if frame_count % 200 == 0:
            vlog(f"  [{frame_count/max(total,1)*100:5.1f}%] "
                 f"frame {frame_count}/{total}  events={len(events)}  "
                 f"elapsed={time.time()-t0:.0f}s")

        frame = cv2.resize(frame, (new_w, new_h))
        clean_frame = frame.copy()
        SEG_CROP_CACHE.clear()
        results = yolo_model.track(frame, persist=True,
                                   tracker=TRACKER_PATH,
                                   classes=[0], conf=0.3, iou=0.5)
        current_ids = set()

        zy1 = new_h - ZONE_TOP_OFFSET
        zy2 = new_h - ZONE_BOTTOM_OFFSET
        cv2.rectangle(frame, (ZONE_LEFT, zy1), (ZONE_RIGHT, zy2), (0,0,255), 2)
        cv2.putText(frame, "EXIT", (ZONE_LEFT - 45, zy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(frame, "ENTRY", (ZONE_RIGHT - 10, zy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,120,0), 2)

        if results[0].boxes.id is not None:
            raw_boxes = results[0].boxes.xyxy.cpu().numpy()
            raw_ids   = results[0].boxes.id.cpu().numpy().astype(int)
            raw_confs = results[0].boxes.conf.cpu().numpy()

            valid = []
            for b, tid, conf in zip(raw_boxes, raw_ids, raw_confs):
                if (b[3] - b[1]) < MIN_BOX_HEIGHT:
                    continue
                if (b[2] - b[0]) * (b[3] - b[1]) < MIN_BOX_AREA:
                    continue
                vis_ratio = get_visible_ratio_in_frame(b, new_w, new_h)
                if vis_ratio < MIN_VISIBLE_RATIO:
                    if WITHIN_VIDEO_VISIBILITY_LOG:
                        vlog(
                            f"  [F{frame_count}] ReID skip(low visibility): "
                            f"tracker={tid} vis={vis_ratio*100:.1f}% "
                            f"< {MIN_VISIBLE_RATIO*100:.1f}%"
                        )
                    continue
                valid.append((b, tid, conf))

            # Occlusion detection
            occ_tids = set()
            for i in range(len(valid)):
                for j in range(i+1, len(valid)):
                    if box_iou(valid[i][0], valid[j][0]) > OCCLUSION_IOU_THRESHOLD:
                        occ_tids.add(valid[i][1])
                        occ_tids.add(valid[j][1])

            # ID assignment
            assignments = []
            claimed_gids = set()
            for box, tid, conf in valid:
                cx, cy = get_centroid(box)
                occ = tid in occ_tids

                # Lock during occlusion if continuously tracked
                if (occ and tid in id_map
                        and id_map[tid] in track_memory
                        and track_memory[id_map[tid]]["missing"] == 0
                        and id_map[tid] not in claimed_gids):
                    gid = id_map[tid]
                    pending_tracker_state.pop(tid, None)
                    claimed_gids.add(gid)
                else:
                    gid = assign_or_relink(frame, box, cx, cy, tid, claimed_gids)
                    if gid is None:
                        update_pending_tracker_state(tid, box, new_h)
                        continue

                assignments.append((gid, box, tid, occ, float(conf)))
                pending_tracker_state.pop(tid, None)

            # Update tracks
            for gid, box, tid, occ, det_conf in assignments:
                raw_box = box.copy()
                cx, cy = get_centroid(box)
                bw, bh = box[2]-box[0], box[3]-box[1]
                current_ids.add(gid)

                if gid not in track_memory:
                    init_emb = extract_embedding(frame, box, cache_key=tid)
                    f_u_init, f_l_init = extract_fashion_pair(frame, box, cache_key=tid)
                    fx, _ = get_foot(box)
                    pre = pop_pending_tracker_state(tid)
                    track_memory[gid] = {
                        "first_cx": cx, "last_cx": cx, "last_cy": cy,
                        "first_foot_x": fx, "last_foot_x": fx,
                        "zone_entry_foot_x": (pre.get("zone_entry_foot_x") if pre else None), "last_zone_foot_x": (pre.get("last_zone_foot_x") if pre else None),
                        "zone_entry_center_x": (pre.get("zone_entry_center_x") if pre else None), "last_zone_center_x": (pre.get("last_zone_center_x") if pre else None),
                        "last_zone_box": (pre.get("last_zone_box") if pre else None), "last_foot_side": (pre.get("last_foot_side") if pre else None),
                        "last_event_box": box.copy(), "last_event_foot_x": fx,
                        "zone_center_history": (pre.get("zone_center_history", []) if pre else []),
                        "zone_width_history": (pre.get("zone_width_history", []) if pre else []),
                        "zone_height_history": (pre.get("zone_height_history", []) if pre else []),
                        "smooth_box": box, "missing": 0, "seen_frames": 1,
                        "was_in_door_zone": (pre.get("was_in_door_zone", False) if pre else False),
                        "last_in_door_zone": (pre.get("last_in_door_zone", False) if pre else False),
                        "state": None, "last_event": None, "last_w": bw, "last_h": bh,
                        "has_entry_history": False,
                        "zone_samples": (pre.get("samples", 0) if pre else 0),
                        "last_det_conf": 1.0,
                        "allow_carryover_exit": False,
                        "last_crop_path": None,
                        "views": [init_emb] if init_emb is not None else [],
                        "fashion_upper_init": f_u_init,
                        "fashion_lower_init": f_l_init,
                        "pre_occ_views": None,
                        "was_occluding": False,
                        "pending_carryover_exit_check": False,
                    }
                    if SAVE_REID_DEBUG_CROPS and init_emb is not None:
                        save_reid_view_crop(clean_frame, box, gid, vname, frame_count, "init_view")

                mem = track_memory[gid]
                was_occ = mem["was_occluding"]
                event_box_ok = is_event_box_stable(mem, raw_box)

                # Snapshot views just before occlusion
                if not was_occ and occ:
                    mem["pre_occ_views"] = mem["views"].copy()

                # Restore clean views right after occlusion ends
                if was_occ and not occ:
                    if mem["pre_occ_views"] is not None:
                        mem["views"] = mem["pre_occ_views"]
                        mem["pre_occ_views"] = None

                mem["was_occluding"] = occ
                mem["smooth_box"] = smooth_box(mem["smooth_box"], box)
                box = mem["smooth_box"]
                event_box = raw_box if event_box_ok else mem.get("last_event_box", box)
                fx, fy = get_foot(event_box)
                mem["last_cx"] = cx
                mem["last_cy"] = cy
                if event_box_ok:
                    mem["last_foot_x"] = fx
                mem["last_w"]  = bw
                mem["last_h"]  = bh
                mem["missing"] = 0
                mem["seen_frames"] += 1
                mem["last_det_conf"] = det_conf

                # Accumulate diverse views on clean frames
                if not occ and mem["seen_frames"] % 8 == 0:
                    new_emb, seg_used_os, seg_reason_os = extract_embedding(
                        frame, box, return_seg_used=True, cache_key=tid
                    )
                    if new_emb is not None:
                        sim_init_os = cosine_sim(new_emb, mem["views"][0]) if mem.get("views") else -1.0
                        sim_views_os = avg_sim_against_views(new_emb, mem.get("views", []))
                        curr_fu, curr_fl, seg_used_fc, seg_reason_fc = extract_fashion_pair(
                            frame, box, return_seg_used=True, cache_key=tid
                        )
                        sim_fc = fashion_pair_similarity(
                            curr_fu, curr_fl,
                            mem.get("fashion_upper_init"),
                            mem.get("fashion_lower_init"),
                        )
                        x1f, y1f, x2f, y2f = map(int, box)
                        x1f, y1f = max(0, x1f), max(0, y1f)
                        x2f, y2f = min(frame.shape[1], x2f), min(frame.shape[0], y2f)
                        curr_face_emb = None
                        if x2f > x1f and y2f > y1f:
                            curr_face_emb = extract_face_embedding_from_crop(frame[y1f:y2f, x1f:x2f])
                        face_debug = get_last_face_reid_debug()
                        face_views_n = len(mem.get("face_views", []))
                        sim_fr = avg_sim_against_views(curr_face_emb, mem.get("face_views", [])) if curr_face_emb is not None else -1.0
                        fc_alpha = max(0.0, min(1.0, FASHIONCLIP_WITHIN_ALPHA)) if sim_fc >= 0.0 else 0.0
                        fr_alpha = FACE_REID_CROSS_ALPHA if sim_fr >= 0.0 else 0.0
                        body_weight = max(0.0, 1.0 - fc_alpha - fr_alpha)
                        weights = []
                        scores = []
                        if sim_views_os >= 0.0:
                            weights.append(body_weight if body_weight > 0.0 else 1.0)
                            scores.append(sim_views_os)
                        if sim_fc >= 0.0:
                            weights.append(fc_alpha if fc_alpha > 0.0 else 1.0)
                            scores.append(sim_fc)
                        if sim_fr >= 0.0:
                            weights.append(fr_alpha if fr_alpha > 0.0 else 1.0)
                            scores.append(sim_fr)
                        fused = sim_views_os
                        if weights:
                            fused = float(sum(w * s for w, s in zip(weights, scores)) / max(1e-9, sum(weights)))
                        if REID_UPDATE_VIEW_LOG:
                            vlog(
                                f"  [F{frame_count}] ReID update-view score: gid={gid} "
                                f"os_init={sim_init_os:.3f} os_views={sim_views_os:.3f} "
                                f"fc={sim_fc:.3f} fr={sim_fr:.3f} fused={fused:.3f} "
                                f"th={REID_UPDATE_VIEW_MIN_SIM:.3f} "
                                f"th_os={REID_UPDATE_VIEW_MIN_OS:.3f} "
                                f"th_fc={REID_UPDATE_VIEW_MIN_FC:.3f} "
                                f"seg_os={'Y' if seg_used_os else 'N'} "
                                f"seg_fc={'Y' if seg_used_fc else 'N'} "
                                f"face_status={face_debug.get('status')} "
                                f"face_reason={face_debug.get('reason')} "
                                f"face_views={face_views_n} "
                                f"seg_reason_os={seg_reason_os} "
                                f"seg_reason_fc={seg_reason_fc}"
                            )
                        if REID_SEG_STRICT and (not seg_used_os or not seg_used_fc):
                            if REID_UPDATE_VIEW_LOG:
                                vlog(
                                    f"  [F{frame_count}] ReID update-view: gid={gid} blocked "
                                    f"(seg_strict os={seg_reason_os} fc={seg_reason_fc})"
                                )
                        elif (fused >= REID_UPDATE_VIEW_MIN_SIM
                                and sim_views_os >= REID_UPDATE_VIEW_MIN_OS
                                and sim_fc >= REID_UPDATE_VIEW_MIN_FC):
                            mem["views"] = ema_update_views(mem["views"], new_emb)
                            if curr_face_emb is not None:
                                if mem.get("face_views"):
                                    mem["face_views"] = ema_update_views(mem.get("face_views", []), curr_face_emb)
                                else:
                                    mem["face_views"] = [curr_face_emb]
                            if SAVE_REID_DEBUG_CROPS:
                                save_reid_view_crop(clean_frame, box, gid, vname, frame_count, "update_view")
                            if REID_UPDATE_VIEW_LOG:
                                vlog(f"  [F{frame_count}] ReID update-view: gid={gid} accepted")
                        else:
                            if REID_UPDATE_VIEW_LOG:
                                vlog(f"  [F{frame_count}] ReID update-view: gid={gid} blocked")

                # Save a crop for every detection of this gid (these crops
                # are the source for offline embedding-based matching).
                crop_path = save_crop(clean_frame, box, gid, vname, frame_count)
                if crop_path is not None:
                    if det_conf >= MIN_EVENT_DET_CONF:
                        mem["last_crop_path"] = crop_path

                # Use smoothed box center for zone-state updates every frame.
                # This keeps state progression even when raw box stability gate fails.
                state_box = box
                in_door = is_in_door_zone_center(state_box, new_h)
                prev_in_door = mem["last_in_door_zone"]
                if event_box_ok:
                    mem["last_event_box"] = raw_box.copy()
                    mem["last_event_foot_x"] = fx

                mem["last_in_door_zone"] = in_door
                scx, _ = get_centroid(state_box)
                sfx, _ = get_foot(state_box)
                if in_door:
                    mem["was_in_door_zone"] = True
                    mem["zone_samples"] = mem.get("zone_samples", 0) + 1
                    if mem.get("zone_entry_center_x") is None:
                        mem["zone_entry_center_x"] = scx
                    mem["last_zone_center_x"] = scx
                    if mem.get("zone_entry_foot_x") is None:
                        mem["zone_entry_foot_x"] = sfx
                    mem["last_zone_foot_x"] = sfx
                    mem["last_zone_box"] = state_box.copy()
                    mem["last_foot_side"] = "INSIDE"
                    append_event_history(mem, state_box)
                else:
                    mem["last_foot_side"] = foot_zone_status(state_box, new_h)

                # If this person was matched from previous videos' persistent
                # gallery and is now already outside the zone, count as Exit.
                if mem.get("pending_carryover_exit_check", False):
                    if (not in_door
                            and mem.get("seen_frames", 0) >= MIN_CARRYOVER_SEEN_FRAMES
                            and mem.get("last_det_conf", 0.0) >= MIN_EVENT_DET_CONF):
                        did_emit = emit_event(gid, "Exit", frame_count, crop_path)
                        if EVENT_DEBUG and did_emit:
                            print(f"  [F{frame_count}] gid {gid} EVENT carry-over: Exit")
                    mem["pending_carryover_exit_check"] = False

                if EVENT_DEBUG and (in_door != prev_in_door):
                    print(
                        f"  [F{frame_count}] gid {gid} zone {'IN' if in_door else 'OUT'} "
                        f"(scx={scx:.1f}, zone_start={mem.get('zone_entry_center_x')}, "
                        f"last_zone={mem.get('last_zone_center_x')})"
                    )

                if prev_in_door and not in_door:
                    event_name = classify_door_event(mem, state_box, new_h)
                    if event_name is not None:
                        if EVENT_DEBUG:
                            print(f"  [F{frame_count}] gid {gid} EVENT on leave: {event_name}")
                        emit_event(gid, event_name, frame_count, crop_path)

                # Draw
                x1, y1, x2, y2 = map(int, box)
                color = ((255,120,0) if gid in event_display and
                         event_display[gid]["event"] == "ENTRY" else
                         (0,0,255) if gid in event_display and
                         event_display[gid]["event"] == "EXIT" else
                         (0,255,0))
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                n_views = len(mem["views"])
                label = f"ID {gid} [{n_views}v]"
                if gid in event_display:
                    label += f" | {event_display[gid]['event']}"
                if occ:
                    label += " [L]"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), (0,0,0), -1)
                cv2.putText(frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Disappeared tracks
        for gid in list(track_memory.keys()):
            if gid not in current_ids:
                mem = track_memory[gid]
                mem["missing"] += 1

                if mem["missing"] == 1:
                    event_box = mem.get("last_event_box", mem["smooth_box"])
                    event_name = classify_door_event(
                        mem, event_box, new_h, disappeared=True)
                    if event_name is not None:
                        if EVENT_DEBUG:
                            print(f"  [F{frame_count}] gid {gid} EVENT on disappear: {event_name}")
                        emit_event(
                            gid,
                            event_name,
                            frame_count,
                            mem.get("last_crop_path"),
                        )

                if (mem["missing"] == 1
                        and mem["seen_frames"] >= MIN_SEEN_FOR_GALLERY
                        and mem["views"]):
                    keep_until_end = not is_at_disappear_boundary(
                        mem["smooth_box"], new_w, new_h)
                    lost_gallery[gid] = {
                        "center": (mem["last_cx"], mem["last_cy"]),
                        "frame":  frame_count,
                        "views":  mem["views"].copy(),
                        "keep_until_end": keep_until_end,
                        # Snapshot full track state so we can restore it on relink
                        "track_snapshot": {
                            k: v for k, v in mem.items()
                            if k not in ("missing", "views",
                                         "pre_occ_views", "was_occluding")
                        }
                    }

                # can_expire = is_at_disappear_boundary(mem["smooth_box"], new_w, new_h)
                # if can_expire and mem["missing"] > REID_MAX_GAP:
                #     del track_memory[gid]
                #     lost_gallery.pop(gid, None)
                #     for k in [k for k, v in id_map.items() if v == gid]:
                #         del id_map[k]

        # for gid in list(lost_gallery.keys()):
        #     if (not lost_gallery[gid].get("keep_until_end", False)
        #             and frame_count - lost_gallery[gid]["frame"] > REID_MAX_GAP):
        #         lost_gallery.pop(gid, None)

        for gid in list(event_display.keys()):
            event_display[gid]["ttl"] -= 1
            if event_display[gid]["ttl"] <= 0:
                del event_display[gid]

        cv2.putText(frame, f"Events: {len(events)}", (10, new_h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        cv2.putText(frame, vname, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        writer.write(frame)

    unresolved_ambiguous = []
    for tid, slot in pending_new_trackers.items():
        if slot.get("ambiguous_hold", False) and tid not in id_map:
            unresolved_ambiguous.append((tid, slot.get("ambiguous_since")))
    if unresolved_ambiguous:
        for tid, since in unresolved_ambiguous:
            vlog(
                f"  [FLAG] Unresolved ambiguous ID until video end: "
                f"tracker={tid}, since_frame={since}"
            )

    cap.release()
    writer.release()
    shutil.copy2(tmp_vid, out_vid)
    os.remove(tmp_vid)
    with open(out_json, "w") as f:
        json.dump(events, f, indent=2)

    # Push final views into the cross-video persistent gallery so the next
    # video can match the same people back to the same gids.
    if CROSS_VIDEO_REID:
        def _commit_views(gid, views, fashion_upper_init=None, fashion_lower_init=None):
            if not views:
                return
            if gid not in persistent_gallery:
                persistent_gallery[gid] = {"views": []}
            for v in views:
                persistent_gallery[gid]["views"] = maybe_add_view(
                    persistent_gallery[gid]["views"], v)
            if fashion_upper_init is not None:
                persistent_gallery[gid]["fashion_upper_init"] = fashion_upper_init
            if fashion_lower_init is not None:
                persistent_gallery[gid]["fashion_lower_init"] = fashion_lower_init

        for gid, mem in track_memory.items():
            if (mem.get("seen_frames", 0) >= MIN_SEEN_FOR_GALLERY
                    and mem.get("last_event") == "Entry"):
                _commit_views(
                    gid,
                    mem.get("views"),
                    mem.get("fashion_upper_init"),
                    mem.get("fashion_lower_init"),
                )
        for gid, lost in lost_gallery.items():
            last_event = (lost.get("track_snapshot") or {}).get("last_event")
            if last_event == "Entry":
                snap = lost.get("track_snapshot", {})
                _commit_views(
                    gid,
                    lost.get("views"),
                    snap.get("fashion_upper_init"),
                    snap.get("fashion_lower_init"),
                )

    cross_state["next_gid"] = next_gid

    vlog(f"  Done — {len(events)} events, {frame_count} frames, "
         f"{time.time()-t0:.1f}s")
    if CROSS_VIDEO_REID:
        vlog(f"  Persistent gallery now: {sorted(persistent_gallery.keys())}")
    return events


# ==============================
# BATCH RUNNER
# ==============================

def find_videos(folder):
    exts = ("*.mp4","*.MP4","*.avi","*.AVI","*.mov","*.MOV","*.mkv","*.MKV")
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

if __name__ == "__main__":
    clear_output_folder(OUTPUT_BASE)
    videos = find_videos(VIDEO_FOLDER)
    if not videos:
        print(f"No videos found in {VIDEO_FOLDER}")
        exit(1)

    print(f"Found {len(videos)} video(s):")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {os.path.basename(v)}")

    all_events = {}
    cross_state = {"next_gid": 1, "persistent_gallery": {}}
    for vp in videos:
        # if '3.mp4' not in vp:
        #     continue
        print(vp)
        #if vp != '/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/5.MP4' and vp != '/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/6.MP4' and vp != '/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/7.MP4' and vp != '/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/8.MP4' :
        # if vp != '/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/2.mp4' and vp != '/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/1.mp4' and vp != '/Users/fredjackyong/Documents/kebunapp/theft_detection/new_video/5.MP4':    
        #     continue
        evts = process_video(vp, OUTPUT_BASE, cross_state)
        if evts:
            all_events[os.path.basename(vp)] = evts

    summary = os.path.join(OUTPUT_BASE, "all_events_summary.json")
    with open(summary, "w") as f:
        json.dump(all_events, f, indent=2)

    total = sum(len(v) for v in all_events.values())
    print(f"\nBatch complete — {len(videos)} videos, {total} total events")
    print(f"Summary: {summary}")
