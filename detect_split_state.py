import os
import pickle


def default_cross_state():
    return {
        "next_gid": 1,
        "persistent_gallery": {},
        "persistent_gallery_view_paths": {},
    }


def load_cross_state(path):
    if not path or not os.path.isfile(path):
        return default_cross_state()
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        return default_cross_state()
    data.setdefault("next_gid", 1)
    data.setdefault("persistent_gallery", {})
    data.setdefault("persistent_gallery_view_paths", {})
    return data


def save_cross_state(path, cross_state):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cross_state, f)
