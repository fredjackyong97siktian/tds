"""Database-owned Detection Analysis model choice, separate from L0 providers."""

import re

from .. import repositories

DEFAULT_MODEL = "xiaomi/mimo-v2.6-flash"
SETTING_KEY = "detection_analysis_model"
MODEL_OPTIONS = [
    {"value": DEFAULT_MODEL, "label": "MiMo V2.6 Flash"},
    {"value": "qwen/qwen3.8-flash", "label": "Qwen 3.8 Flash"},
]


def validate_model(value):
    value = str(value or "").strip()
    if len(value) > 200 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:/-]*", value):
        raise ValueError("Use an OpenRouter model ID such as xiaomi/mimo-v2.6-flash.")
    return value


def current_model(db):
    saved = repositories.get_app_setting(db, SETTING_KEY)
    return validate_model(saved) if saved else DEFAULT_MODEL
