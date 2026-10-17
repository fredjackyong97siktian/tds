"""Experimental OpenRouter Jev matching; deliberately not a live provider."""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

MODEL = "typesafe/jev-1.14"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
MAX_BODY_BYTES = 28 * 1024 * 1024


def prepare_image(raw: bytes) -> str:
    with Image.open(BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024, 1024))
        # Pad rather than distort or stretch small/cropped images. Use the
        # stricter upstream 280px minimum, even where OR documents 256px.
        if min(image.size) < 280:
            canvas = Image.new("RGB", (max(280, image.width), max(280, image.height)))
            canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
            image = canvas
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def download_image(url: str) -> str:
    if urlparse(url).scheme != "https":
        raise ValueError("Test frame URLs must use HTTPS.")
    with urlopen(Request(url, headers={"User-Agent": "TDS-Jev-Test/1.0"}), timeout=20) as response:
        raw = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise ValueError("A frame exceeded the 16 MiB download limit.")
    return prepare_image(raw)


def _timestamp(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value)) if not isinstance(value, datetime) else value
    return result


def build_request(entry: Mapping[str, Any], candidates: list[Mapping[str, Any]], *,
                  frames_per_trigger: int = 5,
                  image_loader: Callable[[str], str] = download_image) -> tuple[dict[str, Any], list[str]]:
    if not 1 <= frames_per_trigger <= 5:
        raise ValueError("Use between 1 and 5 frames per trigger for this experiment.")
    if not 1 <= len(candidates) <= 2:
        raise ValueError("Supply one or two candidate exits for this adjacent-matching experiment.")
    ids = [int(entry["trigger_id"]), *(int(item["trigger_id"]) for item in candidates)]
    if len(set(ids)) != len(ids):
        raise ValueError("Entry and candidate trigger IDs must be different and unique.")
    entry_time = _timestamp(entry["trigger_time"])
    for candidate in candidates:
        gap = _timestamp(candidate["trigger_time"]) - entry_time
        if not timedelta(0) < gap <= timedelta(minutes=30):
            raise ValueError("Candidate exits must be later than entry and within 30 minutes.")
        if entry.get("location_id") is not None and candidate.get("location_id") != entry["location_id"]:
            raise ValueError("All triggers must belong to the same location.")

    urls: list[str] = []
    cache: dict[str, str] = {}

    def images(trigger: Mapping[str, Any]) -> dict[str, Any]:
        frames = sorted(trigger.get("frames") or [], key=lambda frame: int(frame.get("frame_index") or frame.get("index") or 0))[:frames_per_trigger]
        if not frames:
            raise ValueError(f"Trigger {trigger['trigger_id']} has no usable frames.")
        sequence = []
        for frame in frames:
            url = str(frame.get("image_url") or "").strip()
            if not url:
                raise ValueError("A test frame is missing its image URL.")
            if url not in cache:
                cache[url] = image_loader(url)
                urls.append(url)
            sequence.append({
                "frame_index": frame.get("frame_index") or frame.get("index"), "sample_time": str(frame.get("sample_time") or ""),
                "image": {"type": "image_url", "image_url": {"url": cache[url]}},
            })
        return {"trigger_id": int(trigger["trigger_id"]), "trigger_time": str(trigger["trigger_time"]), "frames": sequence}

    state = {"entry": images(entry), "candidates": [images(item) for item in candidates]}
    rules = (
        "Compare the primary customer interacting with or passing through the store doorway, not background pedestrians. "
        "Use every frame in chronological order. Compare visible clothing, trousers, shoes, build, hair and carried items. "
        "Carried items may change after shopping. Timestamps alone never prove a match. Do not guess hidden details. "
        "The entry sequence is a confirmed entry. A matching candidate must show that same customer exiting. "
        "In this fixed camera, entry is moving right/toward the camera; exit is moving left/away from the camera. "
    )
    questions: dict[str, Any] = {
        "exit_choice": {
            "type": "choice", "instructions": rules + "Which candidate shows the same customer exiting as in `entry`? Choose none if no candidate clearly matches.",
            "criteria": {**{str(item["trigger_id"]): f"The customer in candidates[{i}] is the same person as entry and is exiting."
                            for i, item in enumerate(candidates)}, "none": "No candidate clearly shows this customer exiting; they may not have exited yet."},
        },
    }
    for i, item in enumerate(candidates):
        questions[f"same_customer_{int(item['trigger_id'])}"] = {
            "type": "noul", "instructions": rules + f"Do `entry` and `candidates[{i}]` visibly show the same physical customer, with the candidate exiting?",
        }
    return {"model": MODEL, "state": state, "questions": questions}, urls


def _probability(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Jev returned a boolean instead of a probability.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Jev returned an invalid probability.")
    return number


def interpret_response(response: Mapping[str, Any], *, entry_id: int,
                       candidate_ids: list[int], threshold: float = 0.85) -> dict[str, Any]:
    answers = response.get("answers") or {}
    choice = answers.get("exit_choice") or {}
    selected = str(choice.get("choice"))
    allowed = {str(value) for value in candidate_ids} | {"none"}
    if selected not in allowed or choice.get("type") != "choice":
        raise ValueError("Jev returned an unrecognized candidate choice.")
    probabilities = choice.get("probabilities") or {}
    if set(probabilities) != allowed:
        raise ValueError("Jev did not return probabilities for every candidate and none.")
    probabilities = {key: _probability(value) for key, value in probabilities.items()}
    if abs(sum(probabilities.values()) - 1) > 0.02:
        raise ValueError("Jev's choice probabilities do not sum to one.")
    pair_probabilities = {}
    for candidate_id in candidate_ids:
        pair = answers.get(f"same_customer_{candidate_id}") or {}
        if pair.get("type") != "noul":
            raise ValueError("Jev did not return a same-customer decision for every candidate.")
        pair_probabilities[str(candidate_id)] = _probability(pair.get("noul"))
    confidence = _probability(choice.get("confidence"))
    # Provisional thresholds, not calibrated on this store's CCTV. No database
    # group/session writes are authorized by this experimental result.
    decision = "uncertain"
    if selected == "none":
        if confidence >= threshold and all(value <= 1 - threshold for value in pair_probabilities.values()):
            decision = "no_match"
    elif confidence >= threshold and pair_probabilities[selected] >= threshold:
        if all(value <= 1 - threshold for key, value in pair_probabilities.items() if key != selected):
            decision = "match"
    matched_id = int(selected) if decision == "match" else None
    return {
        "decision": decision, "entry_trigger_id": entry_id, "exit_trigger_id": matched_id,
        "choice": selected, "choice_confidence": confidence, "choice_probabilities": probabilities,
        "same_customer_probabilities": pair_probabilities, "experimental_threshold": threshold,
        "groups": [{"entry": [entry_id], "exit": [matched_id]}] if matched_id is not None else [],
        "test_only": True,
    }


def call_openrouter(body: dict[str, Any], *, api_key: str, timeout_seconds: float = 60) -> tuple[dict[str, Any], dict[str, Any]]:
    if not api_key:
        raise ValueError("Set THEFT_API_OPENROUTER_API_KEY or OPENROUTER_API_KEY. A TypeSafe key cannot authenticate with OpenRouter.")
    encoded = json.dumps(body, default=str, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise ValueError("Jev test body is too large; reduce frames per trigger.")
    request = Request(ENDPOINT, data=encoded, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Title": "TDS Jev Matching Experiment",
    }, method="POST")
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as result:
            raw = result.read(4 * 1024 * 1024 + 1)
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace").replace(api_key, "[REDACTED]")
        raise RuntimeError(f"OpenRouter Jev returned HTTP {exc.code}: {detail}") from exc
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("OpenRouter Jev response exceeded the response limit.")
    parsed = json.loads(raw)
    usage = parsed.get("usage") or {}
    return parsed, {
        "provider": "tds_api_openrouter", "model": MODEL,
        "duration_seconds": round(time.monotonic() - started, 3),
        "usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                  "actual_cost_usd": usage.get("cost")},
        "raw_usage": usage, "raw_response": parsed, "response_id": parsed.get("id"),
    }
