from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.client import Config

from .config import settings


def is_spaces_configured() -> bool:
    return all(
        [
            settings.spaces_endpoint_url,
            settings.spaces_bucket,
            settings.spaces_access_key,
            settings.spaces_secret_key,
        ]
    )


def _require_spaces_configured() -> None:
    if not is_spaces_configured():
        raise RuntimeError(
            "DigitalOcean Spaces is not configured. Set endpoint, bucket, access key, and secret key in the API environment."
        )


@lru_cache(maxsize=1)
def _spaces_client():
    _require_spaces_configured()
    return boto3.client(
        "s3",
        region_name=settings.spaces_region,
        endpoint_url=settings.spaces_endpoint_url,
        aws_access_key_id=settings.spaces_access_key,
        aws_secret_access_key=settings.spaces_secret_key,
        config=Config(signature_version="s3v4"),
    )


def is_spaces_public_read_enabled() -> bool:
    return bool(settings.spaces_public_read)


def _public_base_url() -> str:
    configured = str(settings.spaces_public_base_url or "").strip().rstrip("/")
    if configured:
        return configured

    endpoint_url = str(settings.spaces_endpoint_url or "").strip().rstrip("/")
    bucket = str(settings.spaces_bucket or "").strip()
    if not endpoint_url or not bucket:
        raise RuntimeError(
            "DigitalOcean Spaces public URL is not configured. Set THEFT_API_SPACES_PUBLIC_BASE_URL or endpoint/bucket."
        )

    parsed = urlparse(endpoint_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{bucket}.{parsed.netloc}"
    return f"https://{bucket}.{endpoint_url.removeprefix('https://').removeprefix('http://')}"


def generate_public_object_url(object_key: str) -> str:
    _require_spaces_configured()
    return f"{_public_base_url()}/{object_key.lstrip('/')}"


def upload_private_file(local_path: Path, object_key: str, content_type: str | None = None) -> dict[str, str]:
    _require_spaces_configured()
    extra_args: dict[str, str] = {"ACL": "public-read" if is_spaces_public_read_enabled() else "private"}
    if content_type:
        extra_args["ContentType"] = content_type
    _spaces_client().upload_file(str(local_path), settings.spaces_bucket, object_key, ExtraArgs=extra_args)
    payload = {
        "bucket": str(settings.spaces_bucket),
        "object_key": object_key,
        "endpoint_url": str(settings.spaces_endpoint_url),
    }
    if is_spaces_public_read_enabled():
        payload["public_url"] = generate_public_object_url(object_key)
    return payload


def generate_presigned_download_url(object_key: str, expires_seconds: int | None = None) -> str:
    _require_spaces_configured()
    return _spaces_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.spaces_bucket, "Key": object_key},
        ExpiresIn=expires_seconds or settings.spaces_presign_ttl_seconds,
    )
