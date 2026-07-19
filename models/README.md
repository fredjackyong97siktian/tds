# Models for Theft Detection API

This folder is the model home for the FastAPI + runner layer in `tds/`.

## Folder structure

- `detection/`
- `pose/`
- `segmentation/`
- `tracker/`
- `reid/`
- `text/`

## Required files

- `detection/yolo26s.pt`
- `pose/yolo26l-pose.pt`
- `segmentation/yoloe-11l-seg.pt`
- `tracker/custom_tracker.yaml`
- `reid/osnet_x1_0_msmt17.pt`

## Optional files

- `text/mobileclip_blt.pt`

## What each file is used for

- `detection/yolo26s.pt`
  Entry person detection and default kiosk person/product detection.
- `pose/yolo26l-pose.pt`
  Kiosk pose estimation.
- `segmentation/yoloe-11l-seg.pt`
  Entry-side person segmentation for cleaner ReID crops.
- `tracker/custom_tracker.yaml`
  Tracking config used by entry and kiosk flows.
- `reid/osnet_x1_0_msmt17.pt`
  ReID embedding model for matching the same person.
- `text/mobileclip_blt.pt`
  Optional local MobileCLIP checkpoint for text/fashion similarity.

## Why this exists

The API server should have a stable place to look for model files.
`tds/DetectEntry.py` and `tds/DetectKiosk.py` set environment variables from this folder before importing `Detect.py`.

## Important GitHub note

Large `.pt` files usually should not be committed directly to normal GitHub history.
Use one of these:

1. Git LFS
2. private object storage + startup download
3. mount/copy the models onto the server after deploy

## Current local setup

In this repo, the current `tds/models/` files may be symlinks to existing local weights so we avoid duplicating large files.
