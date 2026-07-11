# Models for Theft Detection API

This folder is the model home for the FastAPI + runner layer in `tds/`.

## Expected files

- `yolo26s.pt`
- `yolo26l-pose.pt`
- `yoloe-11l-seg.pt`
- `custom_tracker.yaml`
- `reid/osnet_x1_0_msmt17.pt`
- optional: `mobileclip_blt.pt`

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
