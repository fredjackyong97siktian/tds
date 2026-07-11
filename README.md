# Theft Detection API

This folder contains the FastAPI server, SQL schema, API docs, model manifest, and script runners for the theft-detection system.

It is designed to be the GitHub-ready service layer that:

1. receives Aqara / automation triggers
2. creates theft-detection sessions
3. runs Entry logic
4. runs Kiosk logic
5. stores results into your existing database

## Project Layout

- [app/main.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/app/main.py): FastAPI entrypoint
- [app/config.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/app/config.py): service config
- [mysql_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/mysql_schema.sql): MySQL transactional tables
- [postgres_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/postgres_schema.sql): PostgreSQL vector/gallery tables
- [API_DOCS.md](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/API_DOCS.md): endpoint docs
- [theft_detection_flow.md](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/theft_detection_flow.md): business flow summary
- [DetectEntry.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/DetectEntry.py): Entry runner
- [DetectKiosk.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/DetectKiosk.py): Kiosk runner
- [model_setup.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/model_setup.py): model env setup
- [models/manifest.json](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/models/manifest.json): required model inventory
- [Dockerfile](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/Dockerfile): container image
- [docker-compose.yml](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/docker-compose.yml): server startup

## Important Architecture Note

Right now, the API runners in `tds/` still import the heavy detection engine from:

- `/app/Detect.py`

So for Docker/server deployment, you need:

1. this `tds/` folder
2. `Detect.py`
3. the required model files
4. output/session storage

This repo already has the service layer in `tds/`, but the detection engine is not yet fully extracted into `tds/detection/`.

## Database Roles

This FastAPI service is wired for two databases:

- MySQL for transactional/business records such as triggers, sessions, videos, transactions, kiosk items, and script runs
- PostgreSQL for vector/gallery records such as persistent customer gallery and gallery runtime state

Apply:

- [mysql_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/mysql_schema.sql) to your MySQL application database
- [postgres_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/postgres_schema.sql) to your PostgreSQL vector/gallery database

## Required Models

Expected model files:

- `tds/models/yolo26s.pt`
- `tds/models/yolo26l-pose.pt`
- `tds/models/yoloe-11l-seg.pt`
- `tds/models/custom_tracker.yaml`
- `tds/models/reid/osnet_x1_0_msmt17.pt`

Optional:

- `tds/models/mobileclip_blt.pt`

See:

- [models/README.md](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/models/README.md)
- [models/manifest.json](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/models/manifest.json)

## Docker Deployment

## 1. Files you should have on the server

Suggested server folder:

```text
/opt/theft-detection/
```

Recommended structure:

```text
/opt/theft-detection/
  Detect.py
  session/
  tds/
    Dockerfile
    docker-compose.yml
    requirements.txt
    app/
    models/
```

## 2. Build files included

This folder now includes:

- [Dockerfile](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/Dockerfile)
- [docker-compose.yml](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/docker-compose.yml)
- [.env.example](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/.env.example)
- [.dockerignore](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/.dockerignore)

## 3. Environment setup

Copy the example env file:

```bash
cd /opt/theft-detection/tds
cp .env.example .env
```

Edit:

```env
THEFT_API_TRANSACTIONAL_DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:3306/DBNAME
THEFT_API_VECTOR_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME
THEFT_API_DEBUG=false
THEFT_API_PYTHON_BIN=python
THEFT_API_VIDEO_STORAGE_DIR=/app/session
```

## 4. Build and run with Docker Compose

From inside `tds/`:

```bash
docker compose up -d --build
```

## 5. Open the API

Default:

```text
http://SERVER_IP:8010
```

Swagger:

```text
http://SERVER_IP:8010/docs
```

## 6. Example production reverse proxy

If you use Nginx:

- proxy `https://theft-api.yourdomain.com` -> `http://127.0.0.1:8010`

## Docker Notes

The compose file mounts:

- `../Detect.py` -> `/app/Detect.py`
- `../session` -> `/app/session`
- `./models` -> `/app/tds/models`

That way:

- FastAPI lives inside the container
- output files persist on the host
- models stay on the host
- `Detect.py` stays available to the runners

## Database Setup

Apply the SQL into both databases:

```bash
mysql -h HOST -u USER -p DBNAME < /opt/theft-detection/tds/mysql_schema.sql
psql "postgresql://USER:PASSWORD@HOST:5432/DBNAME" -f /opt/theft-detection/tds/postgres_schema.sql
```

## Local Run Without Docker

```bash
cd /Users/fredjackyong/Documents/kebunapp/theft_detection/tds
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8010
```

## API Endpoints

See:

- [API_DOCS.md](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/API_DOCS.md)

Main ones:

- `GET /health`
- `POST /api/v1/triggers`
- `POST /api/v1/sessions`
- `POST /api/v1/videos/sessions/{session_id}`
- `POST /api/v1/workflows/triggers/{trigger_id}/run-entry`
- `POST /api/v1/workflows/sessions/{session_id}/run-kiosk`
- `POST /api/v1/sessions/{session_id}/finalize`
- `POST /api/v1/vector/sessions/{session_id}/customer-gallery`
- `PUT /api/v1/vector/sessions/{session_id}/gallery-runtime-state`

## Current Limitations

1. Entry and Kiosk script runs are synchronous
2. Da Hua retrieval is not implemented yet
3. `tds/` still depends on root `Detect.py`
4. models are expected to be available on disk before startup

## Recommended Next Steps

Best next steps for production:

1. move detection engine code from `Detect.py` into `tds/detection/`
2. add background job execution for Entry/Kiosk
3. add one full workflow endpoint for n8n
4. add retrieval service integration for Da Hua / NVR
