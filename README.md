# Theft Detection API

This folder contains the FastAPI server, SQL schema, API docs, and retrieval/workflow orchestration for the theft-detection system.

Current recommended split:

1. `tds` = orchestration FastAPI, DB writes, retrieval, workflow control
2. `tds_vision` = frontend/platform
3. `tds_runner` = GPU-heavy entry/kiosk runner service for Runpod or other burst GPU hosts

It is designed to be the GitHub-ready service layer that:

1. receives Aqara / automation triggers
2. creates theft-detection sessions
3. retrieves CCTV videos
4. dispatches analysis jobs to `tds_runner`
5. stores results into your existing database

## Project Layout

- [app/main.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/app/main.py): FastAPI entrypoint
- [app/config.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/app/config.py): service config
- [mysql_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/mysql_schema.sql): MySQL transactional tables
- [postgres_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/postgres_schema.sql): PostgreSQL vector/gallery tables
- [API_DOCS.md](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/API_DOCS.md): endpoint docs
- [theft_detection_flow.md](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/theft_detection_flow.md): business flow summary
- [models/manifest.json](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/models/manifest.json): required model inventory
- [Dockerfile](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/Dockerfile): container image
- [docker-compose.yml](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/docker-compose.yml): server startup

## Important Architecture Note

Runpod runner mode:

- `tds` does not run local entry or kiosk analysis anymore.
- `tds` uploads runner input artifacts to Spaces and enqueues analysis jobs to `tds_runner` / Runpod.

The detection runtime lives in `tds_runner`, while `tds` keeps orchestration, retrieval, DB writes, webhook handling, and workflow control only.

## Database Roles

This FastAPI service is wired for two databases:

- MySQL for transactional/business records such as triggers, sessions, videos, transactions, kiosk items, and script runs
- PostgreSQL for vector/gallery records such as persistent customer gallery and gallery runtime state

Apply:

- [mysql_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/mysql_schema.sql) to your MySQL application database
- [postgres_schema.sql](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/postgres_schema.sql) to your PostgreSQL vector/gallery database

## Required Models

Expected retrieval-side files:

- any files needed only by `tds_runner` should live in the `tds_runner` deployment, not in `tds`

See [models/manifest.json](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/models/manifest.json) only if you still keep a reference inventory here.

## Docker Deployment

## 1. Files you should have on the server

Suggested server folder:

```text
/opt/theft-detection/
```

Recommended structure:

```text
/opt/theft-detection/
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

- `../session` -> `/app/session`
- `./models` -> `/app/tds/models`

That way:

- FastAPI lives inside the container
- output files persist on the host
- models stay on the host

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

1. Entry and Kiosk script runs are synchronous from the API caller point of view
2. Da Hua retrieval is not implemented yet
3. models are expected to be available on disk before startup

## Recommended Next Steps

Best next steps for production:

1. keep `tds_runner` as the only detection runtime owner
2. add background job execution for Entry/Kiosk
3. add one full workflow endpoint for n8n
4. add retrieval service integration for Da Hua / NVR
