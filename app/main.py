from fastapi import FastAPI

from .config import settings
from .routers.alerts import router as alerts_router
from .routers.cctv import router as cctv_router
from .routers.filtering import router as filtering_router
from .routers.health import router as health_router
from .routers.locations import router as locations_router
from .routers.runpod import router as runpod_router
from .routers.sessions import router as sessions_router
from .routers.theft import router as theft_router
from .routers.triggers import router as triggers_router
from .routers.vector import router as vector_router
from .routers.videos import router as videos_router
from .routers.whitelist import router as whitelist_router
from .routers.workers import router as workers_router
from .routers.workflows import router as workflows_router


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.include_router(health_router)
app.include_router(alerts_router)
app.include_router(cctv_router)
app.include_router(filtering_router)
app.include_router(locations_router)
app.include_router(runpod_router)
app.include_router(whitelist_router)
app.include_router(triggers_router)
app.include_router(sessions_router)
app.include_router(theft_router)
app.include_router(videos_router)
app.include_router(vector_router)
app.include_router(workers_router)
app.include_router(workflows_router)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
