"""Knoxis API entry point:  uvicorn api.main:app --port 8000

Layout
  config.py       environment-driven settings and constants
  timeutil.py     time / numeric helpers
  schemas.py      response models
  state.py        engine, model and metadata loaded once at startup
  snapshot.py     data loading, snapshot cache, as_of truncation
  services/       nowcast, forecast, history, timeseries, health
  routes.py       HTTP endpoints

Writer contract: replace the parquet atomically (write a temp file in the same
directory, then os.replace) so readers never see a partial file.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from api.config import ALLOWED_HOSTS, CORS_ORIGINS, PRODUCTION, RATE_LIMIT_PER_MIN
from api.ratelimit import RateLimitMiddleware
from api.routes import router
from api.snapshot import latest_keeper
from api.state import load_runtime

# uvicorn configures only its own loggers; without this, the INFO lines from knoxis.api
# (snapshot timings, causality self-check) are silently dropped.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime()
    latest_keeper.start()
    try:
        yield
    finally:
        latest_keeper.stop()


app = FastAPI(
    title="Knoxis API", lifespan=lifespan,
    docs_url=None if PRODUCTION else "/docs",
    redoc_url=None if PRODUCTION else "/redoc",
    openapi_url=None if PRODUCTION else "/openapi.json",
)

# Development: any origin. Production: KNOXIS_CORS_ORIGINS (default https://knoxis.in and www).
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET"], allow_headers=["*"])
if RATE_LIMIT_PER_MIN > 0:
    app.add_middleware(RateLimitMiddleware, per_minute=RATE_LIMIT_PER_MIN)
if ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.include_router(router)
