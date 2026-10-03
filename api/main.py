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

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.state import load_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime()
    yield


app = FastAPI(title="Knoxis API", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
app.include_router(router)