"""
Knoxis Backend - Step 3 + Forecast: Full Streaming Web Server & Live Dashboard
=============================================================================
This file powers the complete real-time experience:
1. Serves the stunning dark-mode Live Dashboard at http://localhost:8080/
2. Streams 1-second live satellite telemetry via WebSockets (/ws/live)
3. Connects the 1Hz Telemetry Simulator for real-time mission demonstrations
4. Connects the 60-Minute Flare Forecast Service (/api/forecast)
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api.engine import engine
from api.simulator import simulator
from api.forecast_service import forecast_service

# Connected WebSocket browser clients
connected_clients: set[WebSocket] = set()


async def broadcast_packet(packet: dict):
    """Sends each 1-second satellite reading to every open browser tab."""
    # Also attach the 60-minute forecast prediction to the packet
    forecast_result = forecast_service.predict_60min(list(engine.history))
    packet["forecast"] = {
        "flare_probability": forecast_result["flare_probability"],
        "flare_risk_percent": forecast_result["flare_risk_percent"],
        "forecast_alert_level": forecast_result["forecast_alert_level"],
    }

    for ws in list(connected_clients):
        try:
            await ws.send_json(packet)
        except Exception:
            connected_clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Automatically starts the 1Hz satellite simulator loop when the server boots up."""
    sim_task = asyncio.create_task(simulator.run_loop(broadcast_packet))
    yield
    simulator.is_running = False
    sim_task.cancel()


# 1. Create the FastAPI application with background lifespan manager
app = FastAPI(
    title="Knoxis Space Weather Engine",
    description="Real-time Solar Flare Nowcasting and Forecasting Backend for Aditya-L1 data",
    version="0.4.0",
    lifespan=lifespan,
)

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Pydantic Model for manual feed testing
class TelemetryTick(BaseModel):
    counts: float = Field(..., description="SoLEXS photon counts per second", example=65.0)
    is_valid: bool = Field(True, description="Instrument GTI valid flag")


# 4. Main Door: Real-Time Mission Dashboard!
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serves the real-time mission control dashboard directly in your browser."""
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# 5. Health Check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "knoxis-backend",
        "clients_connected": len(connected_clients),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# 6. Live Status (Nowcast)
@app.get("/api/status")
def get_current_status():
    return engine.get_status()


# 7. Live 60-Minute Forecast!
@app.get("/api/forecast")
def get_current_forecast():
    """
    Returns the 60-minute forward solar flare prediction.
    Uses 9 statistical features extracted from recent history.
    """
    return forecast_service.predict_60min(list(engine.history))


# 8. Manual Feed (from Step 2)
@app.post("/api/feed")
def feed_reading(tick: TelemetryTick):
    result = engine.process_tick(counts=tick.counts, is_valid=tick.is_valid)
    return {"message": "Reading processed successfully", "engine_state": result}


# 9. Detected Flare Catalog
@app.get("/api/catalog")
def get_flare_catalog():
    return {
        "total_flares_detected": len(engine.flare_catalog),
        "flares": engine.flare_catalog,
    }


# 10. Simulation Controls: Trigger Flare Sequence!
@app.post("/api/simulate/flare")
def simulate_solar_flare():
    """Triggers an M-class solar flare eruption simulation curve!"""
    simulator.start_flare()
    return {"message": "Solar flare sequence initiated! Watch the live chart surge."}


# 11. Simulation Controls: Reset to Quiet Sun
@app.post("/api/simulate/quiet")
def simulate_quiet_sun():
    """Returns the solar stream back to normal quiet background levels and clears flare history."""
    simulator.set_quiet()
    engine.reset()
    return {"message": "Sun returned to quiet baseline and flare history cleared."}


# 12. WebSocket Endpoint: The 1Hz Live Telephone Line!
@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive listener
    except WebSocketDisconnect:
        connected_clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8080, reload=True)
