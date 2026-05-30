from datetime import datetime, timezone
from fastapi import WebSocket

models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
    "build_time": None,
    "last_trained_at": None,
}

MODEL_REGISTRY = {}
ACTIVE_MODEL_VERSION = None
SHADOW_MODEL_VERSION = None
STAGING_MODEL_VERSION = None

SHADOW_LOGS = []

def generate_model_version():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"1.0.0-{timestamp}"

class RealtimeConnectionHub:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

realtime_hub = RealtimeConnectionHub()
