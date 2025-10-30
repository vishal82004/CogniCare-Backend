from typing import Dict, List

from fastapi import WebSocket


class NotificationManager:
    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, email: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(email, []).append(websocket)

    def disconnect(self, email: str, websocket: WebSocket) -> None:
        connections = self._connections.get(email)
        if not connections:
            return
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self._connections.pop(email, None)

    async def notify_report_ready(self, *, email: str, payload: dict) -> None:
        connections = self._connections.get(email, [])
        for websocket in list(connections):
            try:
                await websocket.send_json(payload)
            except Exception:
                self.disconnect(email, websocket)


notification_manager = NotificationManager()
