from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from services.notifications import notification_manager

router = APIRouter()

@router.websocket("/ws/notifications")
async def websocket_notifications(
    websocket: WebSocket,
    email: str = Query(..., description="Authenticated user email"),
):
    await notification_manager.connect(email, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        notification_manager.disconnect(email, websocket)
