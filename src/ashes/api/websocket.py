"""
WebSocket manager for real-time communication in ASHES
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_data: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept and store new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_data[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.now().isoformat(),
            "subscriptions": set()
        }
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_data = self.client_data.pop(websocket, {})
            logger.info(f"WebSocket client {client_data.get('client_id')} disconnected")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, topic: str = None):
        """Broadcast message to all connected clients or those subscribed to topic"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                # If topic specified, only send to subscribed clients
                if topic:
                    client_subs = self.client_data.get(connection, {}).get("subscriptions", set())
                    if topic not in client_subs:
                        continue
                
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def subscribe(self, websocket: WebSocket, topics: List[str]):
        """Subscribe client to specific topics"""
        if websocket in self.client_data:
            self.client_data[websocket]["subscriptions"].update(topics)
    
    async def unsubscribe(self, websocket: WebSocket, topics: List[str]):
        """Unsubscribe client from specific topics"""
        if websocket in self.client_data:
            self.client_data[websocket]["subscriptions"].difference_update(topics)


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, client_id: str = "anonymous"):
    """Main WebSocket endpoint handler"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_message(websocket, message)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"error": "Invalid JSON format"}),
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def handle_message(websocket: WebSocket, message: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "subscribe":
        topics = message.get("topics", [])
        await manager.subscribe(websocket, topics)
        await manager.send_personal_message(
            json.dumps({
                "type": "subscription_confirmed",
                "topics": topics,
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
    
    elif message_type == "unsubscribe":
        topics = message.get("topics", [])
        await manager.unsubscribe(websocket, topics)
        await manager.send_personal_message(
            json.dumps({
                "type": "unsubscription_confirmed",
                "topics": topics,
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
    
    elif message_type == "ping":
        await manager.send_personal_message(
            json.dumps({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
    
    else:
        await manager.send_personal_message(
            json.dumps({
                "error": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )


async def broadcast_experiment_update(experiment_id: str, status: str, data: Dict[str, Any] = None):
    """Broadcast experiment status updates"""
    message = {
        "type": "experiment_update",
        "experiment_id": experiment_id,
        "status": status,
        "data": data or {},
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(json.dumps(message), "experiments")


async def broadcast_system_alert(level: str, message: str, source: str = "system"):
    """Broadcast system alerts"""
    alert = {
        "type": "system_alert",
        "level": level,
        "message": message,
        "source": source,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(json.dumps(alert), "alerts")


async def broadcast_agent_status(agent_id: str, status: str, task: str = None):
    """Broadcast agent status updates"""
    message = {
        "type": "agent_status",
        "agent_id": agent_id,
        "status": status,
        "task": task,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(json.dumps(message), "agents")


async def broadcast_laboratory_update(device_id: str = None, sensor_data: Dict[str, Any] = None):
    """Broadcast laboratory status updates"""
    message = {
        "type": "laboratory_update",
        "device_id": device_id,
        "sensor_data": sensor_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(json.dumps(message), "laboratory")

import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store connection metadata
        self.connection_info[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "subscription_filters": []
        }
        
        self.logger.info(f"WebSocket connection established for user: {user_id}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Connected to ASHES real-time updates",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        if websocket in self.connection_info:
            user_id = self.connection_info[websocket].get("user_id")
            self.logger.info(f"WebSocket disconnected for user: {user_id}")
            del self.connection_info[websocket]
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            self.logger.error(f"Failed to send message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any], filter_type: str = None):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message, default=str)
        
        # Send to all connections
        disconnected = []
        for connection in self.active_connections:
            try:
                # Apply filters if specified
                if filter_type:
                    connection_filters = self.connection_info.get(connection, {}).get("subscription_filters", [])
                    if filter_type not in connection_filters:
                        continue
                
                await connection.send_text(message_json)
            except Exception as e:
                self.logger.error(f"Failed to broadcast to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send a message to all connections for a specific user."""
        for connection, info in self.connection_info.items():
            if info.get("user_id") == user_id:
                await self.send_personal_message(message, connection)
    
    async def subscribe_to_updates(self, websocket: WebSocket, filters: List[str]):
        """Subscribe a connection to specific update types."""
        if websocket in self.connection_info:
            self.connection_info[websocket]["subscription_filters"] = filters
            await self.send_personal_message({
                "type": "subscription_updated",
                "filters": filters
            }, websocket)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_connected_users(self) -> List[str]:
        """Get list of connected user IDs."""
        return [info.get("user_id") for info in self.connection_info.values() if info.get("user_id")]
