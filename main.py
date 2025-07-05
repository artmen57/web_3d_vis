import sys
import asyncio
import base64
import io
import json
import math
import time
import threading
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException, Response, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from pathlib import Path
# Получаем путь к директории где находится main.py
BASE_DIR = Path(__file__).resolve().parent

# Монтируем статические файлы с абсолютным путём
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Монтируем папку static для обслуживания статических файлов
#mainpage="web_3d_vis\\index\\index.html"

#sys.path.append('web_3d_vis/app/auth.py')

# Import database functions and auth
from app.database import get_db, init_db, save_model, get_user_models, get_model_by_id, get_user_by_session
from app.auth import router as auth_router, get_current_user_dependency

class CommandRequest(BaseModel):
    command: str
    params: dict = {}
    session_id: str

class OBJLoader:
    """Simple OBJ file loader"""
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.normals = []
        self.texture_coords = []
    
    def load_obj(self, obj_content: str) -> Tuple[np.ndarray, List[List[int]]]:
        """Parse OBJ file content and return vertices and faces"""
        vertices = []
        faces = []
        
        lines = obj_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'v':  # Vertex
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
            
            elif parts[0] == 'f':  # Face
                face_vertices = []
                for i in range(1, len(parts)):
                    vertex_data = parts[i].split('/')
                    vertex_index = int(vertex_data[0]) - 1  # OBJ uses 1-based indexing
                    face_vertices.append(vertex_index)
                
                # Convert quads to triangles if necessary
                if len(face_vertices) == 3:
                    faces.append(face_vertices)
                elif len(face_vertices) == 4:
                    # Split quad into two triangles
                    faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
                    faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])
        
        if not vertices:
            raise ValueError("No vertices found in OBJ file")
        
        return np.array(vertices, dtype=float), faces

class ClientRenderer:
    """Per-client 3D renderer instance"""
    def __init__(self, session_id: str, width=800, height=600):
        self.session_id = session_id
        self.width = width
        self.height = height
        self.camera_distance = 5.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 1.0
        self.auto_rotate = False
        self.auto_rotate_speed = 0.02
        self.quality = 85  # JPEG quality
        self.render_scale = 1.0  # Scale factor for performance
        
        # Default cube if no model is loaded
        self.vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=float)
        
        self.faces = [
            [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
        ]
        
        self.model_loaded = False
        self.model_name = "Default Cube"
        
        self._normalize_model()
        
        # Thread-safe lock for rendering
        self.render_lock = threading.Lock()
        
        # Performance optimization: cache face normals
        self.face_normals_cache = {}
        self.update_normals_cache()
        
        # Track last activity
        self.last_activity = time.time()
        self.last_frame_data = None
        self.last_frame_hash = None
        self.frame_changed = True
    
    def load_model(self, vertices: np.ndarray, faces: List[List[int]], model_name: str = "Uploaded Model"):
        """Load a new 3D model"""
        with self.render_lock:
            self.vertices = vertices
            self.faces = faces
            self.model_loaded = True
            self.model_name = model_name
            self._normalize_model()
            self._reset_view()
            self.update_normals_cache()
            self.last_activity = time.time()
            self.frame_changed = True
            
            # Adjust quality based on model complexity
            vertex_count = len(self.vertices)
            if vertex_count > 10000:
                self.quality = 70
                self.render_scale = 0.8
            elif vertex_count > 5000:
                self.quality = 75
                self.render_scale = 0.9
            else:
                self.quality = 85
                self.render_scale = 1.0
    
    def update_normals_cache(self):
        """Pre-calculate face normals for performance"""
        self.face_normals_cache = {}
        for i, face in enumerate(self.faces):
            if len(face) >= 3:
                self.face_normals_cache[i] = self.calculate_face_normal(face)
    
    def _normalize_model(self):
        """Normalize model to fit in a standard view"""
        if len(self.vertices) == 0:
            return
        
        # Center the model
        center = np.mean(self.vertices, axis=0)
        self.vertices -= center
        
        # Scale to fit in a 2x2x2 box
        max_extent = np.max(np.abs(self.vertices))
        if max_extent > 0:
            self.vertices /= max_extent
    
    def _reset_view(self):
        """Reset camera view"""
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 1.0
        self.camera_distance = 5.0
        self.frame_changed = True
        
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "name": self.model_name,
            "vertices": len(self.vertices),
            "faces": len(self.faces),
            "loaded": self.model_loaded,
            "quality": self.quality,
            "render_scale": self.render_scale,
            "session_id": self.session_id
        }
    
    def project_vertex(self, vertex):
        """Project 3D vertex to 2D screen coordinates"""
        x, y, z = vertex
        
        # Apply rotations
        cos_y, sin_y = math.cos(self.rotation_y), math.sin(self.rotation_y)
        x_rot = x * cos_y - z * sin_y
        z_rot = x * sin_y + z * cos_y
        
        cos_x, sin_x = math.cos(self.rotation_x), math.sin(self.rotation_x)
        y_rot = y * cos_x - z_rot * sin_x
        z_final = y * sin_x + z_rot * cos_x
        
        z_final += self.camera_distance
        
        if z_final <= 0.1:
            z_final = 0.1
        
        # Perspective projection
        scale = self.zoom * min(self.width, self.height) * 0.3
        screen_x = (x_rot * scale / z_final) + self.width // 2
        screen_y = (y_rot * scale / z_final) + self.height // 2
        
        return int(screen_x), int(screen_y), z_final
    
    def calculate_face_normal(self, face):
        """Calculate face normal for lighting"""
        v1, v2, v3 = [self.vertices[i] for i in face[:3]]
        
        # Calculate two edge vectors
        edge1 = v2 - v1
        edge2 = v3 - v1
        
        # Cross product gives normal
        normal = np.cross(edge1, edge2)
        length = np.linalg.norm(normal)
        if length > 0:
            normal /= length
        
        return normal
    
    def render_frame(self, fast_mode=False):
        """Render a single frame of the 3D model"""
        with self.render_lock:
            self.last_activity = time.time()
            
            # Auto-rotate if enabled
            if self.auto_rotate:
                self.rotation_y += self.auto_rotate_speed
                self.frame_changed = True
            
            # Adjust render size for performance
            render_width = int(self.width * self.render_scale)
            render_height = int(self.height * self.render_scale)
            
            # Create blank image
            img = Image.new('RGB', (render_width, render_height), color=(20, 20, 30))
            draw = ImageDraw.Draw(img)
            
            # Project all vertices
            projected_vertices = []
            for vertex in self.vertices:
                x, y, z = self.project_vertex(vertex)
                # Scale coordinates if rendering at lower resolution
                if self.render_scale < 1.0:
                    x = int(x * self.render_scale)
                    y = int(y * self.render_scale)
                projected_vertices.append((x, y, z))
            
            # Calculate face depths for sorting
            face_data = []
            light_direction = np.array([0, 0, 1])  # Light coming from camera
            
            for i, face in enumerate(self.faces):
                if len(face) < 3:
                    continue
                
                # Calculate average depth
                avg_z = sum(projected_vertices[j][2] for j in face[:3]) / 3
                
                # Use cached normal or calculate
                if i in self.face_normals_cache:
                    normal = self.face_normals_cache[i]
                else:
                    normal = self.calculate_face_normal(face)
                
                light_intensity = max(0.1, np.dot(normal, light_direction))
                
                face_data.append((avg_z, i, light_intensity))
            
            # Sort faces by depth (back to front)
            face_data.sort(key=lambda x: x[0], reverse=True)
            
            # Skip some faces in fast mode for better performance
            if fast_mode and len(face_data) > 1000:
                # Render every other face for complex models
                face_data = face_data[::2]
            
            # Generate a color palette based on model
            base_colors = [
                (120, 120, 120),  # Light grey
                (100, 100, 100),  # Medium grey
                (80, 80, 80),     # Dark grey
                (140, 140, 140),  # Very light grey
                (90, 90, 90),     # Medium dark grey
                (110, 110, 110),  # Light medium grey
            ]
            
            # Draw faces
            for depth, face_idx, light_intensity in face_data:
                face = self.faces[face_idx]
                
                # Get base color
                base_color = base_colors[face_idx % len(base_colors)]
                
                # Apply lighting
                color = tuple(int(c * light_intensity) for c in base_color)
                
                # Get triangle vertices
                triangle_points = []
                all_visible = True
                
                for vertex_idx in face[:3]:  # Only use first 3 vertices for triangle
                    if vertex_idx >= len(projected_vertices):
                        all_visible = False
                        break
                    x, y, _ = projected_vertices[vertex_idx]
                    
                    # Check if point is visible
                    margin = 50 * self.render_scale
                    if x < -margin or x > render_width + margin or y < -margin or y > render_height + margin:
                        all_visible = False
                        break
                    
                    triangle_points.append((x, y))
                
                # Draw triangle if all vertices are reasonable
                if all_visible and len(triangle_points) == 3:
                    try:
                        draw.polygon(triangle_points, fill=color, outline=(255, 255, 255, 50))
                    except:
                        pass  # Skip problematic triangles
            
            # Scale back up if rendered at lower resolution
            if self.render_scale < 1.0:
                img = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
                draw = ImageDraw.Draw(img)
            
            # Draw model info
            info_text = f"{self.model_name} | V: {len(self.vertices)} | F: {len(self.faces)} | Session: {self.session_id[:8]}"
            draw.text((10, 10), info_text, fill=(255, 255, 255))
            
            # Convert to JPEG with dynamic quality
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=self.quality, optimize=True)
            img_buffer.seek(0)
            
            return img_buffer.getvalue()
    
    def render_frame_base64(self, fast_mode=False):
        """Render frame and return as base64 string directly"""
        frame_data = self.render_frame(fast_mode)
        return base64.b64encode(frame_data).decode('utf-8')

# FastAPI app
app = FastAPI(title="3D Model Streaming Server")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Include auth router
app.include_router(auth_router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("Database initialized")

# Store client sessions with their renderers
client_sessions: Dict[str, ClientRenderer] = {}
sessions_lock = threading.Lock()

# Store WebSocket connections mapped to session IDs
websocket_sessions: Dict[str, WebSocket] = {}

# Cleanup thread to remove inactive sessions
def cleanup_inactive_sessions():
    """Remove sessions that have been inactive for more than 30 minutes"""
    while True:
        time.sleep(60)  # Check every minute
        current_time = time.time()
        with sessions_lock:
            inactive_sessions = []
            for session_id, renderer in client_sessions.items():
                if current_time - renderer.last_activity > 1800:  # 30 minutes
                    inactive_sessions.append(session_id)
            
            for session_id in inactive_sessions:
                del client_sessions[session_id]
                if session_id in websocket_sessions:
                    del websocket_sessions[session_id]
                print(f"Cleaned up inactive session: {session_id}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
cleanup_thread.start()

'''@app.get("/")
async def read_index():
    """Serve the HTML file"""
    html_path = os.path.join("index", "index.html")
    if os.path.exists(html_path):
        with open(mainpage, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Please ensure index.html is in the same directory as server.py</p>")
'''
@app.get("/")
async def read_index():
    """Serve the HTML file"""
    # Используем тот же BASE_DIR
    html_path = BASE_DIR / "static" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Please ensure index.html is in the static directory</p>")
@app.post("/upload_obj")
async def upload_obj(
    file: UploadFile = File(...), 
    session_id: str = Form(...),
    save_to_db: bool = Form(False),
    model_name: Optional[str] = Form(None),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Handle OBJ file upload for a specific session"""
    try:
        # Validate session
        if not session_id or session_id not in client_sessions:
            raise HTTPException(status_code=400, detail=f"Invalid session ID: {session_id}")
        
        # Validate file
        if not file.filename.lower().endswith('.obj'):
            raise HTTPException(status_code=400, detail="Only .obj files are supported")
        
        # Read file content
        content = await file.read()
        obj_content = content.decode('utf-8')
        
        # Parse OBJ file
        obj_loader = OBJLoader()
        vertices, faces = obj_loader.load_obj(obj_content)
        
        # Load into the client's renderer
        renderer = client_sessions[session_id]
        renderer.load_model(vertices, faces, file.filename)
        
        # Save to database if requested and user is authenticated
        saved_model_id = None
        if save_to_db:
            try:
                user = get_current_user_dependency(request, db)
                
                # Generate thumbnail
                thumbnail_data = renderer.render_frame()
                thumbnail_b64 = base64.b64encode(thumbnail_data).decode('utf-8')
                
                # Save model
                model = save_model(
                    db,
                    user_id=user.id,
                    name=model_name or file.filename,
                    obj_content=obj_content,
                    vertex_count=len(vertices),
                    face_count=len(faces),
                    thumbnail=thumbnail_b64
                )
                saved_model_id = model.id
            except HTTPException:
                # User not authenticated, skip saving
                pass
        
        return {
            "success": True,
            "model_name": file.filename,
            "vertices": len(vertices),
            "faces": len(faces),
            "session_id": session_id,
            "saved_model_id": saved_model_id
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding error. Please ensure the file is a valid text-based OBJ file.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid OBJ file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/command")
async def handle_command(request: CommandRequest):
    """Handle commands via HTTP POST (fallback for WebSocket)"""
    if request.session_id not in client_sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    
    renderer = client_sessions[request.session_id]
    handle_renderer_command(renderer, request.command, request.params)
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create new session for this client
    session_id = str(uuid.uuid4())
    
    # Create renderer for this session
    with sessions_lock:
        renderer = ClientRenderer(session_id)
        client_sessions[session_id] = renderer
        websocket_sessions[session_id] = websocket
    
    # Send session info to client
    await websocket.send_text(json.dumps({
        "type": "session_created",
        "session_id": session_id
    }))
    
    print(f"New client connected: {session_id}")
    print(f"Active sessions: {list(client_sessions.keys())}")
    
    try:
        # Start streaming frames for this client
        stream_task = asyncio.create_task(stream_frames(websocket, renderer))
        
        # Handle incoming messages
        while True:
            try:
                # Try to receive text first (for JSON messages)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                    message = json.loads(data)
                    
                    if message["type"] == "command":
                        response = handle_renderer_command(renderer, message["command"], message.get("params", {}))
                        if response:
                            await websocket.send_text(json.dumps(response))
                    elif message["type"] == "get_model_info":
                        await websocket.send_text(json.dumps({
                            "type": "model_info",
                            "info": renderer.get_model_info()
                        }))
                except asyncio.TimeoutError:
                    pass
                except json.JSONDecodeError:
                    # Might be binary data, ignore
                    pass
                    
            except asyncio.TimeoutError:
                continue
                
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up session
        with sessions_lock:
            if session_id in client_sessions:
                del client_sessions[session_id]
            if session_id in websocket_sessions:
                del websocket_sessions[session_id]
        stream_task.cancel()
        print(f"Client disconnected: {session_id}")

def handle_renderer_command(renderer: ClientRenderer, command: str, params: dict) -> Optional[dict]:
    """Handle user interaction commands for a specific renderer"""
    
    if command == "rotate":
        direction = params.get("direction")
        if direction == "left":
            renderer.rotation_y -= 0.1
            renderer.frame_changed = True
        elif direction == "right":
            renderer.rotation_y += 0.1
            renderer.frame_changed = True

        elif direction == "up":
            renderer.rotation_x -= 0.1
            renderer.frame_changed = True
        elif direction == "down":
            renderer.rotation_x += 0.1
            
    elif command == "zoom":
        direction = params.get("direction")
        if direction == "in":
            renderer.zoom = min(renderer.zoom * 1.1, 5.0)
            renderer.frame_changed = True
        elif direction == "out":
            renderer.zoom = max(renderer.zoom * 0.9, 0.1)
            renderer.frame_changed = True

            
    elif command == "mouse_drag":
        delta_x = params.get("deltaX", 0)
        delta_y = params.get("deltaY", 0)
        renderer.rotation_y += delta_x * 0.01
        renderer.rotation_x += delta_y * 0.01
        
    elif command == "auto_rotate":
        renderer.auto_rotate = params.get("enabled", False)
        
    elif command == "reset":
        renderer._reset_view()
    
    return None
'''
async def stream_frames(websocket: WebSocket, renderer: ClientRenderer):
    """Stream rendered frames to a specific client"""
    target_fps = 30
    frame_interval = 1.0 / target_fps
    use_binary = True  # Flag to switch between binary and JSON modes
    
    while True:
        try:
            start_time = time.time()
            
            if use_binary:
                # Send raw JPEG data as binary
                frame_data = renderer.render_frame()
                
                # Create a simple header with timestamp (8 bytes)
                timestamp = int(time.time() * 1000).to_bytes(8, byteorder='big')
                
                # Send binary frame with timestamp header
                await websocket.send_bytes(timestamp + frame_data)
            else:
                # Original base64 JSON method
                frame_b64 = renderer.render_frame_base64()
                
                # Send frame
                await websocket.send_text(json.dumps({
                    "type": "frame",
                    "frame": frame_b64,
                    "timestamp": int(time.time() * 1000)
                }))
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error streaming frame for session {renderer.session_id}: {e}")
            break
'''

async def stream_frames(websocket: WebSocket, renderer: ClientRenderer):
    """Stream rendered frames to a specific client"""
    target_fps = 30
    frame_interval = 1.0 / target_fps
    use_binary = True
    idle_frame_interval = 1.0  # Send frame every 1 second when idle
    last_activity_time = time.time()
    
    while True:
        try:
            start_time = time.time()
            
            # Check if frame needs to be sent
            current_time = time.time()
            time_since_activity = current_time - renderer.last_activity
            
            # Determine if we should send a frame
            should_send = False
            
            if renderer.frame_changed:
                # Always send if frame changed
                should_send = True
                renderer.frame_changed = False
                last_activity_time = current_time
            elif renderer.auto_rotate:
                # Always send if auto-rotating
                should_send = True
            elif time_since_activity < 0.5:
                # Send at normal FPS for 0.5 seconds after activity
                should_send = True
            elif current_time - last_activity_time >= idle_frame_interval:
                # Send keepalive frame every second when idle
                should_send = True
                last_activity_time = current_time
            
            if should_send:
                if use_binary:
                    # Send raw JPEG data as binary
                    frame_data = renderer.render_frame()
                    
                    # Check if frame actually changed
                    frame_hash = hash(frame_data)
                    if frame_hash != renderer.last_frame_hash or renderer.auto_rotate:
                        renderer.last_frame_hash = frame_hash
                        
                        # Create a simple header with timestamp (8 bytes)
                        timestamp = int(time.time() * 1000).to_bytes(8, byteorder='big')
                        
                        # Send binary frame with timestamp header
                        await websocket.send_bytes(timestamp + frame_data)
                else:
                    # Original base64 JSON method
                    frame_b64 = renderer.render_frame_base64()
                    
                    # Send frame
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "frame": frame_b64,
                        "timestamp": int(time.time() * 1000)
                    }))
            
            # Control frame rate
            elapsed = time.time() - start_time
            if renderer.auto_rotate or time_since_activity < 0.5:
                # Normal frame rate when active
                sleep_time = max(0, frame_interval - elapsed)
            else:
                # Slower frame rate when idle
                sleep_time = max(0, 0.1 - elapsed)  # Check every 100ms when idle
            
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error streaming frame for session {renderer.session_id}: {e}")
            break

@app.get("/mjpeg_stream/{session_id}")
async def mjpeg_stream(session_id: str):
    """Provide MJPEG stream for a specific session"""
    if session_id not in client_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    renderer = client_sessions[session_id]
    
    async def generate():
        try:
            while True:
                # Render frame for this client
                frame_data = renderer.render_frame(fast_mode=True)
                
                # MJPEG frame format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                # Control frame rate
                await asyncio.sleep(1/30)
                    
        except GeneratorExit:
            pass
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    with sessions_lock:
        sessions = []
        for session_id, renderer in client_sessions.items():
            sessions.append({
                "session_id": session_id,
                "model_name": renderer.model_name,
                "vertices": len(renderer.vertices),
                "faces": len(renderer.faces),
                "last_activity": time.time() - renderer.last_activity
            })
    return {"sessions": sessions, "count": len(sessions)}

@app.get("/models")
async def get_my_models(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_dependency)
):
    """Get all models for current user"""
    models = get_user_models(db, current_user.id)
    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "vertex_count": m.vertex_count,
                "face_count": m.face_count,
                "created_at": m.created_at.isoformat(),
                "thumbnail": m.thumbnail[:100] + "..." if m.thumbnail else None  # Truncate thumbnail
            }
            for m in models
        ]
    }

@app.get("/models/{model_id}")
async def get_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_dependency)
):
    """Get specific model"""
    model = get_model_by_id(db, model_id, current_user.id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "vertex_count": model.vertex_count,
        "face_count": model.face_count,
        "obj_content": model.obj_content,
        "thumbnail": model.thumbnail,
        "created_at": model.created_at.isoformat()
    }

@app.post("/models/{model_id}/load")
async def load_saved_model(
    model_id: int,
    session_id: str = Form(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_dependency)
):
    """Load a saved model into active session"""
    # Validate session
    if not session_id or session_id not in client_sessions:
        raise HTTPException(status_code=400, detail=f"Invalid session ID: {session_id}")
    
    # Get model
    model = get_model_by_id(db, model_id, current_user.id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Parse OBJ content
    obj_loader = OBJLoader()
    vertices, faces = obj_loader.load_obj(model.obj_content)
    
    # Load into renderer
    renderer = client_sessions[session_id]
    renderer.load_model(vertices, faces, model.name)
    
    # Update last accessed time
    model.last_accessed = datetime.utcnow()
    db.commit()
    
    return {
        "success": True,
        "model_name": model.name,
        "vertices": len(vertices),
        "faces": len(faces)
    }

@app.delete("/models/{model_id}")
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_dependency)
):
    """Delete a model"""
    model = get_model_by_id(db, model_id, current_user.id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Only owner can delete
    if model.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    db.delete(model)
    db.commit()
    
    return {"message": "Model deleted successfully"}

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Session 3D OBJ Model Streaming Server")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("")
    print("Features:")
    print("- Each client gets their own independent session")
    print("- Per-client 3D model and rendering state")
    print("- WebSocket streaming for interactive control")
    print("- MJPEG streaming for low-latency viewing")
    print("- Automatic cleanup of inactive sessions")
    print("")
    print("Instructions:")
    print("1. Ensure index.html is in the same directory as this script")
    print("2. Open http://localhost:8000 in multiple browsers/devices")
    print("3. Each client can upload and interact with their own model")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)