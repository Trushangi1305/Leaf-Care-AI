from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import Cookie, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.ml import predict as disease_model

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(APP_DIR, "dist", "public")
DATA_DIR = os.path.join(APP_DIR, "backend", "data")
DB_PATH = os.getenv("AGRISMART_DB_PATH", os.path.join(DATA_DIR, "agrismart.db"))
SESSION_COOKIE = "agrismart_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30
SESSION_SECRET = os.getenv("JWT_SECRET", "change-this-session-secret")

app = FastAPI(title="AgriSmart AI API", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class ProfileUpdate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    farmName: str = Field(default="", max_length=160)
    location: str = Field(default="", max_length=160)
    mainCrops: str = Field(default="", max_length=300)
    soilType: str = Field(default="", max_length=120)
    irrigationMethod: str = Field(default="", max_length=120)
    farmSize: str = Field(default="", max_length=80)
    latitude: float | None = None
    longitude: float | None = None
    fieldMarkers: list[dict[str, Any]] = Field(default_factory=list)


class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def db() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    actual_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), actual_salt.encode(), 240_000).hex()
    return digest, actual_salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    digest, _ = hash_password(password, salt)
    return hmac.compare_digest(digest, stored_hash)


def clean_email(email: str) -> str:
    return email.strip().lower()


def init_db() -> None:
    with db() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at REAL NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'farmer',
            farm_name TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            main_crops TEXT NOT NULL DEFAULT '',
            soil_type TEXT NOT NULL DEFAULT '',
            irrigation_method TEXT NOT NULL DEFAULT '',
            farm_size TEXT NOT NULL DEFAULT '',
            member_since TEXT NOT NULL DEFAULT '',
            is_demo INTEGER NOT NULL DEFAULT 0,
            latitude REAL,
            longitude REAL,
            field_markers TEXT NOT NULL DEFAULT '[]',
            created_at INTEGER NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plant TEXT NOT NULL,
            condition TEXT NOT NULL,
            healthy INTEGER NOT NULL,
            confidence REAL NOT NULL,
            top_k TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )""")
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
        additions = {
            "main_crops": "TEXT NOT NULL DEFAULT ''",
            "soil_type": "TEXT NOT NULL DEFAULT ''",
            "irrigation_method": "TEXT NOT NULL DEFAULT ''",
            "farm_size": "TEXT NOT NULL DEFAULT ''",
            "member_since": "TEXT NOT NULL DEFAULT ''",
            "is_demo": "INTEGER NOT NULL DEFAULT 0",
            "latitude": "REAL",
            "longitude": "REAL",
            "field_markers": "TEXT NOT NULL DEFAULT '[]'",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")
        connection.execute("UPDATE users SET farm_name = '' WHERE is_demo = 0 AND farm_name = 'My Farm'")
        connection.execute("UPDATE users SET location = '' WHERE is_demo = 0 AND location = 'Add your location'")
        connection.commit()
    seed_demo_users()


def seed_demo_users() -> None:
    demos = [
        ("Ramesh", "ramesh.demo@agrismart.ai", "Ramesh Farm", "Nashik, Maharashtra", "Tomato, Wheat, Chilli", "Loamy soil", "Drip irrigation", "18 acres"),
        ("Suresh", "suresh.demo@agrismart.ai", "Suresh Fields", "Coimbatore, Tamil Nadu", "Rice, Banana, Coconut", "Clay loam", "Sprinkler irrigation", "24 acres"),
    ]
    with db() as connection:
        for name, email, farm_name, location, crops, soil, irrigation, size in demos:
            existing = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                continue
            password_hash, salt = hash_password("demo12345")
            connection.execute(
                """INSERT INTO users (name, email, password_hash, salt, farm_name, location, main_crops, soil_type, irrigation_method, farm_size, member_since, is_demo, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (name, email, password_hash, salt, farm_name, location, crops, soil, irrigation, size, "September 2024", int(time.time())),
            )
        connection.commit()


def user_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"],
        "farmName": row["farm_name"], "location": row["location"], "mainCrops": row["main_crops"],
        "soilType": row["soil_type"], "irrigationMethod": row["irrigation_method"], "farmSize": row["farm_size"],
        "memberSince": row["member_since"], "isDemo": bool(row["is_demo"]),
        "latitude": row["latitude"], "longitude": row["longitude"],
        "fieldMarkers": json.loads(row["field_markers"] or "[]"),
    }


def session_digest(token: str) -> str:
    return hmac.new(SESSION_SECRET.encode(), token.encode(), hashlib.sha256).hexdigest()


def issue_session(user: dict[str, Any]) -> str:
    token = secrets.token_urlsafe(48)
    now = time.time()
    with db() as connection:
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (session_digest(token), user["id"], now + SESSION_MAX_AGE, int(now)),
        )
        connection.commit()
    return token


def current_user(session: str | None) -> dict[str, Any] | None:
    if not session:
        return None
    digest = session_digest(session)
    now = time.time()
    with db() as connection:
        row = connection.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at >= ?",
            (digest, now),
        ).fetchone()
        connection.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        connection.commit()
    return user_payload(row) if row else None


def require_user(session: str | None) -> dict[str, Any]:
    user = current_user(session)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def set_session(response: Response, user: dict[str, Any]) -> None:
    response.set_cookie(SESSION_COOKIE, issue_session(user), max_age=SESSION_MAX_AGE, httponly=True, secure=False, samesite="lax", path="/")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "agrismart-python-api", "database": DB_PATH, "aiConfigured": bool(os.getenv("AI_API_KEY"))}


@app.post("/api/auth/signup", status_code=201)
def signup(payload: SignupRequest, response: Response) -> dict[str, Any]:
    email = clean_email(payload.email)
    password_hash, salt = hash_password(payload.password)
    try:
        with db() as connection:
            cursor = connection.execute(
                """INSERT INTO users (name, email, password_hash, salt, farm_name, location, main_crops, soil_type, irrigation_method, farm_size, member_since, is_demo, field_markers, created_at)
                VALUES (?, ?, ?, ?, '', '', '', '', '', '', ?, 0, '[]', ?)""",
                (payload.name.strip(), email, password_hash, salt, time.strftime("%B %Y"), int(time.time())),
            )
            row = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
            connection.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = user_payload(row)
    set_session(response, user)
    return {"user": user, "authenticated": True, "created": True}


@app.post("/api/auth/login")
def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    with db() as connection:
        row = connection.execute("SELECT * FROM users WHERE email = ?", (clean_email(payload.email),)).fetchone()
    if not row or not verify_password(payload.password, row["password_hash"], row["salt"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect")
    user = user_payload(row)
    set_session(response, user)
    return {"user": user, "authenticated": True}


@app.get("/api/auth/me")
def me(agrismart_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    user = current_user(agrismart_session)
    return {"user": user, "authenticated": bool(user)}


@app.put("/api/auth/profile")
def update_profile(payload: ProfileUpdate, response: Response, agrismart_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    user = require_user(agrismart_session)
    with db() as connection:
        connection.execute(
            "UPDATE users SET name = ?, farm_name = ?, location = ?, main_crops = ?, soil_type = ?, irrigation_method = ?, farm_size = ?, latitude = ?, longitude = ?, field_markers = ? WHERE id = ?",
            (payload.name.strip(), payload.farmName.strip(), payload.location.strip(), payload.mainCrops.strip(), payload.soilType.strip(), payload.irrigationMethod.strip(), payload.farmSize.strip(), payload.latitude, payload.longitude, json.dumps(payload.fieldMarkers), user["id"]),
        )
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        connection.commit()
    updated = user_payload(row)
    return {"user": updated, "updated": True}


@app.post("/api/auth/logout")
def logout(response: Response, agrismart_session: str | None = Cookie(default=None)) -> dict[str, bool]:
    if agrismart_session:
        with db() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (session_digest(agrismart_session),))
            connection.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"success": True}


@app.get("/api/farm/overview")
def farm_overview(agrismart_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    user = require_user(agrismart_session)
    if not user["isDemo"]:
        return {"cropHealth": None, "activeCrops": 0, "irrigation": None, "temperature": None, "sustainabilityScore": None, "location": user["location"]}
    return {"cropHealth": "Good", "activeCrops": 4, "irrigation": "Due soon", "temperature": 28, "sustainabilityScore": 82, "location": user["location"]}


MAX_SCAN_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB, matches the upload UI's stated limit
ALLOWED_SCAN_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def scan_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "plant": row["plant"],
        "condition": row["condition"],
        "healthy": bool(row["healthy"]),
        "confidence": row["confidence"],
        "topK": json.loads(row["top_k"]),
        "model": row["model"],
        "createdAt": row["created_at"],
    }


@app.post("/api/scan", status_code=201)
async def create_scan(image: UploadFile = File(...), agrismart_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    user = require_user(agrismart_session)

    if image.content_type not in ALLOWED_SCAN_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(data) > MAX_SCAN_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB")

    try:
        result = disease_model.predict_bytes(data)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="The disease detection model is not installed on this server yet")
    except Exception:
        raise HTTPException(status_code=422, detail="Couldn't read that image. Try a clear JPG, PNG, or WEBP photo of a leaf")

    now = int(time.time())
    with db() as connection:
        cursor = connection.execute(
            "INSERT INTO scans (user_id, plant, condition, healthy, confidence, top_k, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user["id"], result["plant"], result["condition"], int(result["healthy"]), result["confidence"], json.dumps(result["topK"]), result["model"], now),
        )
        row = connection.execute("SELECT * FROM scans WHERE id = ?", (cursor.lastrowid,)).fetchone()
        connection.commit()

    return {"scan": scan_row_payload(row)}


@app.get("/api/scans")
def scans(agrismart_session: str | None = Cookie(default=None)) -> list[dict[str, Any]]:
    user = require_user(agrismart_session)
    with db() as connection:
        rows = connection.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 50", (user["id"],)).fetchall()
    real_scans = [scan_row_payload(row) for row in rows]
    if real_scans or not user["isDemo"]:
        return real_scans
    return [{"crop": "Tomato", "disease": "Early blight", "confidence": "91%", "status": "Action needed"}, {"crop": "Wheat", "disease": "Healthy leaf", "confidence": "96%", "status": "Healthy"}, {"crop": "Chilli", "disease": "Leaf curl", "confidence": "84%", "status": "Monitor"}]


@app.get("/api/recommendations")
def recommendations(agrismart_session: str | None = Cookie(default=None)) -> list[dict[str, str]]:
    user = require_user(agrismart_session)
    return [] if not user["isDemo"] else [{"crop": "Soybean", "suitability": "94", "water": "Moderate", "season": "90–110 days"}, {"crop": "Maize", "suitability": "89", "water": "Moderate", "season": "80–95 days"}, {"crop": "Groundnut", "suitability": "82", "water": "Low", "season": "100–120 days"}]


@app.get("/api/irrigation")
def irrigation(agrismart_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    user = require_user(agrismart_session)
    return {"soilMoisture": None, "recommendation": None, "rainProbability": None, "nextCheck": None} if not user["isDemo"] else {"soilMoisture": 63, "recommendation": "Delay irrigation", "rainProbability": 82, "nextCheck": "Tomorrow, 7:00 AM"}


@app.get("/api/weather")
def weather(agrismart_session: str | None = Cookie(default=None)) -> dict[str, Any]:
    user = require_user(agrismart_session)
    return {"location": user["location"], "temperature": None, "humidity": None, "rainProbability": None, "windKph": None} if not user["isDemo"] else {"location": user["location"], "temperature": 28, "humidity": 64, "rainProbability": 18, "windKph": 12}


def external_ai(message: str, user: dict[str, Any]) -> str | None:
    key = os.getenv("AI_API_KEY")
    if not key:
        return None
    url = os.getenv("AI_API_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("AI_MODEL", "gpt-4o-mini")
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": f"You are AgriSmart AI, a careful farming assistant. Farmer: {user['name']}. Farm: {user['farmName'] or 'not provided'}. Location: {user['location'] or 'not provided'}. Do not invent farm facts."}, {"role": "user", "content": message}], "temperature": 0.3}).encode()
    request = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as result:
            parsed = json.loads(result.read().decode())
        return parsed["choices"][0]["message"]["content"]
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError):
        return None


@app.post("/api/assistant")
def assistant(payload: AssistantRequest, agrismart_session: str | None = Cookie(default=None)) -> dict[str, str]:
    user = require_user(agrismart_session)
    generated = external_ai(payload.message, user)
    if generated:
        return {"answer": generated, "source": "configured-ai"}
    if not user["isDemo"]:
        return {"answer": "Your AI provider is not configured yet. Add your farm details first, then configure AI_API_KEY on the server to enable personalized answers.", "source": "setup-needed"}
    text = payload.message.lower()
    answer = "Yellowing tomato leaves can come from overwatering, nutrient stress, or early disease. Check lower leaves first and reduce overhead watering." if "yellow" in text else "Rain is likely tomorrow in your demo farm data, so delay Field B irrigation and re-check soil moisture in the morning." if "irrigat" in text or "water" in text else "Start with a clear leaf scan and check soil moisture in your fields. I can turn the result into a simple action plan."
    return {"answer": answer, "source": "demo-fallback"}


if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")


@app.get("/{path:path}")
def frontend(path: str) -> Response:
    if os.path.isdir(STATIC_DIR):
        requested = os.path.abspath(os.path.join(STATIC_DIR, path))
        if requested.startswith(os.path.abspath(STATIC_DIR)) and os.path.isfile(requested):
            return FileResponse(requested)
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
    return Response("AgriSmart AI frontend build not found. Run pnpm build first.", status_code=503)
