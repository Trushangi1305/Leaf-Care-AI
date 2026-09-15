# AgriSmart AI 🌱

> **AI-Powered Precision Agriculture Platform & Farmer Intelligence Dashboard**

AgriSmart AI is an end-to-end smart farming and precision agriculture web application designed to help farmers, agronomists, and agricultural managers make data-informed decisions. The platform combines deep learning computer vision for real-time crop disease diagnosis with a modern dashboard offering weather intelligence, smart irrigation recommendations, crop suitability matching, sustainability tracking, and a farm-context-aware AI assistant.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [AI & Machine Learning](#-ai--machine-learning)
  - [Production Vision Model (EfficientNet-B0)](#production-vision-model-efficientnet-b0)
  - [Supported Crop & Disease Classes (38 Classes)](#supported-crop--disease-classes-38-classes)
  - [Model Performance Metrics](#model-performance-metrics)
  - [Inference Pipeline](#inference-pipeline)
  - [Legacy Baseline Model](#legacy-baseline-model)
  - [AI Farmer Assistant (LLM Integration)](#ai-farmer-assistant-llm-integration)
- [Technology Stack](#-technology-stack)
- [Database & Data Model](#-database--data-model)
- [API Reference](#-api-reference)
- [Environment Variables](#-environment-variables)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
  - [Production Build & Single-Server Mode](#production-build--single-server-mode)
  - [Docker Deployment](#docker-deployment)
- [Demo Accounts](#-demo-accounts)
- [Project Structure](#-project-structure)
- [Verification & Testing](#-verification--testing)
- [License](#-license)

---

## 🌿 Overview

Farming requires continuous risk management—from spotting leaf pathogens before they spread to scheduling irrigation around rainfall and selecting optimal seasonal crops. AgriSmart AI bridges the gap between field-level challenges and actionable agricultural intelligence:

- **Instant Crop Diagnosis**: Farmers upload leaf photos to identify infections across 14 crop species within seconds, complete with confidence scores, disease summaries, symptom lists, and practical precautions.
- **Context-Aware Assistance**: An interactive conversational assistant answers agronomic questions with responses automatically grounded in the user's saved farm name, location, and crop profile.
- **Unified Farmer Workspace**: Integrates farm weather forecasts, rain probability alerts, soil moisture tracking, water-saving schedules, crop suitability scoring, and 6-month sustainability metrics into a clean, responsive interface.

---

## ✨ Key Features

### 🔍 AI Crop Disease Detection
- **Direct Image Upload & Camera Capture**: Supports JPG, PNG, and WEBP leaf images (up to 10 MB) via drag-and-drop or file picker.
- **Deep Learning Inference**: Powered by a fine-tuned EfficientNet-B0 model trained across 38 distinct crop-disease combinations.
- **Actionable Diagnosis Screen**:
  - Primary diagnosis with percentage confidence and severity rating (*Low*, *Moderate*, *High*).
  - Model architecture verification display (`efficientnet_b0`).
  - Comprehensive disease summaries, symptom lists, and preventive precautions.
  - Top alternative prediction rankings with relative confidence percentages.
  - Clear next-action guidance and agronomic advisory disclaimers.
- **Persistent Scan History**: Automatically records all scans to the user's profile with timestamps, predicted labels, confidence ratings, and top-K rankings.

### 🤖 AI Farmer Assistant
- **Farm-Grounded Dialog**: Ingests the farmer's name, farm name, and location into the backend prompt context to generate relevant agronomic guidance.
- **Configurable LLM Backend**: Integrates with any OpenAI-compatible completions API (`gpt-4o-mini`, custom models, or local proxies).
- **Interactive Chat Interface**: Features suggested prompt pills, quick question shortcuts, and fallback responses for pre-configured demo users.

### 💧 Smart Irrigation Guidance
- **Rainfall Delay Intelligence**: Evaluates incoming 24-hour weather data to suggest postponing irrigation before rain events.
- **Soil Moisture Monitoring**: Visual gauge showing optimal, dry, and wet thresholds alongside crop growth stage and ambient temperature.
- **Usage & History**: Weekly water consumption sparkline tracking and log of past irrigation events.

### 🌾 Crop Recommendation Engine
- **Field Condition Matching**: Evaluates soil type, pH, temperature, humidity, annual rainfall, water availability, growing season (e.g., *Kharif*), and crop rotation history.
- **Suitability Scoring**: Ranks suitable crops (e.g., *Soybean*, *Maize*, *Groundnut*) with compatibility scores out of 100, maturation periods, and water demand ratings.

### ⛅ Weather Intelligence & Alerts
- **5-Day Agricultural Forecast**: Daily high/low temperatures, precipitation chances, humidity levels, wind speed, and UV index.
- **Automated Weather Alerts**: Generates proactive tips (e.g., humidity warnings for fungal vulnerability, high heat warnings for soil moisture checks).

### 📈 Sustainability & Impact Tracker
- **Farm Sustainability Index**: Calculates a 100-point composite score assessing water efficiency, crop health, and resource utilization.
- **Regional Benchmarking**: Compares current farm performance against regional averages.
- **Historical Trends & Improvements**: 6-month trend visualization paired with targeted score-boosting recommendations.

### 👤 User Authentication & Profile Management
- **Secure Authentication**: Built-in sign-up and login powered by salted PBKDF2-HMAC-SHA256 password hashing (240,000 iterations).
- **Persistent Sessions**: Cryptographically signed HTTP-only cookies mapped to server-side SQLite session tokens that persist across server restarts.
- **Editable Farm Profile**: Manage farmer identity, farm name, location, primary crops, soil classification, irrigation infrastructure, and acreage.

---

## 🏛 System Architecture

```mermaid
flowchart TD
    subgraph Frontend["Frontend Client (React 19 + Vite 7 + Tailwind CSS)"]
        UI["Landing & Farmer Dashboard UI"]
        Upload["Crop Image Upload / Camera"]
        ChatUI["AI Farmer Assistant Chat"]
        ProfileUI["Farm Profile & Settings"]
    end

    subgraph Backend["Backend API (FastAPI + Uvicorn)"]
        Router["FastAPI Application Router (backend/main.py)"]
        AuthMiddleware["Session Auth & PBKDF2 Password Verifier"]
        ScanService["Scan Controller (/api/scan)"]
        AssistantService["Assistant Controller (/api/assistant)"]
        StaticServer["SPA Static File Server (dist/public)"]
    end

    subgraph ML_Engine["AI & ML Services"]
        Predictor["Inference Engine (backend/ml/predict.py)"]
        EfficientNet["EfficientNet-B0 PyTorch Checkpoint (.pth)"]
        ExternalLLM["OpenAI-Compatible LLM (AI_API_URL)"]
    end

    subgraph Storage["Persistent Storage"]
        SQLite[("SQLite Database (agrismart.db)")]
        UsersTable["users Table"]
        SessionsTable["sessions Table"]
        ScansTable["scans Table"]
    end

    %% Interactions
    UI -->|HTTP Requests / JSON| Router
    Upload -->|Multipart Form-Data| ScanService
    ChatUI -->|Message Payload| AssistantService
    ProfileUI -->|Update Profile| Router

    Router --> AuthMiddleware
    AuthMiddleware <--> SessionsTable
    AuthMiddleware <--> UsersTable

    ScanService -->|Image Bytes| Predictor
    Predictor -->|Tensor (224x224)| EfficientNet
    EfficientNet -->|Top-K Softmax Probs| Predictor
    Predictor -->|Prediction Result| ScanService
    ScanService -->|Record Scan| ScansTable

    AssistantService -->|Prompt + Farm Context| ExternalLLM
    Router <--> UsersTable
    Router <--> ScansTable
    Router --> StaticServer
```

---

## 🧠 AI & Machine Learning

### Production Vision Model (EfficientNet-B0)

The core disease detection model uses the **EfficientNet-B0** convolutional neural network architecture (`torchvision.models.efficientnet_b0`), fine-tuned for multi-class plant disease classification.

- **Model File**: `backend/ml/models/crop_disease_efficientnet_b0.pth` (~16.5 MB)
- **Framework**: PyTorch (`torch`, `torchvision`, `Pillow`)
- **Number of Classes**: 38 plant and disease classes
- **Input Resolution**: 224 × 224 pixels (images are resized to 256px and center-cropped to 224px)
- **Normalization**: ImageNet standards ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Device Support**: Automatic CUDA GPU acceleration with fallback to CPU (`map_location="cpu"`)
- **Concurrency**: Thread-safe lazy singleton loading with Python threading locks.

### Supported Crop & Disease Classes (38 Classes)

The model detects health states across 14 major agricultural crops:

| Crop | Detectable Conditions & Health States |
|---|---|
| **Apple** | Apple scab, Black rot, Cedar apple rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery mildew, Healthy |
| **Corn (Maize)** | Cercospora leaf spot (Gray leaf spot), Common rust, Northern Leaf Blight, Healthy |
| **Grape** | Black rot, Esca (Black Measles), Leaf blight (Isariopsis Leaf Spot), Healthy |
| **Orange** | Huanglongbing (Citrus greening) |
| **Peach** | Bacterial spot, Healthy |
| **Pepper (Bell)** | Bacterial spot, Healthy |
| **Potato** | Early blight, Late blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery mildew |
| **Strawberry** | Leaf scorch, Healthy |
| **Tomato** | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted spider mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, Healthy |

### Model Performance Metrics

Evaluated on a held-out test split of **8,950 images** across all 38 classes (`backend/ml/models/metrics.json`):

| Metric | Score |
|---|---|
| **Accuracy** | **98.66%** |
| **Precision** | **98.70%** |
| **Recall** | **98.66%** |
| **F1 Score** | **98.66%** |
| **Total Test Images** | 8,950 |
| **Total Classes** | 38 |

### Inference Pipeline

```mermaid
flowchart LR
    A[Image Upload] --> B[BytesIO Buffer]
    B --> C[PIL RGB Conversion]
    C --> D[Resize to 256px]
    D --> E[CenterCrop to 224x224]
    E --> F[ToTensor + ImageNet Normalize]
    F --> G[EfficientNet-B0 Forward Pass]
    G --> H[Softmax Probabilities]
    H --> I[Top-5 Ranking & Label Splitting]
    I --> J[JSON Response + DB Persistence]
```

### Legacy Baseline Model

A historical lightweight baseline using feature engineering is preserved for reference under `legacy/`:
- **Algorithm**: `SGDClassifier` (logistic loss, early stopping)
- **Features**: Histogram of Oriented Gradients (HOG) + HSV Color Histograms at 64×64 resolution
- **Test Accuracy**: 63.63% (evaluated on 1,831 held-out test images)
- **Training Script**: `legacy/train_model.py`
- *Note: This baseline is superseded by the deep learning EfficientNet-B0 model and is not used in production.*

### AI Farmer Assistant (LLM Integration)

The assistant route (`POST /api/assistant`) supports external LLM integration:
1. When `AI_API_KEY` is set, user queries are forwarded to an OpenAI-compatible chat completions endpoint (`https://api.openai.com/v1/chat/completions` by default).
2. The user's farm profile (`name`, `farmName`, `location`) is injected into the system prompt:
   ```text
   You are AgriSmart AI, a careful farming assistant. Farmer: {name}. Farm: {farmName}. Location: {location}. Do not invent farm facts.
   ```
3. If no external key is configured:
   - **Demo accounts** use structured, farm-aware fallback responses.
   - **New registered accounts** receive clear instructions to configure their farm profile and environment keys.

---

## 🛠 Technology Stack

### Frontend
- **Framework**: [React 19](https://react.dev/) (`19.2.1`)
- **Language**: [TypeScript](https://www.typescriptlang.org/) (`5.9.3`)
- **Build Tool**: [Vite](https://vitejs.dev/) (`7.1.7`)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) (`4.1.14`) with `@tailwindcss/vite`
- **Routing**: [Wouter](https://github.com/molefrog/wouter) (`3.3.5`)
- **UI Components**: [Radix UI](https://www.radix-ui.com/) Primitives & Lucide Icons (`lucide-react`)
- **Data & Animation**: [TanStack Query](https://tanstack.com/query), [Framer Motion](https://www.framer.com/motion/), [Sonner](https://sonner.emilkowal.ski/) Toasts, [Streamdown](https://github.com/) Markdown Renderer

### Backend & Machine Learning
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) (`0.115.6`)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/) (`0.34.0`)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (`2.9.0`) & [Torchvision](https://pytorch.org/vision/) (`0.24.0`)
- **Image Processing**: [Pillow (PIL)](https://python-pillow.org/) (`11.1.0`)
- **Data Validation**: [Pydantic v2](https://docs.pydantic.dev/) (`2.10.5`)
- **Database**: SQLite3 (Standard Library)
- **Security**: PBKDF2-HMAC-SHA256 password hashing & HMAC-SHA256 session signature tokens

---

## 🗄 Database & Data Model

The application uses an embedded SQLite database (`backend/data/agrismart.db`). Tables are created and migrated automatically upon backend startup:

### 1. `users` Table
Stores farmer account information and farm attributes.
```sql
CREATE TABLE IF NOT EXISTS users (
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
);
```

### 2. `sessions` Table
Stores authenticated user sessions mapped to HTTP-only cookie digests.
```sql
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    expires_at REAL NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### 3. `scans` Table
Stores historical leaf scan results linked to the authenticated user.
```sql
CREATE TABLE IF NOT EXISTS scans (
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
);
```

---

## 🔌 API Reference

### Health & System
| Method | Endpoint | Description | Auth Required |
|---|---|---|:---:|
| `GET` | `/api/health` | Service health status, database path, and AI configuration status | No |

### Authentication
| Method | Endpoint | Payload | Description |
|---|---|---|---|
| `POST` | `/api/auth/signup` | `{ name, email, password }` | Register a new farmer account and set session cookie |
| `POST` | `/api/auth/login` | `{ email, password }` | Authenticate user and issue signed session cookie |
| `GET` | `/api/auth/me` | None | Returns the currently authenticated user profile |
| `PUT` | `/api/auth/profile` | `{ name, farmName, location, mainCrops, soilType, irrigationMethod, farmSize, latitude, longitude, fieldMarkers }` | Update user's farm profile details |
| `POST` | `/api/auth/logout` | None | Revoke session and clear session cookie |

### Farm Operations & AI Vision
| Method | Endpoint | Payload / Format | Description |
|---|---|---|---|
| `POST` | `/api/scan` | `multipart/form-data` (`image` file, max 10 MB) | Run PyTorch disease classification on leaf photo, store scan record |
| `GET` | `/api/scans` | None | Fetch historical scan records for the current user |
| `GET` | `/api/farm/overview` | None | Returns farm health, active crops count, and irrigation status |
| `GET` | `/api/recommendations` | None | Returns crop suitability recommendations |
| `GET` | `/api/irrigation` | None | Returns soil moisture and irrigation scheduling guidance |
| `GET` | `/api/weather` | None | Returns weather telemetry for the farm location |
| `POST` | `/api/assistant` | `{ message: string }` | Sends query to LLM with farm context or returns fallback response |

---

## ⚙️ Environment Variables

Configure these variables via a `.env` file or export them in your runtime environment:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | Port for the FastAPI/Uvicorn server |
| `AGRISMART_DB_PATH` | `backend/data/agrismart.db` | Absolute or relative path to the SQLite database file |
| `JWT_SECRET` | `change-this-session-secret` | Cryptographic secret for signing session tokens (change in production) |
| `AI_API_KEY` | *(None)* | API key for OpenAI-compatible LLM assistant integration |
| `AI_API_URL` | `https://api.openai.com/v1/chat/completions` | Endpoint URL for the external chat completions API |
| `AI_MODEL` | `gpt-4o-mini` | Model identifier to request from the LLM provider |
| `NODE_ENV` | `development` | Node environment (`development` or `production`) |

---

## 🚀 Getting Started

### Prerequisites
- **Node.js**: v18.0.0+ (Node 20 or 22 recommended) with `pnpm` (`corepack enable pnpm`)
- **Python**: v3.11+ (tested on Python 3.11, 3.12, 3.13, and 3.14)
- **Git**

---

### Local Development Setup

For local frontend-backend development with hot-reloading:

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/agrismart-ai.git
cd agrismart-ai
```

#### 2. Configure Python Virtual Environment & Dependencies

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**On Windows (PowerShell):**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 3. Install Frontend Dependencies
```bash
pnpm install
```

#### 4. Start Development Servers

Run the backend on port `8000` and Vite on port `5173` (or auto-assigned port). Vite is configured to proxy `/api` calls directly to `http://127.0.0.1:8000`.

**Terminal 1 (Backend API):**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend Dev Server):**
```bash
pnpm dev
```

Open the local Vite URL displayed in your terminal (typically `http://localhost:5173`).

---

### Production Build & Single-Server Mode

FastAPI can serve both the API endpoints and the compiled React production bundle from a single port:

```bash
# 1. Build the frontend client into dist/public
pnpm build

# 2. Start the unified production server
uvicorn backend.main:app --host 0.0.0.0 --port 3000
```

Access the application at `http://localhost:3000`.

---

### Docker Deployment

A production-ready [Dockerfile](Dockerfile) is included in the root directory:

```bash
# Build the Docker container image
docker build -t agrismart-ai .

# Run the container exposing port 3000
docker run -p 3000:3000 -e JWT_SECRET="your-secure-production-secret" agrismart-ai
```

---

## 👥 Demo Accounts

The database automatically initializes two pre-seeded demonstration farmer accounts on first startup with sample field history:

| Name | Email | Password | Pre-configured Farm |
|---|---|---|---|
| **Ramesh** | `ramesh.demo@agrismart.ai` | `demo12345` | Ramesh Farm · Nashik, Maharashtra (18 acres, Loamy soil, Drip irrigation) |
| **Suresh** | `suresh.demo@agrismart.ai` | `demo12345` | Suresh Fields · Coimbatore, Tamil Nadu (24 acres, Clay loam, Sprinkler) |

*Note: New accounts created via the **Sign up** form start with a clean workspace ready for custom farm data.*

---

## 📁 Project Structure

```text
agrismart-ai/
├── backend/
│   ├── data/                           # SQLite database storage (agrismart.db)
│   ├── ml/
│   │   ├── models/
│   │   │   ├── classes.json            # 38 crop disease class labels
│   │   │   ├── crop_disease_efficientnet_b0.pth # Trained PyTorch checkpoint (~16.5 MB)
│   │   │   └── metrics.json            # Evaluation metrics (Accuracy, Precision, Recall, F1)
│   │   └── predict.py                  # PyTorch inference engine & preprocessing
│   └── main.py                         # FastAPI server, Auth, SQLite ORM & API routes
├── client/
│   ├── public/                         # Static assets (favicons, icons, robots.txt)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/                     # Radix UI and styled components
│   │   │   ├── AIChatBox.tsx           # Streamdown-enabled chat box
│   │   │   ├── DashboardLayout.tsx     # Workspace layout shell
│   │   │   ├── ErrorBoundary.tsx       # UI error boundary handler
│   │   │   └── Map.tsx                 # Farm map visualization
│   │   ├── contexts/                   # React context providers (ThemeContext)
│   │   ├── hooks/                      # Custom React hooks (useMobile, usePersistFn)
│   │   ├── lib/
│   │   │   ├── api.ts                  # Typed client fetch wrapper for FastAPI endpoints
│   │   │   └── utils.ts                # Class merging (clsx / tailwind-merge)
│   │   ├── pages/
│   │   │   ├── Home.tsx                # Main SPA containing all tabs, views & modals
│   │   │   └── NotFound.tsx            # 404 Fallback page
│   │   ├── App.tsx                     # Top-level Router & Provider setup
│   │   ├── index.css                   # Design system, theme tokens & responsive styles
│   │   └── main.tsx                    # React DOM root mounting
│   └── index.html                      # HTML template
├── legacy/                             # Legacy ML baseline (HOG + HSV classifier reference)
│   ├── crop_disease_model.joblib       # scikit-learn SGDClassifier model file
│   ├── train_model.py                  # Baseline training pipeline
│   ├── predict.py                      # Baseline CLI inference script
│   └── README.md                       # Legacy model documentation
├── Dockerfile                          # Multi-stage production container definition
├── package.json                        # Frontend dependencies and npm scripts
├── requirements.txt                    # Python dependencies (FastAPI, PyTorch, Pillow, etc.)
├── tsconfig.json                       # TypeScript compiler configuration
├── vite.config.ts                      # Vite build configuration with proxy rules
└── vitest.config.ts                    # Vitest unit test configuration
```

---

## 🧪 Verification & Testing

Verify system integrity across frontend and backend:

```bash
# 1. Type check TypeScript files
pnpm check

# 2. Build the frontend production bundle
pnpm build

# 3. Verify Python syntax and backend compilation
python -m py_compile backend/main.py backend/ml/predict.py

# 4. Run automated test suites
pnpm test
```

---

## 📄 License

This project is licensed under the [MIT License](package.json).
