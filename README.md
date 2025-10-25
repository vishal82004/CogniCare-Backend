# Cognicare Backend - Autism Detection API

FastAPI-based backend for autism detection using video analysis and form-based assessments powered by machine learning.

## Features

- 🔐 **JWT Authentication** — Secure user registration and login with bcrypt hashed passwords
- 🎥 **Video Prediction** — Upload videos, extract sharp frames, and classify using a TensorFlow CNN
- 📝 **Form Assessment** — Questionnaire-based prediction via scikit-learn Random Forest model
- 📊 **History Tracking** — Store results in PostgreSQL and fetch per-user prediction history
- 🐳 **Dockerized** — Ready-to-run containers for backend and database
- 🔄 **Async Processing** — Efficient frame extraction using asyncio-based pipeline

## Tech Stack

| Layer          | Technology                                   |
| -------------- | --------------------------------------------- |
| Web Framework  | FastAPI, Uvicorn                              |
| Database       | PostgreSQL, SQLAlchemy ORM                    |
| Auth & Security| OAuth2PasswordBearer, JWT (python-jose), bcrypt |
| ML Models      | TensorFlow 2.x (CNN), scikit-learn RandomForest |
| Packaging      | Docker, docker-compose                        |

## Project Structure

```
Cognicare-Backend/
├── routers/
│   ├── auth.py                # Auth endpoints (register/login/token)
│   ├── videos.py              # Video upload & prediction
│   ├── forms.py               # Form submission & prediction
│   ├── data.py                # Prediction history retrieval
│   ├── predictions.py         # TensorFlow model helper
│   └── Mlpredict/form.py      # Random Forest utilities
├── ml_models/
│   ├── best_model_fine_tuned.h5   # TensorFlow CNN (not tracked)
│   └── asd_rf_model.pkl           # Random Forest model (27 MB)
├── models.py                 # SQLAlchemy models (User, Data)
├── database.py               # DB engine/session configuration
├── image.py                  # Video frame extraction & preprocessing
├── main.py                   # FastAPI application entry-point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Backend container definition
├── docker-compose.yml        # Backend + PostgreSQL stack
├── .env.example              # Sample environment variables (create manually)
├── .gitignore / .dockerignore
└── README.md
```

> **Note:** `ml_models/best_model_fine_tuned.h5` (~1.5 GB) is excluded from GitHub due to size limits. Place it manually before running predictions.

## Getting Started

### 1. Clone & Enter Project

```bash
git clone https://github.com/vishal82004/Cognicare-Backend.git
cd Cognicare-Backend
```

### 2. Create Virtual Environment (optional for local dev)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file (not tracked) and set:

```env
DATABASE_URL=postgresql://user:password@127.0.0.1:5432/autism_db
SECRET_KEY=change_me_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 5. Start PostgreSQL

```bash
docker run -d --name cognicare-postgres ^
  -e POSTGRES_USER=user ^
  -e POSTGRES_PASSWORD=password ^
  -e POSTGRES_DB=autism_db ^
  -p 5432:5432 postgres:13
```

> For Docker Compose setup (backend + DB), run `docker-compose up -d` inside `Cognicare-Backend/`.

### 6. Add Machine Learning Models

- `ml_models/asd_rf_model.pkl` (already tracked, ~27 MB)
- `ml_models/best_model_fine_tuned.h5` (download separately, place in `ml_models/`)

### 7. Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs  
- Health check: http://localhost:8000/health

## API Overview

| Method | Endpoint            | Description                       | Auth |
| ------ | ------------------- | --------------------------------- | ---- |
| POST   | `/auth/`            | Register new user                 | ❌   |
| POST   | `/auth/token`       | Obtain JWT access token           | ❌   |
| GET    | `/auth/`            | List users (demo/admin)           | ✅   |
| POST   | `/video`            | Upload video & get prediction     | ✅   |
| POST   | `/forms`            | Submit questionnaire & predict    | ✅   |
| GET    | `/data/history`     | Fetch user prediction history     | ✅   |
| GET    | `/health`           | Service health probe              | ❌   |

### Auth Flow

1. **Register** with `POST /auth/`
2. **Login** using `POST /auth/token` (form fields: `username`, `password`)
3. Use returned `access_token` as Bearer token for protected endpoints

### Video Prediction Requirements

- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`
- Max upload size: 300 MB
- Internally extracts ≤100 sharp frames using variance of Laplacian
- Model output: `Autistic` / `Non_Autistic` + confidence %

### Form Prediction Fields (`POST /forms`)

All sent as form-data:

| Field                 | Type | Description                                 |
| --------------------- | ---- | ------------------------------------------- |
| `A1` - `A10`          | int  | Autism Spectrum Quotient questions (0/1)    |
| `Age_Mons`            | int  | Age in months                               |
| `Sex`                 | str  | `m` or `f`                                  |
| `Ethnicity`           | str  | e.g., `White-European`                      |
| `Jaundice`            | str  | `yes` / `no`                                |
| `Family_mem_with_ASD` | str  | `yes` / `no`                                |

Example `curl`:

```bash
curl -X POST http://localhost:8000/forms ^
  -H "Authorization: Bearer <TOKEN>" ^
  -F "A1=1" -F "A2=0" -F "A3=1" -F "A4=0" -F "A5=1" ^
  -F "A6=0" -F "A7=1" -F "A8=0" -F "A9=1" -F "A10=0" ^
  -F "Age_Mons=36" -F "Sex=m" ^
  -F "Ethnicity=White-European" -F "Jaundice=no" ^
  -F "Family_mem_with_ASD=yes"
```

## Contact

- **Author**: Vishal Balaji  
- **GitHub**: [@vishal82004](https://github.com/vishal82004)  
- **Repo**: [CogniCare-Backend](https://github.com/vishal82004/CogniCare-Backend)

