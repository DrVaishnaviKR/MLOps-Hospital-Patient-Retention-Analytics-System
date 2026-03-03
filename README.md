# 🏥 Patient Churn Prediction — End-to-End MLOps Pipeline

A complete ML classification system for predicting patient churn, built with modern MLOps best practices covering the entire lifecycle from data versioning to deployment.

## 📋 Project Overview

| Component | Technology |
|---|---|
| Database | Neon Postgres (Serverless PostgreSQL) |
| ML Pipeline | scikit-learn + imblearn (SMOTE) |
| Experiment Tracking | Weights & Biases (W&B) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Deployment | Render |

## 🏗️ Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Neon        │───▶│  Feature     │───▶│  Model Training │
│  Postgres    │    │  Engineering │    │  (SMOTE + RF)   │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
                        ┌───────────────────────┘
                        ▼
              ┌──────────────────┐
              │  Hyperparameter  │
              │  Tuning (Grid/   │
              │  Random/Bayes)   │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐    ┌──────────────┐
              │  Best Model      │───▶│  W&B Model   │
              │  (.joblib)       │    │  Registry    │
              └────────┬─────────┘    └──────────────┘
                       │
              ┌────────▼─────────┐    ┌──────────────┐
              │  FastAPI Backend  │◀──│  Streamlit   │
              │  (Docker)        │    │  Frontend    │
              └────────┬─────────┘    └──────────────┘
                       │
              ┌────────▼─────────┐    ┌──────────────┐
              │  Prometheus      │───▶│  Grafana     │
              │  /metrics        │    │  Dashboards  │
              └──────────────────┘    └──────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Neon Postgres account (database credentials in `.env`)

### 1. Clone & Install
```bash
git clone <YOUR_REPO_URL>
cd End_to_end_patient_churn_project
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file:
```env
DB_HOST=your-neon-host
DB_NAME=neondb
DB_USER=your-user
DB_PASSWORD=your-password
DB_PORT=5432
```

### 3. Upload Data & Train Model
```bash
python -m src.database.upload_data
python -m src.training.train
```

### 4. Run API Locally
```bash
uvicorn backend.app:app --reload --port 8000
```

### 5. Run with Docker Compose (API + Prometheus + Grafana)
```bash
docker-compose up --build
```
- 🔗 FastAPI: http://localhost:8000
- 🔗 Prometheus: http://localhost:9090
- 🔗 Grafana: http://localhost:3000 (admin/admin)

### 6. Run Streamlit Frontend
```bash
pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

## 📊 Model Performance

The model uses a **Random Forest Classifier** with **SMOTE** for handling class imbalance (68%/32% churn/no-churn split).

### Hyperparameter Tuning Methods
| Method | Scoring |
|---|---|
| GridSearchCV | ROC-AUC |
| RandomizedSearchCV | ROC-AUC |
| BayesSearchCV (scikit-optimize) | ROC-AUC |

The best model is automatically selected and saved.

## 🧪 Testing
```bash
python -m pytest tests/ -v
```

## 🔍 Code Quality
```bash
flake8 backend/ src/ tests/ frontend/
pylint backend/ src/ tests/ --disable=C0114,C0115,C0116
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict patient churn |
| GET | `/metrics` | Prometheus metrics |

### POST /predict — Example Request
```json
{
    "Age": 45,
    "Gender": "Male",
    "State": "California",
    "Tenure_Months": 24,
    "Specialty": "Cardiology",
    "Insurance_Type": "Private",
    "Visits_Last_Year": 6,
    "Missed_Appointments": 1,
    "Days_Since_Last_Visit": 30,
    "Last_Interaction_Date": "15-01-2025",
    "Overall_Satisfaction": 3.5,
    "Wait_Time_Satisfaction": 3.0,
    "Staff_Satisfaction": 4.0,
    "Provider_Rating": 3.8,
    "Avg_Out_Of_Pocket_Cost": 150.0,
    "Billing_Issues": "No",
    "Portal_Usage": "Medium",
    "Referrals_Made": 2,
    "Distance_To_Facility_Miles": 10.5
}
```

## 📁 Project Structure
```
End_to_end_patient_churn_project/
├── backend/
│   ├── app.py                  # FastAPI application
│   └── schemas.py              # Pydantic models
├── frontend/
│   ├── streamlit_app.py        # Streamlit UI
│   └── requirements.txt
├── src/
│   ├── database/
│   │   ├── db_config.py        # Neon Postgres connection
│   │   ├── create_table.py
│   │   ├── upload_data.py
│   │   └── fetch_data.py
│   └── training/
│       ├── train.py            # Full training pipeline
│       ├── preprocess.py       # sklearn ColumnTransformer
│       ├── feature_engineering.py
│       ├── eda.py
│       └── save_best_model.py
├── tests/
│   └── test_api.py             # Pytest tests (7 tests)
├── models/
│   ├── best_model.joblib
│   └── reference_date.joblib
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/datasource.yml
│       │   └── dashboards/dashboard.yml
│       └── dashboards/fastapi_dashboard.json
├── .github/workflows/
│   ├── backend.yml             # Backend CI/CD
│   └── frontend.yml            # Frontend CI/CD
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── .flake8
├── .gitignore
└── README.md
```

## 🌐 Live URLs
- **FastAPI Backend**: _[Add Render URL after deployment]_
- **Streamlit Frontend**: _[Add Render URL after deployment]_
- **W&B Project**: _[Add W&B project link]_

## 💼 Business Value

Patient churn prediction enables healthcare facilities to:
1. **Proactively identify at-risk patients** before they leave
2. **Optimize resource allocation** by focusing retention efforts
3. **Improve patient satisfaction** through targeted interventions
4. **Reduce revenue loss** from patient attrition
5. **Enhance care continuity** by maintaining patient relationships
