from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
import numpy as np
import pandas as pd
import pickle
import uvicorn
from datetime import datetime
import os
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from pathlib import Path
import uuid
import logging

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "Main Model Train" / "model_saved.pkl"
INDEX_HTML_PATH = BASE_DIR / "index.html"

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(
    title="🏥 Health Prediction System API",
    description="🤖 AI-powered API for predicting diseases based on vital signs and symptoms with MySQL Database",
    version="3.0.0",
    docs_url="/docs",      
    redoc_url="/redoc"
)

# ------------------------------
# CORS Middleware
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               
    allow_credentials=True,           
    allow_methods=["*"],              
    allow_headers=["*"],             
)

# ------------------------------
# Static files & templates
# ------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------------------
# Database configuration
# ------------------------------
class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "1234")
        self.database = os.getenv("DB_NAME", "health_prediction_system")
        self.port = int(os.getenv("DB_PORT", 3306))

    def get_connection(self):
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                autocommit=True
            )
            return conn
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return None

db_config = DatabaseConfig()

# ------------------------------
# Database Manager
# ------------------------------
class DatabaseManager:
    def __init__(self):
        self.config = db_config
        self.init_database()

    def init_database(self):
        """Initialize tables"""
        try:
            connection = self.config.get_connection()
            if not connection:
                logger.error("❌ Could not connect to DB for initialization")
                return
            cursor = connection.cursor()
            # patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id VARCHAR(50) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    total_visits INT DEFAULT 1
                )
            """)
            # health_records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_records (
                    record_id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_id VARCHAR(50),
                    temperature FLOAT,
                    heart_rate FLOAT,
                    bp_sys FLOAT,
                    bp_dia FLOAT,
                    humidity FLOAT,
                    fever BOOLEAN,
                    cough BOOLEAN,
                    chest_pain BOOLEAN,
                    shortness_of_breath BOOLEAN,
                    fatigue BOOLEAN,
                    headache BOOLEAN,
                    predicted_disease VARCHAR(100),
                    confidence FLOAT,
                    record_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                    INDEX idx_patient_date (patient_id, record_date),
                    INDEX idx_disease (predicted_disease)
                )
            """)
            connection.commit()
            cursor.close()
            connection.close()
            logger.info("✅ Database initialized successfully")
        except Error as e:
            logger.error(f"❌ Database initialization failed: {e}")

    @contextmanager
    def get_db_connection(self):
        connection = None
        try:
            connection = self.config.get_connection()
            yield connection
        finally:
            if connection and connection.is_connected():
                connection.close()

    # Create/update patient
    def create_or_update_patient(self, patient_id: str):
        try:
            with self.get_db_connection() as conn:
                if not conn:
                    return False
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE patients
                    SET total_visits = total_visits + 1, last_visit = CURRENT_TIMESTAMP
                    WHERE patient_id = %s
                """, (patient_id,))
                if cursor.rowcount == 0:
                    cursor.execute("INSERT INTO patients (patient_id, total_visits) VALUES (%s, 1)", (patient_id,))
                conn.commit()
                cursor.close()
                return True
        except Error as e:
            logger.error(f"Error creating/updating patient: {e}")
            return False

    # Save health record
    def save_health_record(self, patient_id: str, features: dict, prediction_result: dict):
        try:
            with self.get_db_connection() as conn:
                if not conn:
                    return None
                cursor = conn.cursor()
                self.create_or_update_patient(patient_id)
                query = """
                    INSERT INTO health_records 
                    (patient_id, temperature, heart_rate, bp_sys, bp_dia, humidity,
                     fever, cough, chest_pain, shortness_of_breath, fatigue, headache,
                     predicted_disease, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    patient_id,
                    features['temperature'],
                    features['heart_rate'],
                    features['bp_sys'],
                    features['bp_dia'],
                    features['humidity'],
                    bool(features['fever']),
                    bool(features['cough']),
                    bool(features['chest_pain']),
                    bool(features['shortness_of_breath']),
                    bool(features['fatigue']),
                    bool(features['headache']),
                    prediction_result['disease'],
                    prediction_result['confidence']
                )
                cursor.execute(query, values)
                record_id = cursor.lastrowid
                conn.commit()
                cursor.close()
                return record_id
        except Error as e:
            logger.error(f"Error saving health record: {e}")
            return None

    # Get patient history
    def get_patient_history(self, patient_id: str, limit: int = 50):
        try:
            with self.get_db_connection() as conn:
                if not conn:
                    return None
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT * FROM health_records 
                    WHERE patient_id=%s ORDER BY record_date DESC LIMIT %s
                """, (patient_id, limit))
                records = cursor.fetchall()
                cursor.execute("SELECT * FROM patients WHERE patient_id=%s", (patient_id,))
                patient_info = cursor.fetchone()
                cursor.close()
                return {"patient_info": patient_info, "health_records": records, "total_records": len(records)}
        except Error as e:
            logger.error(f"Error fetching history: {e}")
            return None

db_manager = DatabaseManager()

# ------------------------------
# Models
# ------------------------------
class HealthInput(BaseModel):
    patient_id: Optional[str]
    temperature: float = Field(..., ge=35.0, le=42.0)
    heart_rate: float = Field(..., ge=40.0, le=180.0)
    bp_sys: float = Field(..., ge=70.0, le=200.0)
    bp_dia: float = Field(..., ge=40.0, le=130.0)
    humidity: float = Field(..., ge=20.0, le=80.0)
    fever: int = Field(..., ge=0, le=1)
    cough: int = Field(..., ge=0, le=1)
    chest_pain: int = Field(..., ge=0, le=1)
    shortness_of_breath: int = Field(..., ge=0, le=1)
    fatigue: int = Field(..., ge=0, le=1)
    headache: int = Field(..., ge=0, le=1)

    @field_validator('fever','cough','chest_pain','shortness_of_breath','fatigue','headache')
    @classmethod
    def validate_binary(cls, v):
        if v not in [0,1]:
            raise ValueError("Symptom must be 0 or 1")
        return v

    def dict_for_db(self) -> Dict:
        return self.dict(exclude={"patient_id"})

class PredictionResult(BaseModel):
    patient_id: str
    disease: str
    confidence: float
    probabilities: Dict[str, float]
    suggested_actions: str
    warning: Optional[str]
    timestamp: str
    record_id: Optional[int]
    total_visits: Optional[int]

# ------------------------------
# Model Loader
# ------------------------------
class ModelLoader:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            package = pickle.load(f)
        self.model = package["model"]
        self.scaler = package["scaler"]
        self.label_encoder = package["label_encoder"]
        self.feature_names = package["feature_names"]
        self.numerical_cols = package["numerical_cols"]

try:
    model_loader = ModelLoader()
    MODEL_LOADED = True
    logger.info("✅ Model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    logger.error(f"❌ Failed to load model: {e}")
    model_loader = None

# ------------------------------
# Prediction function
# ------------------------------
def predict_disease(input_data: HealthInput, model_loader: ModelLoader) -> dict:
    features = np.array([[
        input_data.temperature, input_data.heart_rate, input_data.bp_sys, input_data.bp_dia,
        input_data.humidity, input_data.fever, input_data.cough, input_data.chest_pain,
        input_data.shortness_of_breath, input_data.fatigue, input_data.headache
    ]])
    df = pd.DataFrame(features, columns=model_loader.feature_names)
    df[model_loader.numerical_cols] = model_loader.scaler.transform(df[model_loader.numerical_cols])
    proba = model_loader.model.predict(df.values, verbose=0)
    cls = np.argmax(proba)
    disease = model_loader.label_encoder.inverse_transform([cls])[0]
    confidence = float(np.max(proba))
    probabilities = {d: float(p) for d,p in zip(model_loader.label_encoder.classes_, proba[0])}
    suggested_actions = f"Take medical advice for {disease}"
    warning = None
    if confidence < 0.7:
        warning = f"Low confidence: {confidence:.1%}"
    elif confidence < 0.9:
        warning = f"Moderate confidence: {confidence:.1%}"
    return {"disease": disease, "confidence": confidence, "probabilities": probabilities,
            "suggested_actions": suggested_actions, "warning": warning}

# ------------------------------
# API Endpoints
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Health Prediction System</h1><p>Index HTML not found</p>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "🟢 Healthy" if MODEL_LOADED else "🔴 Unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL_LOADED
    }

@app.post("/api/predict", response_model=PredictionResult)
async def make_prediction(input_data: HealthInput):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not input_data.patient_id:
        input_data.patient_id = f"PAT{uuid.uuid4().hex[:8].upper()}"
    result = predict_disease(input_data, model_loader)
    record_id = db_manager.save_health_record(input_data.patient_id, input_data.dict_for_db(), result)
    patient_history = db_manager.get_patient_history(input_data.patient_id)
    total_visits = patient_history["patient_info"]["total_visits"] if patient_history else 1
    return PredictionResult(
        patient_id=input_data.patient_id,
        disease=result["disease"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        suggested_actions=result["suggested_actions"],
        warning=result["warning"],
        timestamp=datetime.now().isoformat(),
        record_id=record_id,
        total_visits=total_visits
    )

@app.get("/api/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    history = db_manager.get_patient_history(patient_id)
    if not history:
        raise HTTPException(status_code=404, detail="Patient not found")
    return history

# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
