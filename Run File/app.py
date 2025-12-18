
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import pickle
import uvicorn
from datetime import datetime
import json
import os


app = FastAPI(
    title="🏥 Health Prediction System API",
    description="🤖 AI-powered API for predicting diseases based on vital signs and symptoms",
    version="2.0.0",
    docs_url="/docs",      
    redoc_url="/redoc",   
    contact={
        "name": "Health Prediction System",
        "url": "http://localhost:8000",
    },
    license_info={
        "name": "Medical Research License",
        "url": "https://opensource.org/licenses/MIT",
    }
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               
    allow_credentials=True,           
    allow_methods=["*"],              
    allow_headers=["*"],             
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 


MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "Main Model Train",
        "model_saved.pkl"
    )
)

class HealthInput(BaseModel):
    """🏥 Input model for health prediction request"""
    temperature: float = Field(
        ..., 
        ge=35.0, 
        le=42.0, 
        description="🌡️ Body temperature in °C (35.0-42.0)",
        example=36.5
    )
    heart_rate: float = Field(
        ..., 
        ge=40.0, 
        le=180.0, 
        description="💓 Heart rate in beats per minute (40-180)",
        example=75.0
    )
    bp_sys: float = Field(
        ..., 
        ge=70.0, 
        le=200.0, 
        description="🩸 Systolic blood pressure in mmHg (70-200)",
        example=120.0
    )
    bp_dia: float = Field(
        ..., 
        ge=40.0, 
        le=130.0, 
        description="📉 Diastolic blood pressure in mmHg (40-130)",
        example=80.0
    )
    humidity: float = Field(
        ..., 
        ge=20.0, 
        le=80.0, 
        description="💧 Environmental humidity in percentage (20-80)",
        example=40.0
    )
    fever: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="🤒 Fever: 0=Absent, 1=Present",
        example=0
    )
    cough: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="😷 Cough: 0=Absent, 1=Present",
        example=0
    )
    chest_pain: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="💔 Chest pain: 0=Absent, 1=Present",
        example=0
    )
    shortness_of_breath: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="😤 Shortness of breath: 0=Absent, 1=Present",
        example=0
    )
    fatigue: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="😴 Fatigue: 0=Absent, 1=Present",
        example=0
    )
    headache: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="🤕 Headache: 0=Absent, 1=Present",
        example=0
    )
    

    @field_validator('fever', 'cough', 'chest_pain', 'shortness_of_breath', 'fatigue', 'headache')
    @classmethod
    def validate_binary(cls, v: int) -> int:
        """✅ Ensure symptom fields are strictly 0 or 1"""
        if v not in [0, 1]:
            raise ValueError('❌ Symptom values must be 0 or 1')
        return v

    class Config:
        """📊 Example for API documentation"""
        json_schema_extra = {
            "example": {
                "temperature": 36.5,
                "heart_rate": 75.0,
                "bp_sys": 120.0,
                "bp_dia": 80.0,
                "humidity": 40.0,
                "fever": 0,
                "cough": 0,
                "chest_pain": 0,
                "shortness_of_breath": 0,
                "fatigue": 0,
                "headache": 0
            }
        }

class PredictionResult(BaseModel):
    """📊 Output model for prediction results"""
    disease: str = Field(..., description="🎯 Predicted disease category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="📈 Prediction confidence score (0.0-1.0)")
    probabilities: Dict[str, float] = Field(..., description="📊 Probability distribution for all diseases")
    suggested_actions: str = Field(..., description="💊 Recommended medical actions and precautions")
    warning: Optional[str] = Field(None, description="⚠️ Warning message for low confidence predictions")
    timestamp: str = Field(..., description="🕒 Prediction timestamp in ISO format")


class ModelLoader:
    """🤖 Singleton class to load and manage the ML model"""
    _instance = None
    
    def __new__(cls):
        """🔧 Ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance
    
    def load_model(self):
        """📦 Load the saved model and preprocessing objects"""
        print("="*60)
        print("🧠 LOADING HEALTH PREDICTION MODEL")
        print("="*60)
        
        model_path_to_try = MODEL_PATH
        print(f"📁 Initial model path: {model_path_to_try}")
        print(f"📂 Current directory: {os.getcwd()}")

   
        if not os.path.exists(model_path_to_try):
            print("❌ Model file not found at initial path")
            
            alt_paths = [
                os.path.join(BASE_DIR, "model_saved.pkl"),
                os.path.join(BASE_DIR, "..", "model_saved.pkl"),
                r"F:\Final Project\Main Model Train\model_saved.pkl",
                r"F:\Final Project\model_saved.pkl",
                "model_saved.pkl",
            ]
            
            for alt_path in alt_paths:
                abs_path = os.path.abspath(alt_path)
                print(f"  🔎 Trying: {abs_path}")
                if os.path.exists(abs_path):
                    print(f"✅ Found model at: {abs_path}")
                    model_path_to_try = abs_path
                    break
            else:
                error_msg = "\n".join([f"  ❌ {p}" for p in [MODEL_PATH] + alt_paths])
                print(f"❌ Model not found at any path:\n{error_msg}")
                raise FileNotFoundError("Model file not found")

        print(f"✅ Loading model from: {model_path_to_try}")
        
       
        with open(model_path_to_try, "rb") as f:
            model_package = pickle.load(f)

   
        self.model = model_package["model"]
        self.scaler = model_package["scaler"]
        self.label_encoder = model_package["label_encoder"]
        self.feature_names = model_package["feature_names"]
        self.numerical_cols = model_package["numerical_cols"]
        self.metadata = model_package["model_metadata"]


        print("✅ Model loaded successfully!")
        print(f"   📊 Accuracy: {self.metadata['accuracy']:.2%}")
        print(f"   🔢 Features: {len(self.feature_names)}")
        print(f"   🏷️  Classes: {list(self.label_encoder.classes_)}")
        print(f"   📅 Training date: {self.metadata['training_date']}")
        print("="*60)
    
    def get_model_info(self) -> Dict[str, Any]:
        """📋 Get comprehensive model metadata"""
        return {
            "model_name": "🏥 Health Disease Predictor",
            "model_type": "🤖 Deep Learning (MLP)",
            "accuracy": f"{self.metadata['accuracy']:.2%}",
            "test_samples": self.metadata['test_samples'],
            "misclassified": self.metadata['misclassified'],
            "error_rate": f"{(self.metadata['misclassified'] / self.metadata['test_samples']) * 100:.2f}%",
            "input_features": len(self.feature_names),
            "output_classes": len(self.label_encoder.classes_),
            "training_date": self.metadata['training_date'],
            "features": self.feature_names,
            "classes": list(self.label_encoder.classes_),
            "model_path": MODEL_PATH,
            "model_size": f"{os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB" if os.path.exists(MODEL_PATH) else "Unknown"
        }


DRUG_MAPPING = {
    "Heart_Risk": {
        "icon": "💔",
        "actions": "Consult cardiologist immediately",
        "medications": "Beta blockers, ACE inhibitors, Aspirin (as prescribed)",
        "precautions": "Avoid strenuous activity, monitor symptoms, emergency contact ready",
        "follow_up": "Cardiology appointment within 24-48 hours",
        "emergency": "If experiencing severe chest pain, call emergency services",
        "severity": "High"
    },
    "Fever_Respiratory": {
        "icon": "🤒",
        "actions": "Rest and monitor symptoms",
        "medications": "Paracetamol for fever, cough syrup if needed",
        "precautions": "Stay hydrated, isolate if contagious, monitor temperature",
        "follow_up": "Consult physician if symptoms persist >3 days or worsen",
        "emergency": "If having difficulty breathing, seek immediate medical help",
        "severity": "Medium"
    },
    "Hypertension": {
        "icon": "📈",
        "actions": "Monitor blood pressure regularly",
        "medications": "Amlodipine, Lisinopril (as prescribed)",
        "precautions": "Reduce salt intake, regular exercise, stress management",
        "follow_up": "Regular BP monitoring, annual checkup",
        "emergency": "If BP >180/120 with symptoms, seek emergency care",
        "severity": "Medium"
    },
    "Hypotension": {
        "icon": "📉",
        "actions": "Increase fluid intake",
        "medications": "Usually none, consult doctor for persistent cases",
        "precautions": "Rise slowly from sitting, increase salt intake (if advised)",
        "follow_up": "Monitor for dizziness, follow up if symptoms worsen",
        "emergency": "If fainting occurs, seek medical attention",
        "severity": "Low"
    },
    "Normal": {
        "icon": "✅",
        "actions": "Maintain healthy lifestyle",
        "medications": "None required",
        "precautions": "Regular exercise, balanced diet, routine checkups",
        "follow_up": "Annual health checkup recommended",
        "emergency": "None",
        "severity": "None"
    }
}

def get_suggested_actions(disease: str, confidence: float) -> str:
    """💡 Generate suggested medical actions based on disease and confidence"""
    if disease not in DRUG_MAPPING:
        return "Consult healthcare professional for diagnosis"
    
    info = DRUG_MAPPING[disease]
    
   
    warning = ""
    if confidence < 0.7:
        warning = "\n\n⚠️ **LOW CONFIDENCE WARNING** - Prediction confidence is low. Please consult a doctor for proper diagnosis."
    elif confidence < 0.9:
        warning = "\n\n⚠️ **MODERATE CONFIDENCE** - Medical consultation is recommended for confirmation."
    

    actions = f"""
{info['icon']} **{disease.replace('_', ' ').upper()}** ({info['severity']} Severity)

📋 **IMMEDIATE ACTIONS:**
   • {info['actions']}

💊 **POSSIBLE MEDICATIONS:**
   • {info['medications']}

🛡️ **PRECAUTIONS:**
   • {info['precautions']}

📅 **FOLLOW-UP:**
   • {info['follow_up']}

🚨 **EMERGENCY:**
   • {info['emergency']}
{warning}

---
**📢 MEDICAL DISCLAIMER:**
This is an AI-powered prediction tool, not a medical diagnosis. 
Always consult with a qualified healthcare professional for medical advice and treatment.
Results are based on statistical patterns and may not be accurate for all individuals.
    """
    
    return actions.strip()


def predict_disease(input_data: HealthInput, model_loader: ModelLoader) -> Dict[str, Any]:
    """
    🤖 Make disease prediction from input features
    
    Args:
        input_data: Validated health input data
        model_loader: Loaded ML model and preprocessors
    
    Returns:
        Dictionary containing prediction results
    """

    features = [
        input_data.temperature,
        input_data.heart_rate,
        input_data.bp_sys,
        input_data.bp_dia,
        input_data.humidity,
        input_data.fever,
        input_data.cough,
        input_data.chest_pain,
        input_data.shortness_of_breath,
        input_data.fatigue,
        input_data.headache
    ]
    
  
    features_array = np.array(features).reshape(1, -1)
    features_df = pd.DataFrame(features_array, columns=model_loader.feature_names)
    

    features_df[model_loader.numerical_cols] = model_loader.scaler.transform(
        features_df[model_loader.numerical_cols]
    )
    
 
    prediction_proba = model_loader.model.predict(features_df.values, verbose=0)
    prediction_class = np.argmax(prediction_proba)
    

    predicted_disease = model_loader.label_encoder.inverse_transform([prediction_class])[0]
    confidence = float(np.max(prediction_proba))
    

    probabilities = {
        disease: float(prob)
        for disease, prob in zip(model_loader.label_encoder.classes_, prediction_proba[0])
    }
    

    suggested_actions = get_suggested_actions(predicted_disease, confidence)
    
  
    warning = None
    if confidence < 0.7:
        warning = f"Low prediction confidence ({confidence:.1%}). Please consult a healthcare professional."
    elif confidence < 0.9:
        warning = f"Moderate prediction confidence ({confidence:.1%}). Medical consultation recommended."
    
    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "probabilities": probabilities,
        "suggested_actions": suggested_actions,
        "warning": warning,
        "timestamp": datetime.now().isoformat()
    }


try:
    model_loader = ModelLoader()
    MODEL_LOADED = True
    print("✅ MODEL INITIALIZATION COMPLETE")
except Exception as e:
    print(f"❌ FAILED TO LOAD MODEL: {e}")
    model_loader = None
    MODEL_LOADED = False



@app.get("/", tags=["🏠 Root"])
async def root():
    """🏠 Root endpoint - API information and navigation"""
    return {
        "message": "🏥 Welcome to Health Prediction System API",
        "version": "2.0.0",
        "status": "🟢 Active" if MODEL_LOADED else "🔴 Model Not Loaded",
        "model_loaded": MODEL_LOADED,
        "model_accuracy": model_loader.metadata['accuracy'] if MODEL_LOADED else None,
        "quick_links": {
            "📚 Documentation": "/docs",
            "❤️ Health Check": "/health",
            "🤖 Model Info": "/model-info",
            "🔮 Predict": "/predict",
            "🌐 Frontend": "Open index.html in browser"
        },
        "note": "🚀 Powered by FastAPI & TensorFlow | 🎯 99.03% Accuracy"
    }

@app.get("/health", tags=["🩺 Health"])
async def health_check():
    """🩺 Health check endpoint - API status"""
    return {
        "status": "🟢 Healthy" if MODEL_LOADED else "🔴 Unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL_LOADED,
        "model_accuracy": model_loader.metadata['accuracy'] if MODEL_LOADED else None,
        "api_uptime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cors_enabled": True,
        "frontend_support": True,
        "server_time": datetime.now().strftime("%I:%M %p")
    }

@app.get("/model-info", tags=["🤖 Model"])
async def get_model_info():
    """🤖 Get comprehensive model metadata and information"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="❌ Model not loaded. Check if model_saved.pkl exists.")
    return model_loader.get_model_info()

@app.get("/features", tags=["📊 Features"])
async def get_features():
    """📊 Get list of input features with descriptions"""
    features_info = {
        "temperature": "🌡️ Body temperature in °C (35.0-42.0)",
        "heart_rate": "💓 Heart rate in bpm (40-180)",
        "bp_sys": "🩸 Systolic blood pressure in mmHg (70-200)",
        "bp_dia": "📉 Diastolic blood pressure in mmHg (40-130)",
        "humidity": "💧 Environmental humidity in % (20-80)",
        "fever": "🤒 Presence of fever (0=No, 1=Yes)",
        "cough": "😷 Presence of cough (0=No, 1=Yes)",
        "chest_pain": "💔 Presence of chest pain (0=No, 1=Yes)",
        "shortness_of_breath": "😤 Shortness of breath (0=No, 1=Yes)",
        "fatigue": "😴 Presence of fatigue (0=No, 1=Yes)",
        "headache": "🤕 Presence of headache (0=No, 1=Yes)"
    }
    return {
        "features": features_info,
        "count": len(features_info),
        "numerical_features": 5,
        "categorical_features": 6,
        "total_features": 11
    }

@app.get("/diseases", tags=["🏥 Diseases"])
async def get_diseases():
    """🏥 Get list of supported diseases with descriptions"""
    diseases_info = {
        "Heart_Risk": "💔 Potential cardiovascular issues requiring attention",
        "Fever_Respiratory": "🤒 Fever and respiratory infection symptoms",
        "Hypertension": "📈 High blood pressure condition",
        "Hypotension": "📉 Low blood pressure condition",
        "Normal": "✅ Within normal health parameters"
    }
    return {
        "diseases": diseases_info,
        "count": len(diseases_info),
        "classes": list(diseases_info.keys()),
        "note": "All predictions include confidence scores and medical advice"
    }

@app.post("/predict", response_model=PredictionResult, tags=["🔮 Prediction"])
async def make_prediction(input_data: HealthInput):
    """
    🔮 Make disease prediction based on vital signs and symptoms
    
    ## 📝 Example Input:
    ```json
    {
        "temperature": 37.0,
        "heart_rate": 85.0,
        "bp_sys": 120.0,
        "bp_dia": 80.0,
        "humidity": 40.0,
        "fever": 0,
        "cough": 0,
        "chest_pain": 0,
        "shortness_of_breath": 0,
        "fatigue": 0,
        "headache": 0
    }
    ```
    
    ## 🎯 Returns:
    - **Disease**: Predicted category
    - **Confidence**: Prediction confidence (0-1)
    - **Probabilities**: Distribution for all diseases
    - **Suggested Actions**: Medical advice
    - **Warning**: Confidence alerts
    - **Timestamp**: Prediction time
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503, 
            detail="❌ Model not loaded. Please ensure model_saved.pkl is available."
        )
    
    try:
        # 📊 LOG REQUEST
        print(f"📥 Prediction request: {input_data.dict()}")
        
        # 🔮 MAKE PREDICTION
        result = predict_disease(input_data, model_loader)
        
        # ✅ LOG SUCCESS
        print(f"✅ Prediction: {result['disease']} ({result['confidence']:.1%} confidence)")
        
        return result
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"❌ Prediction error: {str(e)}")

@app.post("/batch-predict", tags=["📦 Batch"])
async def batch_predict(inputs: List[HealthInput]):
    """
    📦 Make predictions for multiple inputs at once
    
    Useful for:
    - 📊 Batch processing
    - 📈 Data analysis
    - 🧪 Testing multiple scenarios
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="❌ Model not loaded")
    
    try:
        results = []
        for i, input_data in enumerate(inputs):
            result = predict_disease(input_data, model_loader)
            results.append(result)
        
        print(f"📦 Batch prediction: {len(results)} records")
        
        return {
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat(),
            "average_confidence": f"{np.mean([r['confidence'] for r in results]):.1%}" if results else "0%",
            "disease_distribution": {
                disease: sum(1 for r in results if r['disease'] == disease)
                for disease in model_loader.label_encoder.classes_
            }
        }
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"❌ Batch prediction error: {str(e)}")

@app.get("/example", tags=["🧪 Examples"])
async def get_example():
    """🧪 Get example inputs for testing"""
    examples = {
        "normal": {
            "temperature": 36.5,
            "heart_rate": 75.0,
            "bp_sys": 120.0,
            "bp_dia": 80.0,
            "humidity": 40.0,
            "fever": 0,
            "cough": 0,
            "chest_pain": 0,
            "shortness_of_breath": 0,
            "fatigue": 0,
            "headache": 0,
            "description": "✅ Normal healthy person",
            "expected": "Normal"
        },
        "heart_risk": {
            "temperature": 37.0,
            "heart_rate": 105.0,
            "bp_sys": 150.0,
            "bp_dia": 95.0,
            "humidity": 40.0,
            "fever": 0,
            "cough": 0,
            "chest_pain": 1,
            "shortness_of_breath": 1,
            "fatigue": 1,
            "headache": 0,
            "description": "💔 Heart risk patient",
            "expected": "Heart_Risk"
        },
        "fever_respiratory": {
            "temperature": 39.0,
            "heart_rate": 110.0,
            "bp_sys": 115.0,
            "bp_dia": 75.0,
            "humidity": 40.0,
            "fever": 1,
            "cough": 1,
            "chest_pain": 0,
            "shortness_of_breath": 0,
            "fatigue": 1,
            "headache": 1,
            "description": "🤒 Fever/Respiratory infection",
            "expected": "Fever_Respiratory"
        }
    }
    return {
        "examples": examples,
        "note": "🧪 Use these examples to test the /predict endpoint",
        "total_examples": len(examples),
        "quick_test": "Copy any example JSON and paste in /predict endpoint"
    }

@app.get("/system-status", tags=["⚙️ System"])
async def system_status():
    """⚙️ Get complete system status and diagnostics"""
    return {
        "api": {
            "status": "🟢 Running" if MODEL_LOADED else "🔴 Stopped",
            "version": "2.0.0",
            "uptime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cors_enabled": True
        },
        "model": {
            "loaded": MODEL_LOADED,
            "accuracy": model_loader.metadata['accuracy'] if MODEL_LOADED else None,
            "features": len(model_loader.feature_names) if MODEL_LOADED else 0,
            "classes": len(model_loader.label_encoder.classes_) if MODEL_LOADED else 0,
            "path": MODEL_PATH
        },
        "frontend": {
            "compatible": True,
            "recommendation": "Open index.html in browser",
            "endpoint": "http://localhost:8000"
        },
        "server": {
            "time": datetime.now().strftime("%I:%M %p"),
            "date": datetime.now().strftime("%B %d, %Y"),
            "timezone": "UTC"
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """🛡️ Handle HTTP exceptions with user-friendly messages"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "suggestion": "Check API documentation at /docs",
            "icon": "❌"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """🛡️ Handle unexpected exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
            "suggestion": "Contact system administrator",
            "icon": "🔥"
        }
    )


if __name__ == "__main__":
    print("="*60)
    print(" HEALTH PREDICTION SYSTEM API v2.0.0")
    print("="*60)
    
    if MODEL_LOADED:
        print(f" MODEL STATUS: LOADED")
        print(f"    Accuracy: {model_loader.metadata['accuracy']:.2%}")
        print(f"    Features: {len(model_loader.feature_names)}")
        print(f"     Classes: {list(model_loader.label_encoder.classes_)}")
        print(f"    Model path: {MODEL_PATH}")
    else:
        print(f" MODEL STATUS: FAILED")
        print(f"    Expected: {MODEL_PATH}")
        print("\n    SOLUTIONS:")
        print("   1.  Copy model_saved.pkl from 'Main Model Train' to 'Run File'")
        print("   2.  Verify file exists at above path")
        print("   3.  Visit /debug/files to check locations")
    
    print(f"\n API SERVER STARTING...")
    print(f"    Documentation: http://localhost:8000/docs")
    print(f"    Health check: http://localhost:8000/health")
    print(f"    Frontend: Open index.html in browser")
    print(f"    CORS: Enabled ✓")
    print(f"    Performance: Optimized")
    
    print(f"\n    READY FOR PREDICTIONS!")
    print(f"    Press CTRL+C to stop server")
    print("="*60)
    
    
    uvicorn.run(
        app,
        host="0.0.0.0",   
        port=8000,       
        reload=False       
    )