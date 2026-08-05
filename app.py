# ==============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# ==============================================================================
import joblib
import numpy as np
from datetime import date
from typing import Annotated, Literal

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import custom processing functions from the local 'function.py' module
from function import prepare_data, predict, explain

# ==============================================================================
# 2. LIFESPAN CONTEXT MANAGER (SERVER STARTUP & SHUTDOWN)
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Loads machine learning models and background data into memory before accepting requests.
    Clears memory gracefully upon server shutdown.
    """
    print(f"\n{'=' * 70}")
    print("🚀 [SYSTEM BOOT] Initializing FastAPI Server...")
    print(f"{'=' * 70}")
    
    try:
        # ---------------------------------------------------------
        # PHASE 1: LOAD ASSETS INTO MEMORY
        # ---------------------------------------------------------
        # Double safety check: Verify app has 'state' before injecting
        if hasattr(app, "state"):
            if not hasattr(app.state, "best_model"):
                print("📦 [LOAD] Loading trained Machine Learning model (best_hp_model.joblib)...")
                app.state.best_model = joblib.load("Maternal-Health-Risk-Model/best_hp_model.joblib")

            if not hasattr(app.state, "lime_training_data"):
                print("📊 [LOAD] Loading background training data for LIME (lime_training_data.npy)...")
                app.state.lime_training_data = np.load("Maternal-Health-Risk-Model/lime_training_data.npy")

            if not hasattr(app.state, "feature_names"):
                print("📝 [LOAD] Loading feature names metadata (feature_names.joblib)...")
                app.state.feature_names = joblib.load("Maternal-Health-Risk-Model/feature_names.joblib")

        print("✅ [SYSTEM] All ML assets loaded successfully. Server is ready to accept traffic!\n")
        
        # ---------------------------------------------------------
        # PHASE 2: YIELD TO FASTAPI (SERVER RUNNING)
        # ---------------------------------------------------------
        # Yield is empty because assets are injected directly into app.state
        yield  

    # ---------------------------------------------------------
    # PHASE 3: EXCEPTION HANDLING & ERROR ROUTING (BOOT PHASE)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        print("\n" + "="*70)
        print("💥 [CRITICAL BOOT FAILURE] Server failed to start!")
        print("-" * 70)

        # 1. Handling Missing Files (Wrong Path / File Deleted)
        if error_type == "FileNotFoundError" or "no such file" in error_msg:
            print(f"🚨 [FILE ERROR] {error_type}: A required model or data file is missing. Details: {error_raw}")
            print("="*70 + "\n")
            raise RuntimeError(f"[FILE ERROR] {error_type}: Could not find the required file. Please check the file path. Details: {error_raw}")

        # 2. Handling Corrupted Joblib/Numpy files
        elif error_type == "ValueError" or "unpickling" in error_msg:
            print(f"🚨 [LOAD ERROR] {error_type}: Failed to load file. Corrupted joblib/npy. Details: {error_raw}")
            print("="*70 + "\n")
            raise RuntimeError(f"[LOAD ERROR] {error_type}: The model or data file is corrupted or incompatible. Details: {error_raw}")
            
        # 3. Handling Environment/Dependency Issues (e.g., Scikit-learn version mismatch)
        elif error_type == "ModuleNotFoundError" or "module" in error_msg:
            print(f"🚨 [ENV ERROR] {error_type}: Missing Python package required by the model. Details: {error_raw}")
            print("="*70 + "\n")
            raise RuntimeError(f"[ENV ERROR] {error_type}: A required library is missing from your environment. Details: {error_raw}")

        # 4. Fallback for any other unknown boot errors
        else:
            print(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during server boot. Details: {error_raw}")
            print("="*70 + "\n")
            raise RuntimeError(f"[UNKNOWN ERROR] {error_type}: An unexpected error occurred while loading assets. Details: {error_raw}")

    # ---------------------------------------------------------
    # PHASE 4: SHUTDOWN CLEANUP
    # ---------------------------------------------------------
    finally:
        print(f"\n{'=' * 70}")
        print("🛑 [SYSTEM SHUTDOWN] Cleaning up memory and stopping server...")
        
        # Safely delete attributes from app.state to free up RAM
        if hasattr(app, "state"):
            if hasattr(app.state, "best_model"):
                del app.state.best_model
            if hasattr(app.state, "lime_training_data"):
                del app.state.lime_training_data
            if hasattr(app.state, "feature_names"):
                del app.state.feature_names
                
        print("✅ [SYSTEM] Memory cleared. Server safely stopped.")
        print(f"{'=' * 70}\n")

# ==============================================================================
# 3. FASTAPI APPLICATION INITIALIZATION
# ==============================================================================
app = FastAPI(
    title="Maternal Health Risk Prediction API",
    version="1.0.1",
    description="""
    An AI-powered diagnostic API designed to predict maternal health risks during pregnancy.
    Integrates a hyperparameter-tuned CatBoost model with LIME Explainable AI for transparent medical insights.
    """,
    lifespan=lifespan  # Attaching the lifespan manager we built earlier
)

# ==============================================================================
# 4. CORS MIDDLEWARE CONFIGURATION
# ==============================================================================
# Ensures the API can receive requests from front-end web applications (React, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Note: For production, replace "*" with specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==============================================================================
# 5. INPUT VALIDATION SCHEMA (PYDANTIC)
# ==============================================================================
class ConditionalInput(BaseModel):
    """
    Data validation schema for patient health metrics.
    Automatically rejects out-of-bound inputs and returns a 422 HTTP Error.
    """
    age: int = Field(
        default=25, 
        ge=10, 
        le=70, 
        description="Patient's age in years (Range: 10 - 70)."
    )
    systolic_bp: int = Field(
        default=120, 
        ge=70, 
        le=160, 
        description="Upper blood pressure metric in mmHg (Range: 70 - 160)."
    )
    diastolic_bp: int = Field(
        default=80, 
        ge=49, 
        le=100, 
        description="Lower blood pressure metric in mmHg (Range: 49 - 100)."
    )
    blood_glucose: float = Field(
        default=7.5, 
        ge=6.0, 
        le=19.0, 
        description="Blood sugar level in mmol/L (Range: 6.0 - 19.0)."
    )
    body_temp: float = Field(
        default=98.6, 
        ge=98.0, 
        le=103.0, 
        description="Core body temperature in Fahrenheit (Range: 98.0 - 103.0)."
    )
    heart_rate: int = Field(
        default=75, 
        ge=60,
        le=90, 
        description="Resting heart rate in beats per minute (Range: 60 - 90)."
    )

# ==============================================================================
# 6. ROOT ENDPOINT (API METADATA & DOCUMENTATION)
# ==============================================================================
@app.get("/")
async def home() -> dict:
    """
    Root endpoint: Provides server health status, API metadata, and detailed usage documentation.
    Serves as a friendly landing page for developers integrating this Maternal Health Risk API.
    """
    print("🌐 [API] Root endpoint accessed. Serving metadata and documentation.")
    
    return {
        "status": "✅ Online",
        "service": "Maternal Health Risk Prediction & LIME Explanation API",
        "version": "1.0.1",
        "live_urls": {
            # [NOTE]: The URLs provided below are purely illustrative examples.
            # They do not reflect the actual live server environment.
            "base_url": "https://silvio0-maternal-health-api.hf.space",
            "documentation": "https://silvio0-maternal-health-api.hf.space/docs",
            "prediction_endpoint": "https://silvio0-maternal-health-api.hf.space/predict",
            "explanation_endpoint": "https://silvio0-maternal-health-api.hf.space/explain"
        },
        "usage_guide": {
            "endpoints": {
                "/predict": "POST method - Predicts maternal health risk (low, mid, high risk) based on patient health metrics.",
                "/explain": "POST method - Generates an interactive LIME HTML explanation detailing how the model made its prediction."
            },
            "payload_structure": {
                "age": "integer (Required) - Range: 10 to 70.",
                "systolic_bp": "integer (Required) - Range: 70 to 160.",
                "diastolic_bp": "integer (Required) - Range: 49 to 100.",
                "blood_glucose": "float (Required) - Range: 6.0 to 19.0.",
                "body_temp": "float (Required) - Range: 98.0 to 103.0.",
                "heart_rate": "integer (Required) - Range: 60 to 90."
            },
            "payload_example": {
                "age": 25,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "blood_glucose": 7.5,
                "body_temp": 98.6,
                "heart_rate": 75
            }
        },
        "author": "Silvio Christian Joe"
    }

# ==============================================================================
# 7. PREDICTION ENDPOINT (INFERENCE API)
# ==============================================================================
@app.post("/predict")
async def predict_maternal_risk(request: Request, form_data: Annotated[ConditionalInput, Form()]):
    """
    Receives patient health metrics via Form Data, processes them, 
    and returns the predicted maternal health risk along with confidence scores.
    """
    print("\n" + "="*70)
    print("🌐 [API REQUEST] Incoming prediction request at POST /predict")
    print("-" * 70)

    try: 
        # Step 1: Retrieve the pre-loaded ML model from the application state (Zero Lag)
        best_model = request.app.state.best_model

        # Step 2: Structure the raw form inputs into a Pandas DataFrame
        df_testing = prepare_data(
            age=form_data.age,
            systolic_bp=form_data.systolic_bp,
            diastolic_bp=form_data.diastolic_bp,
            blood_glucose=form_data.blood_glucose,
            body_temp=form_data.body_temp,
            heart_rate=form_data.heart_rate
        )

        # Step 3: Execute the inference function using the loaded model
        # Note: Calling the imported 'predict' function from function.py
        prediction, prediction_conf, low_risk_score, mid_risk_score, high_risk_score = predict(
            best_model=best_model, 
            df_testing=df_testing
        )

        # Step 4: Return the formatted JSON response to the client
        print("✅ [API RESPONSE] Prediction successfully delivered to client.")
        return {
            "prediction": prediction,
            "prediction_conf": prediction_conf,
            "low_risk_score": low_risk_score,
            "mid_risk_score": mid_risk_score,
            "high_risk_score": high_risk_score
        }

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (ENDPOINT LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        print("\n" + "="*70)
        print("💥 [CRITICAL FAILURE] API Request aborted during /predict endpoint!")
        print("-" * 70)

        # 1. Handling Missing Columns/Features
        if error_type == "KeyError" or "key" in error_msg:
            print(f"🚨 [DATA ERROR] {error_type}: Missing required feature/column. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=400, detail=f"[DATA ERROR] {error_type}: A required data field is missing from the processing pipeline. Details: {error_raw}")

        # 2. Handling Data Type Mismatches
        elif error_type == "TypeError" or "type" in error_msg:
            print(f"🚨 [DATA ERROR] {error_type}: Incompatible data type encountered. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=400, detail=f"[DATA ERROR] {error_type}: Incorrect data type passed to the processing function. Details: {error_raw}")

        # 3. Handling Value/Shape Mismatch
        elif error_type == "ValueError" or "value" in error_msg or "shape" in error_msg:
            print(f"🚨 [DATA ERROR] {error_type}: Input data shape or value mismatch. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=422, detail=f"[DATA ERROR] {error_type}: The input data shape or value does not match the model's requirements. Details: {error_raw}")

        # 4. Handling Application State / Lifespan Failures
        elif error_type == "AttributeError" and ("state" in error_msg or "best_model" in error_msg):
            print(f"🚨 [SYSTEM ERROR] {error_type}: Model not found in app state. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[SYSTEM ERROR] {error_type}: Model not found in application state. Lifespan boot may have failed. Details: {error_raw}")

        # 5. Handling Corrupted Model Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            print(f"🚨 [SYSTEM ERROR] {error_type}: Model object is corrupted. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[SYSTEM ERROR] {error_type}: Internal model architecture error during prediction. Details: {error_raw}")

        # 6. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            print(f"🚨 [MODEL ERROR] {error_type}: Attempting to predict with an untrained model. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[MODEL ERROR] {error_type}: The loaded machine learning model is not trained. Details: {error_raw}")

        # 7. Fallback for any other unknown errors
        else:
            print(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during execution. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[UNKNOWN ERROR] {error_type}: An unexpected server error occurred during prediction. Details: {error_raw}")

# ==============================================================================
# 8. EXPLAINABILITY ENDPOINT (XAI API)
# ==============================================================================
@app.post("/explain")
async def generate_explanation(request: Request, form_data: Annotated[ConditionalInput, Form()]):
    """
    Receives patient health metrics via Form Data, processes them, 
    and returns a LIME HTML explanation detailing how the model made its prediction.
    """
    print("\n" + "="*70)
    print("🌐 [API REQUEST] Incoming explanation request at POST /explain")
    print("-" * 70)

    try:
        # Step 1: Retrieve the pre-loaded assets from the application state (Zero Lag)
        best_model = request.app.state.best_model
        lime_training_data = request.app.state.lime_training_data
        feature_names = request.app.state.feature_names

        # Step 2: Structure the raw form inputs into a Pandas DataFrame
        df_testing = prepare_data(
            age=form_data.age,
            systolic_bp=form_data.systolic_bp,
            diastolic_bp=form_data.diastolic_bp,
            blood_glucose=form_data.blood_glucose,
            body_temp=form_data.body_temp,
            heart_rate=form_data.heart_rate
        )

        # Step 3: Execute the LIME explainer function using the loaded model and data
        # Note: Calling the imported 'explain' function from function.py
        explanation_html = explain(
            best_model=best_model, 
            feature_names=feature_names,
            X_train_processed=lime_training_data, 
            df_testing=df_testing
        )

        # Step 4: Return the generated HTML to the client
        print("✅ [API RESPONSE] LIME HTML explanation successfully delivered to client.")
        return {
            "explanation_html": explanation_html
        }

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (ENDPOINT LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        print("\n" + "="*70)
        print("💥 [CRITICAL FAILURE] API Request aborted during /explain endpoint!")
        print("-" * 70)

        # 1. Handling Missing Columns/Features
        if error_type == "KeyError" or "key" in error_msg:
            print(f"🚨 [DATA ERROR] {error_type}: Missing required feature/column. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=400, detail=f"[DATA ERROR] {error_type}: A required data field is missing from the processing pipeline. Details: {error_raw}")

        # 2. Handling Data Type Mismatches
        elif error_type == "TypeError" or "type" in error_msg:
            print(f"🚨 [DATA ERROR] {error_type}: Incompatible data type encountered. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=400, detail=f"[DATA ERROR] {error_type}: Incorrect data type passed to the processing function. Details: {error_raw}")

        # 3. Handling LIME Data Mismatch (Crucial for Explainable AI)
        elif error_type == "ValueError" or "value" in error_msg or "shape" in error_msg:
            print(f"🚨 [LIME ERROR] {error_type}: Background training data mismatch. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=422, detail=f"[LIME ERROR] {error_type}: The input data shape does not match the LIME background dataset. Details: {error_raw}")

        # 4. Handling Application State / Lifespan Failures
        elif error_type == "AttributeError" and ("state" in error_msg or "best_model" in error_msg or "lime_training_data" in error_msg):
            print(f"🚨 [SYSTEM ERROR] {error_type}: Assets not found in app state. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[SYSTEM ERROR] {error_type}: Assets not found in application state. Lifespan boot may have failed. Details: {error_raw}")

        # 5. Handling Corrupted Model/Explainer Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            print(f"🚨 [SYSTEM ERROR] {error_type}: Model or Explainer object is corrupted. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[SYSTEM ERROR] {error_type}: Internal model architecture error during explanation generation. Details: {error_raw}")

        # 6. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            print(f"🚨 [MODEL ERROR] {error_type}: Attempting to explain an untrained model. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[MODEL ERROR] {error_type}: The loaded machine learning model is not trained. Details: {error_raw}")

        # 7. Fallback for any other unknown errors
        else:
            print(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during execution. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[UNKNOWN ERROR] {error_type}: An unexpected server error occurred during LIME generation. Details: {error_raw}")
