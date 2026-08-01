# ==============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# ==============================================================================
import numpy as np
import pandas as pd
import scipy
import lime
import lime.lime_tabular
from datetime import date
from fastapi import HTTPException

# ==============================================================================
# 2. GLOBAL CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
TARGET_LABELS = ["low risk", "mid risk", "high risk"]
num_columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
cat_columns = ["RiskLevel"]

# ==============================================================================
# 3. DATA PREPARATION FUNCTIONS (FEATURE ENGINEERING)
# ==============================================================================
def prepare_data(
    age: int,
    systolic_bp: int,
    diastolic_bp: int,
    blood_glucose: float,
    body_temp: float,
    heart_rate: int
) -> pd.DataFrame:
    """
    Transforms raw patient health input parameters into a structured Pandas DataFrame.
    Prepares the data exactly as required by the preprocessing pipeline.
    """
    print("⏳ [FEATURE ENG] Structuring raw patient health inputs...")
    
    # Wrap scalars in lists to correctly construct a single-row DataFrame
    df_testing = pd.DataFrame({
        "Age": [age],
        "SystolicBP": [systolic_bp],
        "DiastolicBP": [diastolic_bp],
        "BS": [blood_glucose],
        "BodyTemp": [body_temp],
        "HeartRate": [heart_rate]
    })
    
    print(f"✅ [FEATURE ENG] Patient DataFrame ready. Shape: {df_testing.shape}")
    return df_testing

# ==============================================================================
# 4. INFERENCE FUNCTION (PREDICTION)
# ==============================================================================
def predict(best_model, df_testing: pd.DataFrame, target_labels: list = TARGET_LABELS) -> tuple:
    """
    Executes model inference on the prepared DataFrame.
    Extracts the predicted class label, overall confidence, and exact probabilities for all 3 risk levels.
    """
    print("🧠 [INFERENCE] Executing multiclass model prediction...")
    
    try:
        # Generate predictions and probabilities using the loaded model
        y_pred = best_model.predict(df_testing)
        y_pred_proba = best_model.predict_proba(df_testing)
        
        # Bulletproof flattening to safely handle both 1D (XGBoost) and 2D (CatBoost) arrays
        y_pred_flat = np.array(y_pred).flatten()
        
        # Extract specific values for the single patient instance (index 0)
        pred_index = int(y_pred_flat[0])
        prediction = target_labels[pred_index]

        # Extract the highest probability (confidence score) and format it as a percentage string (e.g., "95.5%")
        prediction_conf = ((y_pred_proba.max(axis=1) * 100).round(2).astype(str) + "%")[0]
        
        # Extract the raw floating-point probabilities for each specific risk level class for granular UI display
        low_risk_score  = y_pred_proba[:, 0].round(2).astype(float)[0]
        mid_risk_score  = y_pred_proba[:, 1].round(2).astype(float)[0]
        high_risk_score = y_pred_proba[:, 2].round(2).astype(float)[0]

        # Log the extracted metrics to the terminal for observability
        print(f"🎯 [INFERENCE] Result: {prediction.upper()} | Conf: {prediction_conf}")
        print(f"   ↳ [PROBABILITIES] Low: {low_risk_score}, Mid: {mid_risk_score}, High: {high_risk_score}")
        
        return prediction, prediction_conf, low_risk_score, mid_risk_score, high_risk_score

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (API LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        print("\n" + "="*70)
        print("💥 [CRITICAL FAILURE] API Request aborted during Model Prediction!")
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

        # 4. Handling Corrupted Model Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            print(f"🚨 [SYSTEM ERROR] {error_type}: Model object is corrupted. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[SYSTEM ERROR] {error_type}: Internal model architecture error during prediction. Details: {error_raw}")

        # 5. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            print(f"🚨 [MODEL ERROR] {error_type}: Attempting to predict with an untrained model. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[MODEL ERROR] {error_type}: The loaded machine learning model is not trained. Details: {error_raw}")

        # 6. Fallback for any other unknown errors
        else:
            print(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during execution. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[UNKNOWN ERROR] {error_type}: An unexpected server error occurred during prediction. Details: {error_raw}")

# ==============================================================================
# 5. EXPLAINABILITY FUNCTION (LIME)
# ==============================================================================
def explain(best_model, feature_names, X_train_processed: np.ndarray, df_testing: pd.DataFrame, target_labels: list = TARGET_LABELS, random_seed: int = RANDOM_SEED) -> str:
    """
    Generates a LIME (Local Interpretable Model-agnostic Explanations) HTML output.
    Bypasses background data transformation as X_train_processed is already encoded.
    """
    print("🔍 [XAI] Initializing LIME Explainer...")
    
    try:
        # Extract the preprocessing step and the machine learning model from the pipeline
        preprocessor = best_model.named_steps["Preprocessing"]
        ml_model = best_model.named_steps["Model"]

        print("🧮 [XAI] Loading pre-processed background training data...")
        
        # Ensure the background data is a dense array for LIME compatibility
        if scipy.sparse.issparse(X_train_processed):
            X_train_processed = X_train_processed.toarray()

        # Use the injected feature names metadata to accurately label the LIME output chart
        features = feature_names
        
        # Initialize the Tabular Explainer with the pre-processed background data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_processed,
            feature_names=features,
            class_names=target_labels,
            mode="classification",
            random_state=random_seed
        )

        print("📊 [XAI] Processing single patient instance for explanation...")
        
        # Isolate the single row to explain and apply the preprocessing pipeline
        status_raw = df_testing.iloc[[0]]
        status_processed = preprocessor.transform(status_raw)

        # Convert the processed single instance to a dense array if necessary
        if scipy.sparse.issparse(status_processed):
            status_processed = status_processed.toarray()

        # Flatten to a 1D array as required by LIME's explain_instance method
        status_data_1d = status_processed[0]

        print("🧩 [XAI] Generating instance explanation (Top 10 features)...")
        
        # Generate the explanation using ONLY the isolated ML model's predict_proba method
        explanation = explainer.explain_instance(
            data_row=status_data_1d,
            predict_fn=ml_model.predict_proba,
            num_features=10,
            top_labels=len(target_labels)
        )
        
        print("✅ [XAI] LIME HTML generation complete.")
        return explanation.as_html()

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (API LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        print("\n" + "="*70)
        print("💥 [CRITICAL FAILURE] API Request aborted during LIME explanation!")
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

        # 4. Handling Corrupted Model/Explainer Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            print(f"🚨 [SYSTEM ERROR] {error_type}: Model or Explainer object is corrupted. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[SYSTEM ERROR] {error_type}: Internal model architecture error during explanation generation. Details: {error_raw}")

        # 5. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            print(f"🚨 [MODEL ERROR] {error_type}: Attempting to explain an untrained model. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[MODEL ERROR] {error_type}: The loaded machine learning model is not trained. Details: {error_raw}")

        # 6. Fallback for any other unknown errors
        else:
            print(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during execution. Details: {error_raw}")
            print("="*70 + "\n")
            raise HTTPException(status_code=500, detail=f"[UNKNOWN ERROR] {error_type}: An unexpected server error occurred during LIME generation. Details: {error_raw}")