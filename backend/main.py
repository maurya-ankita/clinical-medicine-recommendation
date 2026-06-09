import os
import sys
import json
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

# Appends the root directory to the system path so Python can find your custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.architecture import RegularizedNeuMF
from schemas import RecommendationRequest

app = FastAPI(
    title="🏥 Spatiotemporal Clinical Recommendation Engine",
    description="Production-grade API serving multi-label medicine predictions using SBERT and NCF.",
    version="1.0.0"
)

# Global variables to hold assets in server memory for fast execution
sbert_encoder = None
model = None
med_vocab = None

@app.on_event("startup")
def load_assets():
    """Executes immediately when the server starts up, loading files into RAM."""
    global sbert_encoder, model, med_vocab
    try:
        print("Initializing Sentence-BERT Transformer...")
        sbert_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading Medicine Vocabulary Map...")
        with open('../model/med_vocab.json', 'r') as f:
            med_vocab = json.load(f)
            
        num_medicines = len(med_vocab)
        sbert_dim = 384  # Dimension size for all-MiniLM-L6-v2
        
        # Instantiating the architecture with dimensions matching the synthetic dataset setup
        print("Instantiating NeuMF Network Architecture...")
        model = RegularizedNeuMF(num_patients=60, num_medicines=num_medicines, sbert_dim=sbert_dim)
        
        print("Loading trained weights from disk...")
        model.load_state_dict(torch.load('../model/neumf_model.pth', map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode (turns off dropout and batchnorm)
        print("✅ All production assets loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error during server asset startup: {str(e)}")

@app.get("/")
def health_check():
    """Simple endpoint to verify server status and model accessibility."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_treatment(payload: RecommendationRequest):
    """
    Accepts clinical note text, encodes it via SBERT, runs a forward pass
    through the regularized NeuMF model, and returns the top-K recommended medicines.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is training or uninitialized.")
        
    try:
        # 1. Transform raw string note into a 384-dimensional dense semantic vector
        text_embedding = sbert_encoder.encode([payload.condition_description])
        text_tensor = torch.tensor(text_embedding, dtype=torch.float32)
        
        # 2. Map incoming patient ID cleanly into our embedding row matrix limits (0-59)
        safe_patient_id = payload.patient_id % 60
        patient_tensor = torch.tensor([safe_patient_id], dtype=torch.long)
        
        # 3. Suppress gradients and run forward pass for fast inference
        with torch.no_grad():
            probabilities = model(patient_tensor, text_tensor).squeeze(0).numpy()
            
        # 4. Filter, rank, and map back to real drug text names
        ranked_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:payload.top_k]
        
        recommendations = [med_vocab[str(idx)] for idx in ranked_indices]
        confidence_scores = [float(probabilities[idx]) for idx in ranked_indices]
        
        return {
            "status": "success",
            "patient_id": payload.patient_id,
            "condition_processed": payload.condition_description,
            "recommended_treatment": recommendations,
            "confidence_metrics": confidence_scores
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")