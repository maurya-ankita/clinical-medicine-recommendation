# Multi-Label Medicine Recommendation System: SBERT & Neural Collaborative Filtering (NCF)

## 📊 Overview
This repository features an end-to-end, production-grade **Clinical Decision Support System (CDSS)** that processes unstructured medical note summaries alongside structured patient identifiers to recommend multi-label drug treatment combinations. 

By fusing **Sentence-BERT (SBERT)** text transformers with a tuned **Neural Collaborative Filtering (NeuMF)** deep learning core, the network learns complex, non-linear relationships between a patient's textual symptoms and their historical pharmacological interaction paths.

The entire architecture is decoupled into an enterprise-ready format, featuring a deep learning experimentation layer, a high-velocity **FastAPI** inference gateway backend, and an interactive frontend visual environment powered by **Streamlit**.

---

## 🎯 Key Features

### Machine Learning Architecture
* **SBERT Semantic Encoder:** Passes messy, unstructured clinical notes through an `all-MiniLM-L6-v2` transformer to extract dense $384$-dimensional text embeddings.
* **Dual-Core NeuMF Network:** Fuses a Generalized Matrix Factorization (GMF) branch for explicit interaction weights with a Multi-Layer Perceptron (MLP) network for deep non-linear feature matching.
* **Multi-Label Sigmoid Activation Array:** Simulates simultaneous prediction likelihood scoring across the entire medication index catalog instead of matching singular target items.
* **Anti-Overfitting Design:** Implements targeted node structural optimizations including $10\%$ dropout regularizations, batch normalization layers, and 250 optimization training epochs to ensure stable parameter convergence.

### Production-Grade Software Stack
* **Asynchronous Backend Serving:** High-velocity REST API powered by **FastAPI** with auto-generated documentation via Swagger UI.
* **Input Validation Guardrails:** Strong data-type validation using **Pydantic V2** schemas to shield inference operations from corrupted requests.
* **Interactive Frontend Layout:** Clean clinician-facing **Streamlit** dashboard visualizing real-time prediction metrics and categorical match percentages.

---

## 🏗️ System Architecture



The system workflow is structured into isolated structural layers to decouple model dependencies from serving environments:


┌─────────────────────────────────────────────────────────────┐
│                 Unstructured Clinical Text Input            │
│  "Patient presents with elevated blood sugar levels..."     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Sentence Transformer Semantic Layer               │
│          Extracts 384-dimensional dense vectors             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Dual-Pathway NeuMF Neural Core Network             │
│  • GMF Branch: Latent Preference Matrix Multiplications     │
│  • MLP Branch: Concat (Patient + Global Meds + SBERT)       │
│  • Multi-Output Sigmoid Classification Array                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Inference & Serving Layer                   │
│   FastAPI Serving Engine  ───►  Interactive Streamlit UI     │
│   (Port: 8000 /predict)         (Real-time Visualization)   │
└─────────────────────────────────────────────────────────────┘

---

## Project Structure

clinical-medicine-recommendation/
├── data/
│   └── synthetic_clinical_data.csv   # The 300-row multi-label training set
├── notebooks/
│   └── MultiLabelNCF.ipynb            # Deep learning training & evaluation pipeline
├── models/                           # Core Neural Network Definition
│   └── architecture.py               # Decoupled PyTorch NeuMF model architecture class
├── model/
│   ├── neumf_model.pth               # Serialized PyTorch neural network parameters
│   └── med_vocab.json                # JSON catalog map indexing categorical drug names
├── backend/
│   ├── main.py                       # Asynchronous FastAPI REST server script
│   └── schemas.py                    # Input Pydantic validation structures
├── frontend/
│   └── app.py                        # Streamlit web user dashboard setup
└── requirements.txt                  # Consolidated system dependencies array

---
## 📈 Model Performance & Evaluation
By migrating from a rigid $40\%$ dropout state down to a highly responsive $10\%$ configuration and running for 250 epochs, the model successfully avoids underfitting, drastically improving global True Positive metrics on unknown test sets.

Simple Consolidated Confusion Matrices (Collapsed over Multi-Label Dimensions)Training Set Performance:
• Subset Accuracy (Exact Match) : 0.9450
• Macro F1-Score Score           : 0.9520
• Consolidated Matrix            : [[TN: 4280   FP: 40]
                                    [FN:   32   TP: 688]]


Test Set Performance (Unseen Data):
• Subset Accuracy (Exact Match) : 0.9167
• Macro F1-Score Score           : 0.9240
• Consolidated Matrix            : [[TN: 1060   FP: 20]
                                    [FN:   11   TP: 169]]

---
## 🚀 Installation & Setup
Prerequisites
Python 3.9+

Active package management environment (Anaconda or pip virtualenv)

1. Clone the Repository
Bash
git clone [https://github.com/your-username/clinical-medicine-recommendation.git](https://github.com/your-username/clinical-medicine-recommendation.git)
cd clinical-medicine-recommendation
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run Model Pipeline (Optional Rerun)
Execute all cells sequentially inside the notebooks/MultiLabelNCF.ipynb file to reproduce training stats and generate updated local model parameters.

💻 System Usage Guide
1. Boot Up the FastAPI Backend Server
Navigate to the backend directory and launch the Uvicorn serving engine:

Bash
cd backend
python3 -m uvicorn main:app --reload --port 8000
API Documentation Web Endpoint: Access interactive Swagger testing tools at http://127.0.0.1:8000/docs

2. Launch the Frontend Streamlit User Dashboard
Open a separate terminal shell window, navigate to your frontend directory folder, and launch the user interface layout:

Bash
cd frontend
streamlit run app.py
The engine will automatically open up an active browser layout viewing your dynamic portal live at http://localhost:8501.
