from pydantic import BaseModel, Field

class RecommendationRequest(BaseModel):
    patient_id: int = Field(..., description="The unique numerical index of the patient", example=0)
    condition_description: str = Field(..., description="The raw medical text or doctor notes", example="Patient has high blood sugar and frequent urination.")
    top_k: int = Field(default=3, description="Number of top medicine recommendations to return", ge=1, le=5)

