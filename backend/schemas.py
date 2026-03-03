"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel


class PatientData(BaseModel):
    """Input schema for patient churn prediction request."""
    Age: int
    Gender: str
    State: str
    Tenure_Months: int
    Specialty: str
    Insurance_Type: str
    Visits_Last_Year: int
    Missed_Appointments: int
    Days_Since_Last_Visit: int
    Last_Interaction_Date: str
    Overall_Satisfaction: float
    Wait_Time_Satisfaction: float
    Staff_Satisfaction: float
    Provider_Rating: float
    Avg_Out_Of_Pocket_Cost: float
    Billing_Issues: int
    Portal_Usage: int
    Referrals_Made: int
    Distance_To_Facility_Miles: float


class PredictionResponse(BaseModel):
    """Output schema for churn prediction result."""
    prediction: int
    churn_probability: float
