from pydantic import BaseModel

# Input data schema (make sure these match your model features exactly)
class CustomerData(BaseModel):
    SumInsured: float
    NumberOfVehiclesInFleet: float
    policy_claim_count: float
    CalculatedPremiumPerTerm: float
    vehicle_age: float

# Output response schema
class PredictionResponse(BaseModel):
    risk_probability: float
