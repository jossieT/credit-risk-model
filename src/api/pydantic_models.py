from pydantic import BaseModel
from typing import Optional

class CustomerFeatures(BaseModel):
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    avg_transaction_hour: float
    std_transaction_hour: float
    avg_transaction_day: float
    avg_transaction_month: float
    ProductCategory: str
    ChannelId: str
    ProviderId: str

class PredictionResponse(BaseModel):
    is_high_risk: int
    probability_of_default: float
