from pydantic import BaseModel

class TransactionData(BaseModel):
    transaction_id: str
    amount: float
