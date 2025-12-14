from fastapi import FastAPI

app = FastAPI(title="Credit Risk Model API")

@app.get("/health")
def health():
    return {"status": "ok"}
