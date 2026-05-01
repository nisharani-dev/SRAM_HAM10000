from fastapi import FastAPI, UploadFile, File
import shutil
import os
from run_openram import run_openram

app = FastAPI()

@app.get("/")
def home():
    return {"message": "SRAM Studio running"}

@app.post("/run")
async def run_sram(file: UploadFile = File(...)):
    os.makedirs("jobs", exist_ok=True)
    
    file_path = f"jobs/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stdout, stderr = run_openram(file_path)

    return {
        "stdout": stdout,
        "stderr": stderr
    }