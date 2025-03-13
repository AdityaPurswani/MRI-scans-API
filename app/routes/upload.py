from fastapi import APIRouter, File, UploadFile, HTTPException
import os
import shutil
from app.services.pipeline import MRIProcessingPipeline

router = APIRouter(prefix="/api/upload", tags=["Upload & Process"])

UPLOAD_DIR = "uploads/"
PROCESSED_DIR = "processed/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@router.post("/")
async def upload_and_process(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the MRI file
    pipeline = MRIProcessingPipeline(file_path)
    result = pipeline.run_pipeline()

    return result
