from fastapi import APIRouter, HTTPException
import os
from app.services.pipeline import MRIProcessingPipeline


router = APIRouter(prefix="/api/process", tags=["Processing"])

@router.get("/{filename}")
async def process_mri(filename: str):
    file_path = os.path.join("uploads", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    pipeline = MRIProcessingPipeline(file_path)
    result = pipeline.run_pipeline()
    
    return result
