from fastapi import APIRouter, Query, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
import os
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Potentially import your pipeline orchestrator ---
# This assumes MRIProcessingPipelineWithSpaces is in app/services/pipeline.py
# and your project root (MRISCANSAPI) is in the Python path.
try:
    from app.services.pipeline import MRIProcessingPipelineWithSpaces
except ImportError as e:
    logging.error(f"Failed to import MRIProcessingPipelineWithSpaces: {e}. Ensure app.services.pipeline.py exists and is accessible.")
    # Define a dummy class if import fails, so the rest of the file can be parsed
    # This should be fixed by ensuring the correct import path and file existence.
    class MRIProcessingPipelineWithSpaces:
        def __init__(self, input_object_key: str):
            logging.warning(f"Using DUMMY MRIProcessingPipelineWithSpaces for key: {input_object_key}")
            self.input_object_key = input_object_key
        def run_pipeline(self) -> Dict:
            logging.warning(f"DUMMY run_pipeline called for: {self.input_object_key}")
            return {"status": "dummy pipeline run", "input_key": self.input_object_key, "outputs": {}}

# --- Load environment variables ---
# Consistent with other modules. Assumes this file is in app/router/
# and .env is in the project root (MRISCANSAPI/).
try:
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"pipeline_trigger_routes.py: .env loaded from: {dotenv_path}")
    else:
        logging.info(f"pipeline_trigger_routes.py: .env file not found at {dotenv_path}, assuming variables are set externally or by main app.")
except Exception as e:
    logging.error(f"pipeline_trigger_routes.py: Error loading .env file: {e}")

router = APIRouter(prefix="/api/pipeline", tags=["MRI Processing Pipeline"])

# Configure logging
if not logging.getLogger().hasHandlers(): # Avoid reconfiguring if already set by main app
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Request Body Model ---
class PipelineRunRequest(BaseModel):
    input_object_key: str = Field(..., 
                                  description="The S3 object key for the input NIFTI file in DigitalOcean Spaces.",
                                  example="uploads/patientA/scan001.nii.gz")

@router.post("/run", response_model=Dict[str, Any])
async def run_mri_pipeline(
    request_body: PipelineRunRequest = Body(...)
):
    """
    Triggers the MRI processing pipeline for a NIFTI file stored in DigitalOcean Spaces.
    
    The pipeline will:
    1. Download the input file from the specified S3 object key.
    2. Perform brain extraction, normalization, bias correction, and registration.
    3. Segment the brain based on atlases.
    4. Extract volumetric features.
    5. Upload all intermediate and final results (compressed NIFTI, uncompressed NIFTI, CSVs) 
       to a structured output location in DigitalOcean Spaces, typically under 
       `mri_pipeline_outputs/YOUR_INPUT_FILENAME_WITHOUT_EXT/`.
       
    The endpoint returns a dictionary containing S3 URIs for the generated output files.
    """
    input_key = request_body.input_object_key
    logger.info(f"Received request to run MRI pipeline for input object key: {input_key}")

    if not input_key:
        raise HTTPException(status_code=400, detail="input_object_key must be provided.")

    try:
        # Initialize the pipeline orchestrator with the S3 object key
        pipeline_orchestrator = MRIProcessingPipelineWithSpaces(input_object_key=input_key)
        
        # Run the pipeline
        # This is a potentially long-running operation. For production, consider background tasks (e.g., Celery).
        logger.info(f"Starting pipeline execution for: {input_key}")
        pipeline_results = pipeline_orchestrator.run_pipeline() # This is a synchronous call
        
        logger.info(f"Pipeline completed for: {input_key}. Results: {pipeline_results}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "MRI processing pipeline initiated and completed successfully.",
                "input_object_key": input_key,
                "outputs": pipeline_results 
                # pipeline_results should be the dictionary of S3 URIs from MRIProcessingPipelineWithSpaces
            }
        )

    except ValueError as ve: # Catch config errors from MRIProcessingPipelineWithSpaces.__init__
        logger.error(f"Configuration error for pipeline with input {input_key}: {str(ve)}")
        raise HTTPException(status_code=500, detail=f"Pipeline configuration error: {str(ve)}")
    except RuntimeError as rte: # Catch errors during pipeline execution
        logger.error(f"Runtime error during pipeline execution for input {input_key}: {str(rte)}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(rte)}")
    except FileNotFoundError as fnf: # If a critical local file (like atlas) is missing
        logger.error(f"File not found during pipeline execution for {input_key}: {str(fnf)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed due to a missing critical file: {str(fnf)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while running the pipeline for input {input_key}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To include this router in your main FastAPI app (e.g., in app/main.py):
# from app.router import pipeline_trigger_routes 
# app.include_router(pipeline_trigger_routes.router)
