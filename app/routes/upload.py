from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import os
import shutil
import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import httpx # HTTP client for making internal API calls
from dotenv import load_dotenv
from typing import Dict, Any

# --- Load environment variables ---
# Consistent with other modules. Assumes this file is in app/router/
# and .env is in the project root (MRISCANSAPI/).
try:
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"upload.py: .env loaded from: {dotenv_path}")
    else:
        logging.info(f"upload.py: .env file not found at {dotenv_path}, assuming variables are set externally or by main app.")
except Exception as e:
    logging.error(f"upload.py: Error loading .env file: {e}")

# --- Configuration (from environment variables) ---
DO_SPACES_REGION_NAME = os.getenv("DO_SPACES_REGION_NAME", "nyc3")
DO_SPACES_ENDPOINT_URL = os.getenv("DO_SPACES_ENDPOINT_URL")
DO_SPACES_ACCESS_KEY_ID = os.getenv("DO_SPACES_ACCESS_KEY_ID")
DO_SPACES_SECRET_ACCESS_KEY = os.getenv("DO_SPACES_SECRET_ACCESS_KEY")
DO_SPACES_BUCKET_NAME = os.getenv("DO_SPACES_BUCKET_NAME")
DO_SPACES_UPLOAD_PREFIX = os.getenv("DO_SPACES_UPLOAD_PREFIX", "uploads/") # Default prefix for uploads in Spaces

if not DO_SPACES_ENDPOINT_URL and DO_SPACES_REGION_NAME:
    DO_SPACES_ENDPOINT_URL = f"https://{DO_SPACES_REGION_NAME}.digitaloceanspaces.com"

router = APIRouter(prefix="/api/upload", tags=["Upload & Process (DO Spaces)"])

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Boto3 S3 Client Dependency (similar to other route files) ---
def get_s3_client():
    if not all([DO_SPACES_ACCESS_KEY_ID, DO_SPACES_SECRET_ACCESS_KEY, DO_SPACES_BUCKET_NAME, DO_SPACES_ENDPOINT_URL, DO_SPACES_REGION_NAME]):
        missing_vars = [var_name for var_name, var_value in [
            ("DO_SPACES_ACCESS_KEY_ID", DO_SPACES_ACCESS_KEY_ID),
            ("DO_SPACES_SECRET_ACCESS_KEY", DO_SPACES_SECRET_ACCESS_KEY),
            ("DO_SPACES_BUCKET_NAME", DO_SPACES_BUCKET_NAME),
            ("DO_SPACES_ENDPOINT_URL", DO_SPACES_ENDPOINT_URL),
            ("DO_SPACES_REGION_NAME", DO_SPACES_REGION_NAME)
        ] if not var_value]
        logger.error(f"S3 client configuration incomplete in upload.py. Missing: {missing_vars}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: Storage service not fully configured. Missing: {', '.join(missing_vars)}")
    
    try:
        client = boto3.client(
            's3',
            region_name=DO_SPACES_REGION_NAME,
            endpoint_url=DO_SPACES_ENDPOINT_URL,
            aws_access_key_id=DO_SPACES_ACCESS_KEY_ID,
            aws_secret_access_key=DO_SPACES_SECRET_ACCESS_KEY
        )
        logger.info("S3 client for upload.py initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"upload.py: Unexpected error during S3 client initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: An unexpected error occurred with storage service: {e}")

# Temporary local directory for staging uploads before sending to Spaces
TEMP_UPLOAD_DIR = "temp_uploads/"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_to_spaces_and_trigger_pipeline(
    request: Request, # To get base URL for internal API call
    file: UploadFile = File(...),
    s3_client: boto3.client = Depends(get_s3_client)
):
    """
    Uploads a NIFTI file to DigitalOcean Spaces and then triggers the MRI processing pipeline.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    
    # Sanitize filename (optional, but good practice)
    # For simplicity, using the original filename. Consider more robust sanitization.
    safe_filename = os.path.basename(file.filename) 
    
    # Define a temporary local path to save the uploaded file before sending to Spaces
    temp_local_file_path = os.path.join(TEMP_UPLOAD_DIR, safe_filename)

    try:
        # Save uploaded file temporarily to local disk
        with open(temp_local_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{safe_filename}' temporarily saved to '{temp_local_file_path}'.")

        # Define the object key for DigitalOcean Spaces
        # Ensure the prefix ends with a slash
        upload_prefix = DO_SPACES_UPLOAD_PREFIX
        if not upload_prefix.endswith('/'):
            upload_prefix += '/'
        
        object_key = f"{upload_prefix}{safe_filename}"
        
        # Upload the temporary local file to DigitalOcean Spaces
        logger.info(f"Uploading '{temp_local_file_path}' to Spaces bucket '{DO_SPACES_BUCKET_NAME}' with key '{object_key}'.")
        s3_client.upload_file(temp_local_file_path, DO_SPACES_BUCKET_NAME, object_key)
        logger.info(f"File '{safe_filename}' successfully uploaded to Spaces with key '{object_key}'.")

    except FileNotFoundError:
        logger.error(f"Temporary file not found for upload: {temp_local_file_path}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file locally before S3 upload.")
    except NoCredentialsError:
        logger.error("S3 credentials not available for upload.")
        raise HTTPException(status_code=500, detail="Server configuration error: Storage credentials missing for S3 upload.")
    except ClientError as e:
        logger.error(f"S3 ClientError during upload of '{safe_filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file to storage: {e.response.get('Error', {}).get('Message', 'Unknown S3 error')}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during file upload to Spaces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file upload: {str(e)}")
    finally:
        # Clean up the temporary local file
        if os.path.exists(temp_local_file_path):
            os.remove(temp_local_file_path)
            logger.info(f"Temporary local file '{temp_local_file_path}' cleaned up.")

    # Now, trigger the pipeline by calling the /api/pipeline/run endpoint internally
    pipeline_trigger_url = f"{request.base_url}api/pipeline/run" # Construct full URL
    payload = {"input_object_key": object_key}
    
    logger.info(f"Triggering pipeline by calling internal endpoint: {pipeline_trigger_url} with payload: {payload}")

    async with httpx.AsyncClient(timeout=None) as client: # Timeout=None for potentially long pipeline runs
        try:
            response = await client.post(pipeline_trigger_url, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            pipeline_response_data = response.json()
            logger.info(f"Pipeline trigger successful. Response from pipeline: {pipeline_response_data}")
            
            # Return the response from the pipeline trigger endpoint
            return JSONResponse(
                status_code=response.status_code, # Use status code from internal call
                content=pipeline_response_data
            )
        except httpx.HTTPStatusError as exc:
            logger.error(f"Error calling pipeline trigger endpoint '{exc.request.url}': Status {exc.response.status_code}, Response: {exc.response.text}")
            error_detail = f"Failed to trigger processing pipeline. Status: {exc.response.status_code}."
            try: # Try to parse error from pipeline response
                error_detail += f" Detail: {exc.response.json().get('detail', exc.response.text)}"
            except: # If response is not JSON or no detail field
                 error_detail += f" Raw Response: {exc.response.text}"
            raise HTTPException(status_code=exc.response.status_code, detail=error_detail)
        except httpx.RequestError as exc:
            logger.error(f"RequestError calling pipeline trigger endpoint '{exc.request.url}': {exc}")
            raise HTTPException(status_code=503, detail=f"Failed to connect to processing pipeline service: {exc}")
        except Exception as e:
            logger.error(f"Unexpected error when calling pipeline trigger endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while triggering the pipeline: {str(e)}")

# To include this router in your main FastAPI app (e.g., in app/main.py):
# from app.router import upload as upload_routes # Assuming this file is upload.py in app/router/
# app.include_router(upload_routes.router)
