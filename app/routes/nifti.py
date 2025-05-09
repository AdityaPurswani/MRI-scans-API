from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import Response, JSONResponse
import os
import base64
import boto3
from botocore.exceptions import NoCredentialsError, ClientError, ParamValidationError
import logging
from typing import Optional, List, Dict, Any # Added Dict, Any
from datetime import datetime # Added datetime
from dotenv import load_dotenv

# --- Load environment variables ---
# Consistent with other modules. Assumes this file is in app/router/
# and .env is in the project root (MRISCANSAPI/).
try:
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"nifti.py: .env loaded from: {dotenv_path}")
    else:
        logging.info(f"nifti.py: .env file not found at {dotenv_path}, assuming variables are set externally or by main app.")
except Exception as e:
    logging.error(f"nifti.py: Error loading .env file: {e}")

# --- Configuration (from environment variables) ---
DO_SPACES_REGION_NAME = os.getenv("DO_SPACES_REGION_NAME", "nyc3")
DO_SPACES_ENDPOINT_URL = os.getenv("DO_SPACES_ENDPOINT_URL")
DO_SPACES_ACCESS_KEY_ID = os.getenv("DO_SPACES_ACCESS_KEY_ID")
DO_SPACES_SECRET_ACCESS_KEY = os.getenv("DO_SPACES_SECRET_ACCESS_KEY")
DO_SPACES_BUCKET_NAME = os.getenv("DO_SPACES_BUCKET_NAME")

if not DO_SPACES_ENDPOINT_URL and DO_SPACES_REGION_NAME:
    DO_SPACES_ENDPOINT_URL = f"https://{DO_SPACES_REGION_NAME}.digitaloceanspaces.com"

# The router prefix remains /api, but the tag can be more specific
router = APIRouter(prefix="/api", tags=["NIFTI (from DO Spaces)"])

# Configure logging
if not logging.getLogger().hasHandlers(): # Avoid reconfiguring if already set by main app
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Boto3 S3 Client Dependency (reusable from other modules) ---
def get_s3_client():
    """
    FastAPI dependency to get a boto3 S3 client configured for DigitalOcean Spaces.
    """
    if not all([DO_SPACES_ACCESS_KEY_ID, DO_SPACES_SECRET_ACCESS_KEY, DO_SPACES_BUCKET_NAME, DO_SPACES_ENDPOINT_URL, DO_SPACES_REGION_NAME]):
        missing_vars = [var_name for var_name, var_value in [
            ("DO_SPACES_ACCESS_KEY_ID", DO_SPACES_ACCESS_KEY_ID),
            ("DO_SPACES_SECRET_ACCESS_KEY", DO_SPACES_SECRET_ACCESS_KEY),
            ("DO_SPACES_BUCKET_NAME", DO_SPACES_BUCKET_NAME),
            ("DO_SPACES_ENDPOINT_URL", DO_SPACES_ENDPOINT_URL),
            ("DO_SPACES_REGION_NAME", DO_SPACES_REGION_NAME)
        ] if not var_value]
        logger.error(f"S3 client configuration incomplete in nifti.py. Missing: {missing_vars}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: Storage service not fully configured. Missing: {', '.join(missing_vars)}")
    
    try:
        client = boto3.client(
            's3',
            region_name=DO_SPACES_REGION_NAME,
            endpoint_url=DO_SPACES_ENDPOINT_URL,
            aws_access_key_id=DO_SPACES_ACCESS_KEY_ID,
            aws_secret_access_key=DO_SPACES_SECRET_ACCESS_KEY
        )
        logger.info("S3 client for nifti.py initialized successfully.")
        return client
    except Exception as e: # Catch any other client creation errors
        logger.error(f"nifti.py: Unexpected error during S3 client initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: An unexpected error occurred with storage service: {e}")

# --- Helper Function to Get Object Metadata (reusable) ---
async def get_object_metadata(s3_client: boto3.client, bucket_name: str, object_key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves metadata (like size, content type) for an object in S3/Spaces.
    Returns metadata dict or None if object not found or error.
    """
    try:
        if not bucket_name:
            logger.error("get_object_metadata: Bucket name is not configured.")
            raise HTTPException(status_code=500, detail="Server configuration error: Storage bucket name missing.")
        metadata = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return metadata
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404' or error_code == 'NoSuchKey':
            logger.warning(f"Object not found in Spaces: Bucket='{bucket_name}', Key='{object_key}'")
            return None
        elif error_code == '403':
            logger.error(f"Access denied (403) for metadata: Bucket='{bucket_name}', Key='{object_key}'. Error: {e}")
            raise HTTPException(status_code=403, detail=f"Access denied to file metadata: {e.response.get('Error', {}).get('Message', 'Permission error')}")
        else:
            logger.error(f"S3 ClientError accessing metadata: Bucket='{bucket_name}', Key='{object_key}'. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error accessing file metadata: {e.response.get('Error', {}).get('Message', 'Unknown S3 error')}")
    except ParamValidationError as e: # Catch errors from invalid parameters passed to head_object
        logger.error(f"Invalid parameters for head_object: Bucket='{bucket_name}', Key='{object_key}'. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid file path/key format: {e}")
    except Exception as e: 
        logger.error(f"Unexpected error getting metadata: Bucket='{bucket_name}', Key='{object_key}'. Error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching file metadata.")


@router.get("/nifti") # Endpoint path remains the same
async def get_nifti_file_from_digital_ocean(
    object_key: str = Query(..., description="The S3 object key (path) of the NIFTI file in DigitalOcean Spaces. E.g., 'mri_pipeline_outputs/scan_name/processed/uncompressed/brain.nii'"),
    s3_client: boto3.client = Depends(get_s3_client)
):
    """
    Serve a NIFTI file directly from DigitalOcean Spaces.
    The 'object_key' is the full path to the file within your Space bucket.
    """
    logger.info(f"Requested NIFTI file from Spaces. Bucket: '{DO_SPACES_BUCKET_NAME}', Key: '{object_key}'")

    if not DO_SPACES_BUCKET_NAME:
        logger.error("/nifti endpoint: DO_SPACES_BUCKET_NAME not configured.")
        raise HTTPException(status_code=500, detail="Server configuration error: Bucket name not set.")

    # Get object metadata (includes size and S3 content type)
    metadata = await get_object_metadata(s3_client, DO_SPACES_BUCKET_NAME, object_key)
    if metadata is None: # File not found
        raise HTTPException(status_code=404, detail="File not found in storage.")

    file_size = metadata.get('ContentLength', 0)
    s3_content_type = metadata.get('ContentType', 'application/octet-stream') # Get content type from S3
    logger.info(f"File found in Spaces. Key: '{object_key}', Size: {file_size} bytes, S3 ContentType: {s3_content_type}")

    # File size limit (retained from original logic, adjust as needed)
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB limit, adjust as needed
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"File too large: {file_size} bytes. Key: '{object_key}'")
        raise HTTPException(status_code=413, detail=f"File too large to process (limit: {MAX_FILE_SIZE_BYTES // (1024*1024)}MB)")
            
    try:
        # Get the object from S3
        s3_response_object = s3_client.get_object(Bucket=DO_SPACES_BUCKET_NAME, Key=object_key)
        
        file_content = s3_response_object['Body'].read()
        actual_read_bytes = len(file_content)
        logger.info(f"Successfully read {actual_read_bytes} bytes from Spaces for key: '{object_key}'")
        
        # Determine media type: prioritize S3 metadata, then by extension, then fallback
        media_type = s3_content_type
        if object_key.endswith(".nii") and media_type == 'application/octet-stream': # If S3 type is generic
            media_type = "application/x-nifti" 
        elif object_key.endswith(".nii.gz") and media_type == 'application/octet-stream':
            media_type = "application/gzip"
        
        return Response(
            content=file_content,
            media_type=media_type,
            headers={
                "X-Content-Type-Options": "nosniff",
                "Content-Length": str(actual_read_bytes),
                "Access-Control-Allow-Origin": "*", # Consider restricting this in production
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Disposition": f"inline; filename=\"{os.path.basename(object_key)}\"" # Suggests filename
            }
        )
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == 'NoSuchKey': # Should be caught by head_object, but as safeguard
            logger.warning(f"File not found during get_object: Bucket='{DO_SPACES_BUCKET_NAME}', Key='{object_key}'")
            raise HTTPException(status_code=404, detail="File not found.")
        elif error_code == '403':
            logger.error(f"Access Denied (403) fetching file: Key='{object_key}'. Error: {e}")
            raise HTTPException(status_code=403, detail=f"Access denied to file: {e.response.get('Error', {}).get('Message', 'Permission error')}")
        else:
            logger.error(f"S3 ClientError fetching file: Key='{object_key}'. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file from storage: {e.response.get('Error', {}).get('Message', 'Unknown S3 error')}")
    except ParamValidationError as e: # If get_object parameters are somehow invalid
        logger.error(f"Invalid parameters for get_object: Key='{object_key}'. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid file path/key format for retrieval: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading file from Spaces: Key='{object_key}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read file from storage: {e}")


@router.get("/nifti-base64") # Endpoint path remains the same
async def get_nifti_file_base64_from_digital_ocean(
    object_key: str = Query(..., description="The S3 object key (path) of the NIFTI file in DigitalOcean Spaces."),
    s3_client: boto3.client = Depends(get_s3_client)
):
    """
    Serve a NIFTI file from DigitalOcean Spaces as base64 encoded data.
    """
    logger.info(f"Requested NIFTI file (base64) from Spaces. Bucket: '{DO_SPACES_BUCKET_NAME}', Key: '{object_key}'")
    
    if not DO_SPACES_BUCKET_NAME:
        logger.error("/nifti-base64 endpoint: DO_SPACES_BUCKET_NAME not configured.")
        raise HTTPException(status_code=500, detail="Server configuration error: Bucket name not set.")

    metadata = await get_object_metadata(s3_client, DO_SPACES_BUCKET_NAME, object_key)
    if metadata is None:
        raise HTTPException(status_code=404, detail="File not found in storage.")

    file_size = metadata.get('ContentLength', 0)
    s3_content_type = metadata.get('ContentType', 'application/octet-stream')
    logger.info(f"File found for base64. Key: '{object_key}', Size: {file_size} bytes, S3 ContentType: {s3_content_type}")
        
    # File size limit for base64 encoding (original file size)
    MAX_BASE64_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB limit
    if file_size > MAX_BASE64_FILE_SIZE_BYTES:
        logger.warning(f"File too large for base64 encoding: {file_size} bytes. Key: '{object_key}'")
        raise HTTPException(status_code=413, detail=f"File too large to process as base64 (limit: {MAX_BASE64_FILE_SIZE_BYTES // (1024*1024)}MB)")
            
    try:
        s3_response_object = s3_client.get_object(Bucket=DO_SPACES_BUCKET_NAME, Key=object_key)
        file_content = s3_response_object['Body'].read()
            
        base64_content = base64.b64encode(file_content).decode('utf-8')
        
        logger.info(f"Successfully read and base64 encoded file from Spaces: '{object_key}'")
        
        return JSONResponse(content={ # Ensure 'content=' for JSONResponse
            "filename": os.path.basename(object_key),
            "original_file_size_bytes": file_size,
            "s3_content_type": s3_content_type, # Content type of the original file from S3
            "base64_data": base64_content
        })
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == 'NoSuchKey':
            logger.warning(f"File not found during get_object (base64): Bucket='{DO_SPACES_BUCKET_NAME}', Key='{object_key}'")
            raise HTTPException(status_code=404, detail="File not found.")
        elif error_code == '403':
            logger.error(f"Access Denied (403) fetching for base64: Key='{object_key}'. Error: {e}")
            raise HTTPException(status_code=403, detail=f"Access denied to file for base64: {e.response.get('Error', {}).get('Message', 'Permission error')}")
        else:
            logger.error(f"S3 ClientError fetching for base64: Key='{object_key}'. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file for base64: {e.response.get('Error', {}).get('Message', 'Unknown S3 error')}")
    except ParamValidationError as e:
        logger.error(f"Invalid parameters for get_object (base64): Key='{object_key}'. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid file path/key format for base64 retrieval: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading for base64 from Spaces: Key='{object_key}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read file for base64: {e}")

# To include this router in your main FastAPI app (e.g., in app/main.py):