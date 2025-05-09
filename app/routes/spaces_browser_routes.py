from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError, ParamValidationError
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# --- Load environment variables ---
# Consistent with other modules. Assumes this file is in app/router/
# and .env is in the project root (MRISCANSAPI/).
try:
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"spaces_browser_routes.py: .env loaded from: {dotenv_path}")
    else:
        logging.info(f"spaces_browser_routes.py: .env file not found at {dotenv_path}, assuming variables are set externally or by main app.")
except Exception as e:
    logging.error(f"spaces_browser_routes.py: Error loading .env file: {e}")

# --- Configuration (from environment variables) ---
DO_SPACES_REGION_NAME = os.getenv("DO_SPACES_REGION_NAME", "nyc3")
DO_SPACES_ENDPOINT_URL = os.getenv("DO_SPACES_ENDPOINT_URL")
DO_SPACES_ACCESS_KEY_ID = os.getenv("DO_SPACES_ACCESS_KEY_ID")
DO_SPACES_SECRET_ACCESS_KEY = os.getenv("DO_SPACES_SECRET_ACCESS_KEY")
DO_SPACES_BUCKET_NAME = os.getenv("DO_SPACES_BUCKET_NAME")

if not DO_SPACES_ENDPOINT_URL and DO_SPACES_REGION_NAME:
    DO_SPACES_ENDPOINT_URL = f"https://{DO_SPACES_REGION_NAME}.digitaloceanspaces.com"

router = APIRouter(prefix="/api/spaces", tags=["DigitalOcean Spaces Browser"])

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Boto3 S3 Client Dependency (same as in nifti_routes.py) ---
def get_s3_client():
    if not all([DO_SPACES_ACCESS_KEY_ID, DO_SPACES_SECRET_ACCESS_KEY, DO_SPACES_BUCKET_NAME, DO_SPACES_ENDPOINT_URL, DO_SPACES_REGION_NAME]):
        missing_vars = [var_name for var_name, var_value in [
            ("DO_SPACES_ACCESS_KEY_ID", DO_SPACES_ACCESS_KEY_ID),
            ("DO_SPACES_SECRET_ACCESS_KEY", DO_SPACES_SECRET_ACCESS_KEY),
            ("DO_SPACES_BUCKET_NAME", DO_SPACES_BUCKET_NAME),
            ("DO_SPACES_ENDPOINT_URL", DO_SPACES_ENDPOINT_URL),
            ("DO_SPACES_REGION_NAME", DO_SPACES_REGION_NAME)
        ] if not var_value]
        logger.error(f"S3 client configuration incomplete in spaces_browser. Missing: {missing_vars}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: Storage service not fully configured. Missing: {', '.join(missing_vars)}")
    
    try:
        client = boto3.client(
            's3',
            region_name=DO_SPACES_REGION_NAME,
            endpoint_url=DO_SPACES_ENDPOINT_URL,
            aws_access_key_id=DO_SPACES_ACCESS_KEY_ID,
            aws_secret_access_key=DO_SPACES_SECRET_ACCESS_KEY
        )
        logger.info("S3 client for spaces_browser_routes initialized successfully.")
        return client
    except NoCredentialsError:
        logger.error("spaces_browser_routes: S3 credentials not available.")
        raise HTTPException(status_code=500, detail="Server configuration error: Storage credentials missing.")
    except ClientError as e:
        logger.error(f"spaces_browser_routes: S3 ClientError during initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: Could not connect to storage service. Error: {e}")
    except Exception as e:
        logger.error(f"spaces_browser_routes: Unexpected error during S3 client initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: An unexpected error occurred with storage service: {e}")

@router.get("/list-objects", response_model=List[Dict[str, Any]])
async def list_objects_in_spaces(
    prefix: Optional[str] = Query(None, description="The prefix (folder path) to list objects from. E.g., 'mri_pipeline_outputs/YOUR_SCAN_NAME/' or 'mri_pipeline_outputs/YOUR_SCAN_NAME/processed/uncompressed/'"),
    s3_client: boto3.client = Depends(get_s3_client)
):
    """
    Lists objects (files and folders) within a specified prefix in your DigitalOcean Space.
    If no prefix is provided, it lists objects from the root of the bucket (use with caution for large buckets).
    The output includes object keys, sizes, and last modified timestamps.
    Common prefixes used by the pipeline:
    - `mri_pipeline_outputs/` (shows all pipeline run folders)
    - `mri_pipeline_outputs/YOUR_INPUT_FILENAME_WITHOUT_EXT/` (shows all outputs for a specific scan)
    - `mri_pipeline_outputs/YOUR_INPUT_FILENAME_WITHOUT_EXT/processed/`
    - `mri_pipeline_outputs/YOUR_INPUT_FILENAME_WITHOUT_EXT/processed/uncompressed/`
    - `mri_pipeline_outputs/YOUR_INPUT_FILENAME_WITHOUT_EXT/processed/segmented/`
    - `mri_pipeline_outputs/YOUR_INPUT_FILENAME_WITHOUT_EXT/processed/features/`
    """
    if not DO_SPACES_BUCKET_NAME:
        logger.error("list-objects endpoint: DO_SPACES_BUCKET_NAME not configured.")
        raise HTTPException(status_code=500, detail="Server configuration error: Bucket name not set.")

    # Corrected logger line:
    display_prefix = prefix if prefix else "(root)"
    logger.info(f"Listing objects in Spaces. Bucket: '{DO_SPACES_BUCKET_NAME}', Prefix: '{display_prefix}'")

    try:
        list_kwargs = {'Bucket': DO_SPACES_BUCKET_NAME}
        if prefix:
            # Ensure prefix for listing acts like a folder, but don't add double slash if already ends with one
            current_prefix = prefix.rstrip('/') + '/' if prefix else '' # Ensure trailing slash for folder-like listing
            list_kwargs['Prefix'] = current_prefix


        paginator = s3_client.get_paginator('list_objects_v2')
        listed_objects = []
        
        for page in paginator.paginate(**list_kwargs):
            if 'Contents' in page:
                for item in page['Contents']:
                    # Don't list the prefix itself if it's an empty "folder" object and it's exactly the prefix
                    # This check is for the case where the prefix itself might be an object (e.g. a file named like a folder)
                    if item['Key'] == list_kwargs.get('Prefix') and item['Size'] == 0:
                        continue
                    listed_objects.append({
                        "key": item['Key'],
                        "size_bytes": item['Size'],
                        "last_modified": item['LastModified'].isoformat() if isinstance(item['LastModified'], datetime) else str(item['LastModified'])
                    })
            # To list "common prefixes" (folders) as well, you could inspect page.get('CommonPrefixes')
            # This would allow showing subdirectories even if they are "empty" in terms of direct objects.

        if not listed_objects and prefix:
             # Check if the prefix itself exists as an object (e.g. an empty folder object)
            try:
                # Use the original prefix for head_object check, not the one with appended slash if it wasn't there originally
                s3_client.head_object(Bucket=DO_SPACES_BUCKET_NAME, Key=prefix)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.warning(f"Prefix '{prefix}' does not exist or is empty.")
                    # Return empty list if prefix doesn't exist or has no listable contents
                    # raise HTTPException(status_code=404, detail=f"Prefix '{prefix}' not found or no objects within.")

        logger.info(f"Found {len(listed_objects)} objects under prefix '{display_prefix}'.")
        return JSONResponse(content=listed_objects)

    except ParamValidationError as e:
        logger.error(f"Invalid parameters for list_objects_v2: Prefix='{prefix}'. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid prefix format: {e}")
    except ClientError as e:
        logger.error(f"S3 ClientError during list_objects: Prefix='{prefix}'. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list objects from storage: {e.response.get('Error', {}).get('Message', 'Unknown S3 error')}")
    except Exception as e:
        logger.error(f"Unexpected error listing objects from Spaces: Prefix='{prefix}'. Error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while listing objects: {e}")

# To include this router in your main FastAPI app (e.g., in app/main.py):
# from app.router import spaces_browser_routes # Assuming this file is in app/router/
# app.include_router(spaces_browser_routes.router)
