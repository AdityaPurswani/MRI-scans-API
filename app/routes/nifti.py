# app/api/nifti_routes.py
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response, JSONResponse
import os
import base64
from typing import List
import logging

router = APIRouter(prefix="/api", tags=["NIFTI"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define allowed directories for security
ALLOWED_DIRS = [
    "/Users/adityapurswani/Documents/MRIscansAPI/processed/uncompressed",
    # Add other allowed directories
]

@router.get("/nifti")
async def get_nifti_file(path: str = Query(...)):
    """
    Serve a NIFTI file from a specified path.
    """
    logger.info(f"Requested NIFTI file at path: {path}")
    
    # Security check - only allow access to certain directories
    is_allowed = any(path.startswith(dir) for dir in ALLOWED_DIRS)
    if not is_allowed:
        logger.warning(f"Access denied to path: {path}")
        raise HTTPException(status_code=403, detail="Access to this file path is not allowed")
    
    # Check if file exists
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Read file
    try:
        logger.info(f"Reading file: {path}")
        file_size = os.path.getsize(path)
        
        # For very large files, we might want to limit size
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            logger.warning(f"File too large: {file_size} bytes")
            raise HTTPException(status_code=413, detail="File too large to process")
            
        logger.info(f"File size: {file_size} bytes")
        
        with open(path, "rb") as f:
            file_content = f.read()
            
        logger.info(f"Successfully read file: {path}, returning {len(file_content)} bytes")
        
        # Return raw binary response with explicit headers to prevent download
        return Response(
            content=file_content,
            media_type="application/x-binary",  # Changed from octet-stream
            headers={
                "X-Content-Type-Options": "nosniff",
                "Content-Length": str(len(file_content)),
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.error(f"Failed to read file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


# Alternative approach: Return base64 encoded data
@router.get("/nifti-base64")
async def get_nifti_file_base64(path: str = Query(...)):
    """
    Serve a NIFTI file from a specified path as base64 encoded data.
    """
    logger.info(f"Requested NIFTI file (base64) at path: {path}")
    
    # Security check - only allow access to certain directories
    is_allowed = any(path.startswith(dir) for dir in ALLOWED_DIRS)
    if not is_allowed:
        logger.warning(f"Access denied to path: {path}")
        raise HTTPException(status_code=403, detail="Access to this file path is not allowed")
    
    # Check if file exists
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Read file
    try:
        logger.info(f"Reading file: {path}")
        file_size = os.path.getsize(path)
        
        # For very large files, we might want to limit size
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            logger.warning(f"File too large: {file_size} bytes")
            raise HTTPException(status_code=413, detail="File too large to process")
            
        logger.info(f"File size: {file_size} bytes")
        
        with open(path, "rb") as f:
            file_content = f.read()
            
        # Encode file content as base64
        base64_content = base64.b64encode(file_content).decode('utf-8')
        
        logger.info(f"Successfully read file: {path}, encoded as base64")
        
        # Return JSON response with base64 data
        return JSONResponse({
            "filename": os.path.basename(path),
            "file_size": file_size,
            "base64_data": base64_content
        })
    except Exception as e:
        logger.error(f"Failed to read file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")