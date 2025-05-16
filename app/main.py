from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
from app.routes import upload as upload_routes
from app.routes import nifti as nifti_routes
from app.routes import spaces_browser_routes
from app.routes import pipeline_trigger_routes

# --- Load environment variables from .env file ---
# This should be one of the first things your application does.
# It ensures that all modules have access to the environment variables when they are imported.
try:
    # Assuming main.py is in app/ and .env is in the project root (MRISCANSAPI/)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"main.py: .env loaded from: {dotenv_path}")
    else:
        logging.warning(f"main.py: .env file not found at {dotenv_path}. Relying on environment variables being set externally.")
except Exception as e:
    logging.error(f"main.py: Error loading .env file: {e}")


# Import your routers
# Ensure these paths match your project structure (app/router/...)

app = FastAPI(
    title="MRI Processing API",
    description="An API for MRI preprocessing, segmentation, and volumetric feature extraction",
    version="1.1.0", # Updated version
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
# Adjust allow_origins as needed for your frontend application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, restrict this to your frontend's domain.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configure logging (basic global setup)
# You might want a more sophisticated logging setup for production.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("FastAPI application starting up...")

# Include your API routers
app.include_router(upload_routes.router)
app.include_router(nifti_routes.router)
app.include_router(spaces_browser_routes.router)
app.include_router(pipeline_trigger_routes.router)
# if 'process_routes' in locals() and hasattr(process_routes, 'router'): # If you have a process.py router
#     app.include_router(process_routes.router)

@app.get("/", tags=['Root'])
async def root():
    """
    Root endpoint for the MRI Processing API.
    """
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the MRI Processing API with DigitalOcean Spaces!"}

# Example of how to run this app (if this file is executed directly, though typically done via uvicorn command):
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Uvicorn server directly from main.py (for development).")
#     # Ensure .env is loaded if running this way and it's not done above.
#     # The load_dotenv() at the top should handle this.
#     uvicorn.run(app, host="0.0.0.0", port=8000)