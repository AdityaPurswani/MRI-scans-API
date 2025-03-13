from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload

app = FastAPI(
    title="MRI Processing API",
    description="An API for MRI preprocessing, segmentation, and volumetric feature extraction.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the MRI Processing API!"}
