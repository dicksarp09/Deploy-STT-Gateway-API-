"""
Hausa STT FastAPI Application.
Standalone FastAPI service for Hausa speech-to-text transcription.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import transcribe
from services import asr
from shared.utils.logging import setup_logger


# Setup logger
logger = setup_logger("hausa_stt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Hausa STT service...")
    transcribe.init_semaphore()
    
    # Preload model on startup (optional, will load on first request otherwise)
    try:
        asr.load_model(logger)
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Could not preload model: {e}. Will load on first request.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hausa STT service...")


# Create FastAPI app
app = FastAPI(
    title="Hausa STT API",
    version="3.0",
    description="Hausa Speech-to-Text API using CLEAR-Global w2v-bert model",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transcribe.router, tags=["transcription"])


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Hausa STT API",
        "version": "3.0",
        "model": asr.MODEL_ID,
        "language": asr.LANGUAGE,
        "endpoints": {
            "transcribe": "/transcribe",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "stats": "/accuracy/stats"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Set environment variable for CUDA memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
