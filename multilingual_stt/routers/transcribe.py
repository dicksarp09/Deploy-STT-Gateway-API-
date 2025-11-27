"""
Transcription router for Multilingual STT service.
Provides endpoints for transcription with language selection, health checks, and metrics.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from shared.utils.metrics import (
    active_requests, rejected_requests_total, requests_total
)
from services.asr import transcribe_audio, get_cache_stats, LANGUAGE

# Router instance
router = APIRouter()

# Concurrency control
MAX_CONCURRENCY = 4
transcription_semaphore = None


def init_semaphore():
    """Initialize concurrency semaphore."""
    global transcription_semaphore
    if transcription_semaphore is None:
        transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


@router.get("/health")
async def health_endpoint():
    """
    Health check endpoint for Kubernetes liveness probe.
    """
    return JSONResponse(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "language": LANGUAGE,
        }
    )


@router.get("/ready")
async def ready_endpoint():
    """
    Readiness check endpoint for Kubernetes readiness probe.
    """
    return JSONResponse(
        {
            "status": "ready",
            "timestamp": time.time(),
            "language": LANGUAGE,
        }
    )


@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    source_lang: str = Form(default="fuv"),
    reference_text: Optional[str] = Form(default=None)
):
    """
    Transcribe audio file to native language and English.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
        source_lang: Source language code (default: "fuv" for Fulfulde)
        reference_text: Optional ground truth text for WER calculation
    
    Returns:
        JSON with:
        - native_text: Transcription in source language
        - english_text: Translation to English
        - latency_sec: Total latency in seconds
        - preprocessing_sec: Audio preprocessing time
        - inference_sec: Model inference time
        - cache_hit: Whether result was from cache
        - chunks_processed: Number of audio chunks processed
        - wer: Word Error Rate (if reference provided)
        - cer: Character Error Rate (if reference provided)
        - gpu_memory_used_mb: GPU memory usage in MB
    """
    from shared.utils.logging import setup_logger
    logger = setup_logger("multilingual_st")
    
    request_id = str(uuid.uuid4())
    
    # Initialize semaphore if needed
    init_semaphore()
    
    try:
        # Acquire semaphore (concurrency control)
        if transcription_semaphore.locked():
            # Max concurrency reached
            rejected_requests_total.labels(language=LANGUAGE).inc()
            logger.warning(
                "Request rejected due to concurrency limit",
                extra={"request_id": request_id}
            )
            requests_total.labels(language=LANGUAGE, status="rejected").inc()
            raise HTTPException(
                status_code=503,
                detail=f"Service busy. Maximum {MAX_CONCURRENCY} concurrent requests."
            )
        
        async with transcription_semaphore:
            active_requests.labels(language=source_lang).inc()
            start_time = time.time()
            
            audio_bytes = await file.read()
            result = transcribe_audio(audio_bytes, source_lang, reference_text, logger)
            
            latency = time.time() - start_time
            
            # Log successful request
            logger.info(
                "Transcription request completed",
                extra={
                    "request_id": request_id,
                    "latency_seconds": latency,
                }
            )
            
            active_requests.labels(language=source_lang).dec()
            return JSONResponse(result)
            
    except HTTPException:
        raise
    except Exception as e:
        requests_total.labels(language=LANGUAGE, status="error").inc()
        logger.error(
            f"Transcription error: {str(e)}",
            extra={"request_id": request_id}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy/stats")
async def accuracy_stats_endpoint():
    """
    Get accuracy statistics (WER/CER) and cache statistics.
    """
    cache_stats = get_cache_stats()
    return JSONResponse(
        {
            "cache_stats": cache_stats,
            "note": "WER/CER requires reference transcriptions (ground truth).",
        }
    )
