"""
Yoruba ASR service using CLEAR-Global w2v-bert model.
Handles model loading, caching, and transcription with GPU optimization.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import time
import uuid
import logging
from typing import Optional
from transformers import pipeline

from shared.utils.cache import TranscriptionCache
from shared.utils.audio import (
    load_audio, detect_and_remove_silence, split_audio_into_chunks,
    get_optimal_chunk_size
)
from shared.utils.metrics import (
    cache_hits_total, cache_misses_total, requests_duration_seconds,
    requests_total, chunks_processed_gauge, wer_gauge, cer_gauge,
    gpu_memory_used_mb, cpu_usage_percent, memory_usage_mb,
    get_system_metrics
)

# Language configuration
LANGUAGE = "yoruba"
MODEL_ID = "CLEAR-Global/w2v-bert-2.0-yoruba_naijavoices_1h"

# Global model cache (process-level persistence)
_asr_pipe = None
_transcription_cache = TranscriptionCache(max_size=10)


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(logger: logging.Logger):
    """
    Load ASR model pipeline with memory optimization.
    Uses float16 on GPU with fallback to float32.
    """
    global _asr_pipe
    
    if _asr_pipe is not None:
        return  # Already loaded
    
    device = get_device()
    
    # Clear any existing CUDA cache before loading model
    if device == "cuda":
        torch.cuda.empty_cache()
        # Set memory fraction to leave some headroom
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Try to load with float16 for memory efficiency
    try:
        if device == "cuda":
            # Load model with float16
            _asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                device=0,
                torch_dtype=torch.float16,
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16,
                }
            )
            # Convert model to half precision if not already
            if hasattr(_asr_pipe.model, 'half'):
                _asr_pipe.model = _asr_pipe.model.half()
        else:
            _asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                device=-1,
            )
    except Exception as e:
        # Fallback to default loading if float16 fails
        logger.warning(f"Could not load with float16: {e}. Using default precision.")
        _asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device=0 if device == "cuda" else -1,
            model_kwargs={"low_cpu_mem_usage": True}
        )
    
    # Enable inference mode optimizations
    if device == "cuda":
        _asr_pipe.model.eval()
        torch.cuda.empty_cache()
    
    logger.info(f"Model loaded on {device}")


def transcribe_audio(
    audio_bytes: bytes,
    logger: logging.Logger
) -> dict:
    """
    Transcribe audio bytes to text.
    
    Args:
        audio_bytes: Audio file bytes
        logger: Logger instance
    
    Returns:
        Dictionary with transcription results and metadata
    """
    global _asr_pipe
    
    # Generate unique request ID for tracing
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Create logger with request context
    request_logger = logging.LoggerAdapter(logger, {"request_id": request_id})
    
    # Check cache first
    cached_result = _transcription_cache.get(audio_bytes)
    if cached_result:
        latency = time.time() - start_time
        cached_result["cache_hit"] = True
        
        # Update metrics
        cache_hits_total.labels(language=LANGUAGE).inc()
        requests_duration_seconds.labels(language=LANGUAGE).observe(latency)
        requests_total.labels(language=LANGUAGE, status="success").inc()
        
        # Log cache hit
        request_logger.info(
            "Cache hit",
            extra={
                "request_id": request_id,
                "latency_seconds": latency,
                "cache_hit": True,
            }
        )
        
        return cached_result
    
    # Cache miss
    cache_misses_total.labels(language=LANGUAGE).inc()
    
    # Ensure model is loaded
    if _asr_pipe is None:
        load_model(logger)
    
    # Load and process audio
    audio_array = load_audio(audio_bytes)
    
    # Apply silence removal
    audio_array = detect_and_remove_silence(audio_array, threshold_db=-40, sample_rate=16000)
    
    # Clear CUDA cache before transcription to reduce fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Use dynamic chunk sizing
    chunk_length_seconds = get_optimal_chunk_size()
    overlap_seconds = 2
    sample_rate = 16000
    
    # Calculate total duration
    total_duration = len(audio_array) / sample_rate
    
    transcription_start_time = time.time()
    
    # For short audio (< chunk_length_seconds), process directly
    if total_duration <= chunk_length_seconds:
        with torch.inference_mode():
            result = _asr_pipe(audio_array)
        full_text = result["text"]
        chunks_processed = 1
    else:
        # Split into chunks and process each separately
        chunks = split_audio_into_chunks(
            audio_array,
            chunk_length_seconds=chunk_length_seconds,
            overlap_seconds=overlap_seconds,
            sample_rate=sample_rate
        )
        
        transcribed_chunks = []
        for i, chunk in enumerate(chunks):
            # Clear cache before each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Transcribe chunk
            with torch.inference_mode():
                chunk_result = _asr_pipe(chunk)
            
            transcribed_chunks.append(chunk_result["text"])
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Combine transcriptions (simple concatenation with space)
        full_text = " ".join(transcribed_chunks)
        chunks_processed = len(chunks)
    
    transcription_latency = time.time() - transcription_start_time
    latency = time.time() - start_time
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get system metrics
    sys_metrics = get_system_metrics()
    
    # Note: WER/CER calculation requires a reference transcription
    wer = 0.0
    cer = 0.0
    
    result_dict = {
        "text": full_text,
        "latency_sec": latency,
        "cache_hit": False,
        "chunks_processed": chunks_processed,
        "cache_stats": _transcription_cache.stats(),
        "wer": wer,
        "cer": cer,
        "gpu_memory_used_mb": sys_metrics["gpu_memory_mb"],
    }
    
    # Update metrics
    requests_duration_seconds.labels(language=LANGUAGE).observe(latency)
    requests_total.labels(language=LANGUAGE, status="success").inc()
    chunks_processed_gauge.labels(language=LANGUAGE).set(chunks_processed)
    wer_gauge.labels(language=LANGUAGE).set(wer)
    cer_gauge.labels(language=LANGUAGE).set(cer)
    gpu_memory_used_mb.labels(language=LANGUAGE).set(sys_metrics["gpu_memory_mb"])
    cpu_usage_percent.labels(language=LANGUAGE).set(sys_metrics["cpu_percent"])
    memory_usage_mb.labels(language=LANGUAGE).set(sys_metrics["memory_mb"])
    
    # Log structured data
    request_logger.info(
        "Transcription completed",
        extra={
            "request_id": request_id,
            "latency_seconds": latency,
            "chunks_processed": chunks_processed,
            "cache_hit": False,
            "gpu_memory_used_mb": sys_metrics["gpu_memory_mb"],
            "wer": wer,
            "cer": cer,
        }
    )
    
    # Store in cache for future requests
    _transcription_cache.put(audio_bytes, result_dict)
    
    return result_dict


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return _transcription_cache.stats()
