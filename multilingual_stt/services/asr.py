"""
Multilingual ASR service using SeamlessM4Tv2 model.
Handles model loading, VAD-based chunking, hallucination detection,
and dual transcription (native + English).
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import time
import uuid
import logging
from typing import Optional, Tuple, List
from transformers import SeamlessM4TProcessor, SeamlessM4Tv2Model
import numpy as np

from shared.utils.cache import TranscriptionCache
from shared.utils.audio import (
    load_audio, process_audio_with_vad, detect_hallucination
)
from shared.utils.metrics import (
    cache_hits_total, cache_misses_total, requests_duration_seconds,
    requests_total, chunks_processed_gauge, wer_gauge, cer_gauge,
    gpu_memory_used_mb, cpu_usage_percent, memory_usage_mb,
    preprocessing_duration_seconds, inference_duration_seconds,
    average_audio_duration_seconds, hallucination_events_total,
    transcription_errors_total, get_system_metrics
)

# Language configuration
LANGUAGE = "multilingual"
MODEL_ID = "facebook/seamless-m4t-v2-large"

# Global model cache (process-level persistence)
class ModelState:
    model = None
    processor = None

_transcription_cache = TranscriptionCache(max_size=10)


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(logger: logging.Logger):
    """
    Load SeamlessM4T model and processor with memory optimization.
    Uses float16 on GPU.
    """
    if ModelState.model is not None:
        return  # Already loaded
    
    device = get_device()
    
    # Clear any existing CUDA cache
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    try:
        # Load Processor
        ModelState.processor = SeamlessM4TProcessor.from_pretrained(MODEL_ID)
        
        # Load Model
        if device == "cuda":
            ModelState.model = SeamlessM4Tv2Model.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
        else:
            ModelState.model = SeamlessM4Tv2Model.from_pretrained(
                MODEL_ID,
                low_cpu_mem_usage=True
            )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Model loading failed")
    
    # Enable inference mode optimizations
    if device == "cuda":
        ModelState.model.eval()
        torch.cuda.empty_cache()
    
    logger.info(f"Model loaded on {device}")


def run_inference(
    vad_chunks: List[np.ndarray],
    tgt_lang: str,
    source_lang: str,
    logger: logging.Logger
) -> Tuple[str, int]:
    """
    Helper to run inference with chunking for a specific target language.
    
    Args:
        vad_chunks: List of audio chunks from VAD
        tgt_lang: Target language code
        source_lang: Source language code (for metrics)
        logger: Logger instance
    
    Returns:
        Tuple of (transcribed_text, chunks_processed)
    """
    transcribed_chunks = []
    device = get_device()
    
    # Process VAD chunks
    for i, chunk in enumerate(vad_chunks):
        if len(chunk) == 0:
            continue
        
        # Clear cache before each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(chunk).float()
            
            # Prepare inputs
            inputs = ModelState.processor(
                audios=audio_tensor,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Language-specific parameter tuning
            if source_lang == "yor":
                # Yoruba needs stronger constraints
                num_beams = 8
                repetition_penalty = 1.5
                no_repeat_ngram = 5
            elif source_lang == "ibo":
                # Igbo works well with default
                num_beams = 4
                repetition_penalty = 1.2
                no_repeat_ngram = 4
            else:
                # Default for other languages
                num_beams = 4
                repetition_penalty = 1.2
                no_repeat_ngram = 4
            
            # Generate with anti-hallucination params
            with torch.inference_mode():
                generated_output = ModelState.model.generate(
                    **inputs,
                    tgt_lang=tgt_lang,
                    generate_speech=False,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram,
                    num_beams=num_beams,
                    return_dict_in_generate=True
                )
            
            # Extract sequences from the output
            output_tokens = generated_output.sequences
            
            # Decode
            text = ModelState.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
            
            # Hallucination Check
            if detect_hallucination(text):
                hallucination_events_total.labels(language=source_lang).inc()
                logger.warning(f"Hallucination detected in chunk {i}: {text[:50]}...")
                
                # Retry with stronger penalties
                try:
                    with torch.inference_mode():
                        generated_output = ModelState.model.generate(
                            **inputs,
                            tgt_lang=tgt_lang,
                            generate_speech=False,
                            repetition_penalty=2.0,  # Stronger penalty
                            no_repeat_ngram_size=3,
                            num_beams=2,
                            return_dict_in_generate=True
                        )
                        output_tokens = generated_output.sequences
                        text = ModelState.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
                except Exception as e:
                    logger.error(f"Retry failed: {e}")
            
            transcribed_chunks.append(text)
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            transcription_errors_total.labels(language=source_lang, error_type="inference").inc()
        
        # Clear cache after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Combine transcriptions
    return " ".join(transcribed_chunks), len(vad_chunks)


def transcribe_audio(
    audio_bytes: bytes,
    source_lang: str,
    reference_text: Optional[str],
    logger: logging.Logger
) -> dict:
    """
    Transcribe audio bytes to text in native language and English.
    
    Args:
        audio_bytes: Audio file bytes
        source_lang: Source language code (e.g., "fuv", "yor", "ibo")
        reference_text: Optional ground truth for WER/CER calculation
        logger: Logger instance
    
    Returns:
        Dictionary with transcription results and metadata
    """
    # Generate unique request ID for tracing
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Create logger with request context
    request_logger = logging.LoggerAdapter(logger, {"request_id": request_id})
    
    # Check cache
    cached_result = _transcription_cache.get(audio_bytes, source_lang)
    if cached_result:
        latency = time.time() - start_time
        cached_result["cache_hit"] = True
        
        # Recalculate WER if reference text is provided
        if reference_text:
            try:
                import jiwer
                wer = jiwer.wer(reference_text, cached_result["native_text"])
                cer = jiwer.cer(reference_text, cached_result["native_text"])
                cached_result["wer"] = wer
                cached_result["cer"] = cer
                wer_gauge.labels(language=source_lang).set(wer)
                cer_gauge.labels(language=source_lang).set(cer)
            except Exception as e:
                request_logger.error(f"Error calculating WER/CER on cache hit: {e}")
        
        cache_hits_total.labels(language=source_lang).inc()
        requests_duration_seconds.labels(language=source_lang).observe(latency)
        requests_total.labels(language=source_lang, status="success").inc()
        request_logger.info("Cache hit", extra={"request_id": request_id, "latency_seconds": latency, "cache_hit": True})
        return cached_result
    
    # Cache miss
    cache_misses_total.labels(language=LANGUAGE).inc()
    
    # Ensure model is loaded
    if ModelState.model is None:
        load_model(logger)
    
    # Load and process audio (Preprocessing)
    preprocessing_start = time.time()
    raw_audio_array = load_audio(audio_bytes)
    
    # Apply VAD and Chunking
    vad_chunks = process_audio_with_vad(raw_audio_array)
    
    # Clear CUDA cache before transcription
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Calculate total duration
    sample_rate = 16000
    total_duration = len(raw_audio_array) / sample_rate
    average_audio_duration_seconds.labels(language=source_lang).observe(total_duration)
    
    preprocessing_duration = time.time() - preprocessing_start
    
    transcription_start_time = time.time()
    
    # 1. Native Transcription (ASR)
    native_text, chunks_processed = run_inference(vad_chunks, source_lang, source_lang, request_logger)
    
    # 2. English Translation (S2TT)
    english_text, _ = run_inference(vad_chunks, "eng", source_lang, request_logger)
    
    inference_duration = time.time() - transcription_start_time
    latency = time.time() - start_time
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get system metrics
    sys_metrics = get_system_metrics()
    
    # Calculate WER/CER if reference text is provided
    wer = 0.0
    cer = 0.0
    if reference_text:
        try:
            import jiwer
            wer = jiwer.wer(reference_text, native_text)
            cer = jiwer.cer(reference_text, native_text)
        except Exception as e:
            request_logger.error(f"Error calculating WER/CER: {e}")
    
    result_dict = {
        "native_text": native_text,
        "english_text": english_text,
        "latency_sec": latency,
        "preprocessing_sec": preprocessing_duration,
        "inference_sec": inference_duration,
        "cache_hit": False,
        "chunks_processed": chunks_processed,
        "cache_stats": _transcription_cache.stats(),
        "wer": wer,
        "cer": cer,
        "gpu_memory_used_mb": sys_metrics["gpu_memory_mb"],
    }
    
    # Update metrics
    requests_duration_seconds.labels(language=source_lang).observe(latency)
    preprocessing_duration_seconds.labels(language=source_lang).observe(preprocessing_duration)
    inference_duration_seconds.labels(language=source_lang).observe(inference_duration)
    
    requests_total.labels(language=source_lang, status="success").inc()
    chunks_processed_gauge.labels(language=source_lang).set(chunks_processed)
    wer_gauge.labels(language=source_lang).set(wer)
    cer_gauge.labels(language=source_lang).set(cer)
    gpu_memory_used_mb.labels(language=source_lang).set(sys_metrics["gpu_memory_mb"])
    cpu_usage_percent.labels(language=source_lang).set(sys_metrics["cpu_percent"])
    memory_usage_mb.labels(language=source_lang).set(sys_metrics["memory_mb"])
    
    # Log structured data
    request_logger.info(
        "Transcription completed",
        extra={
            "request_id": request_id,
            "latency_seconds": latency,
            "preprocessing_seconds": preprocessing_duration,
            "inference_seconds": inference_duration,
            "chunks_processed": chunks_processed,
            "cache_hit": False,
            "gpu_memory_used_mb": sys_metrics["gpu_memory_mb"],
            "wer": wer,
            "cer": cer,
        }
    )
    
    # Store in cache for future requests
    _transcription_cache.put(audio_bytes, result_dict, source_lang)
    
    return result_dict


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return _transcription_cache.stats()
