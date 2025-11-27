"""
Structured JSON logging utilities for STT services.
Provides custom JSON formatter for structured logging with request context.
"""
import logging
import json
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "latency_seconds"):
            log_data["latency_seconds"] = record.latency_seconds
        if hasattr(record, "chunks_processed"):
            log_data["chunks_processed"] = record.chunks_processed
        if hasattr(record, "cache_hit"):
            log_data["cache_hit"] = record.cache_hit
        if hasattr(record, "gpu_memory_used_mb"):
            log_data["gpu_memory_used_mb"] = record.gpu_memory_used_mb
        if hasattr(record, "wer"):
            log_data["wer"] = record.wer
        if hasattr(record, "cer"):
            log_data["cer"] = record.cer
        if hasattr(record, "preprocessing_seconds"):
            log_data["preprocessing_seconds"] = record.preprocessing_seconds
        if hasattr(record, "inference_seconds"):
            log_data["inference_seconds"] = record.inference_seconds
        if hasattr(record, "hallucinations"):
            log_data["hallucinations"] = record.hallucinations
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a JSON logger.
    
    Args:
        name: Logger name (e.g., "hausa_stt", "yoruba_stt", "multilingual_st")
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger
