"""
Prometheus metrics definitions and system metrics collection.
Provides all metrics used across STT services.
"""
import psutil
from prometheus_client import Counter, Histogram, Gauge

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False


# --- Prometheus Metrics ---
requests_total = Counter(
    "stt_requests_total",
    "Total number of transcription requests",
    ["language", "status"]
)

requests_duration_seconds = Histogram(
    "stt_latency_seconds",
    "Transcription latency in seconds",
    ["language"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
)

preprocessing_duration_seconds = Histogram(
    "stt_preprocessing_duration_seconds",
    "Audio preprocessing latency in seconds",
    ["language"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2)
)

inference_duration_seconds = Histogram(
    "stt_inference_duration_seconds",
    "Model inference latency in seconds",
    ["language"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30)
)

active_requests = Gauge(
    "stt_active_requests",
    "Number of active requests",
    ["language"]
)

cache_hits_total = Counter(
    "stt_cache_hits_total",
    "Total cache hits",
    ["language"]
)

cache_misses_total = Counter(
    "stt_cache_misses_total",
    "Total cache misses",
    ["language"]
)

wer_gauge = Gauge(
    "stt_wer",
    "Word Error Rate (WER) for last transcription",
    ["language"]
)

cer_gauge = Gauge(
    "stt_cer",
    "Character Error Rate (CER) for last transcription",
    ["language"]
)

chunks_processed_gauge = Gauge(
    "stt_chunks_processed",
    "Number of chunks processed in last request",
    ["language"]
)

gpu_memory_used_mb = Gauge(
    "stt_gpu_memory_used_mb",
    "GPU memory used in MB",
    ["language"]
)

cpu_usage_percent = Gauge(
    "stt_cpu_usage_percent",
    "CPU usage percentage",
    ["language"]
)

memory_usage_mb = Gauge(
    "stt_memory_usage_mb",
    "Memory usage in MB",
    ["language"]
)

rejected_requests_total = Counter(
    "stt_rejected_requests_total",
    "Total rejected requests due to concurrency limits",
    ["language"]
)

hallucination_events_total = Counter(
    "stt_hallucination_events_total",
    "Total number of detected hallucination events",
    ["language"]
)

transcription_errors_total = Counter(
    "stt_transcription_errors_total",
    "Total transcription errors",
    ["language", "error_type"]
)

average_audio_duration_seconds = Histogram(
    "stt_audio_duration_seconds",
    "Audio duration in seconds",
    ["language"],
    buckets=(1, 5, 10, 30, 60, 120, 300)
)

language_confidence_avg = Gauge(
    "stt_language_confidence_avg",
    "Average language detection confidence",
    ["language"]
)


def get_system_metrics() -> dict:
    """
    Collect CPU, RAM, and GPU usage metrics.
    
    Returns:
        Dictionary with cpu_percent, memory_mb, and gpu_memory_mb
    """
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_mb": psutil.virtual_memory().used / (1024 ** 2),
        "gpu_memory_mb": 0
    }
    
    if NVIDIA_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics["gpu_memory_mb"] = mem_info.used / (1024 ** 2)
                pynvml.nvmlShutdown()
        except Exception:
            pass  # Silently fail if GPU metrics unavailable
    
    return metrics
