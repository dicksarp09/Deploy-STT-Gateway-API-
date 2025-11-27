# FastAPI STT Microservices

Production-ready FastAPI microservices for Speech-to-Text (STT) transcription, converted from Modal-based endpoints. Supports Hausa, Yoruba, and multilingual transcription with full observability.

## 🎯 Features

### Core Functionality
- **Hausa STT**: CLEAR-Global w2v-bert model for Hausa transcription
- **Yoruba STT**: CLEAR-Global w2v-bert model for Yoruba transcription
- **Multilingual STT**: SeamlessM4Tv2 for 100+ languages with native + English output

### Production Features
- ✅ **Structured JSON Logging**: Request tracing with unique IDs
- ✅ **Prometheus Metrics**: Full observability (latency, cache, GPU, WER/CER)
- ✅ **LRU Caching**: SHA256-based caching with hit rate tracking
- ✅ **Concurrency Control**: asyncio.Semaphore limiting (4 concurrent requests)
- ✅ **GPU Optimization**: Dynamic chunk sizing, float16 support, memory management
- ✅ **VAD Processing**: Silero VAD for silence removal (multilingual)
- ✅ **Hallucination Detection**: Automatic retry with stronger penalties
- ✅ **Health Checks**: `/health` and `/ready` endpoints for Kubernetes
- ✅ **WER/CER Calculation**: Accuracy metrics with ground truth

## 📁 Project Structure

```
e:/FastAPI/
├── shared/
│   └── utils/
│       ├── logging.py      # JSON formatter
│       ├── metrics.py      # Prometheus metrics
│       ├── cache.py        # LRU cache
│       └── audio.py        # Audio processing, VAD, WER/CER
├── hausa_stt/
│   ├── main.py
│   ├── services/asr.py
│   ├── routers/transcribe.py
│   └── requirements.txt
├── yoruba_stt/
│   ├── main.py
│   ├── services/asr.py
│   ├── routers/transcribe.py
│   └── requirements.txt
└── multilingual_stt/
    ├── main.py
    ├── services/asr.py
    ├── routers/transcribe.py
    └── requirements.txt
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- ffmpeg installed and in PATH
- CUDA-capable GPU (optional, will use CPU otherwise)

### Install Dependencies

**Hausa STT:**
```bash
cd hausa_stt
pip install -r requirements.txt
```

**Yoruba STT:**
```bash
cd yoruba_stt
pip install -r requirements.txt
```

**Multilingual STT:**
```bash
cd multilingual_stt
pip install -r requirements.txt
```

## 🏃 Running Services

### Hausa STT Service
```bash
cd hausa_stt
python main.py
```
Service runs on `http://localhost:8000`

### Yoruba STT Service
```bash
cd yoruba_stt
python main.py
```
Service runs on `http://localhost:8000`

### Multilingual STT Service
```bash
cd multilingual_stt
python main.py
```
Service runs on `http://localhost:8000`

### Custom Port
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

## 📡 API Endpoints

### POST /transcribe

**Hausa/Yoruba:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "text": "Transcribed text...",
  "latency_sec": 2.5,
  "cache_hit": false,
  "chunks_processed": 3,
  "wer": 0.0,
  "cer": 0.0,
  "gpu_memory_used_mb": 4096.5,
  "cache_stats": {
    "hits": 5,
    "misses": 10,
    "total": 15,
    "hit_rate": 33.33,
    "size": 8
  }
}
```

**Multilingual:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "source_lang=fuv" \
  -F "reference_text=Optional ground truth"
```

**Response:**
```json
{
  "native_text": "Native language transcription...",
  "english_text": "English translation...",
  "latency_sec": 3.2,
  "preprocessing_sec": 0.5,
  "inference_sec": 2.7,
  "cache_hit": false,
  "chunks_processed": 4,
  "wer": 0.15,
  "cer": 0.08,
  "gpu_memory_used_mb": 8192.0
}
```

### GET /health
Health check for liveness probes.

```bash
curl http://localhost:8000/health
```

### GET /ready
Readiness check for readiness probes.

```bash
curl http://localhost:8000/ready
```

### GET /metrics
Prometheus metrics endpoint.

```bash
curl http://localhost:8000/metrics
```

**Metrics Provided:**
- `stt_requests_total` - Total requests by language and status
- `stt_latency_seconds` - Request latency histogram
- `stt_preprocessing_duration_seconds` - Preprocessing time
- `stt_inference_duration_seconds` - Inference time
- `stt_active_requests` - Current active requests
- `stt_cache_hits_total` / `stt_cache_misses_total` - Cache performance
- `stt_wer` / `stt_cer` - Accuracy metrics
- `stt_chunks_processed` - Chunks per request
- `stt_gpu_memory_used_mb` - GPU memory usage
- `stt_cpu_usage_percent` - CPU usage
- `stt_memory_usage_mb` - RAM usage
- `stt_rejected_requests_total` - Rejected due to concurrency
- `stt_hallucination_events_total` - Detected hallucinations
- `stt_audio_duration_seconds` - Audio duration histogram

### GET /accuracy/stats
Cache and accuracy statistics.

```bash
curl http://localhost:8000/accuracy/stats
```

## ⚙️ Configuration

### Environment Variables

```bash
# CUDA memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU device selection (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0
```

### Concurrency Limits

Edit `MAX_CONCURRENCY` in `routers/transcribe.py`:
```python
MAX_CONCURRENCY = 4  # Adjust based on GPU memory
```

### Cache Size

Edit cache size in `services/asr.py`:
```python
_transcription_cache = TranscriptionCache(max_size=10)
```

## 🐳 Docker Deployment

**Dockerfile Example (Hausa STT):**
```dockerfile
FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared utilities
COPY shared/ /app/shared/

# Copy service files
COPY hausa_stt/ /app/hausa_stt/

# Install dependencies
RUN pip install --no-cache-dir -r hausa_stt/requirements.txt

# Set environment
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app/hausa_stt

# Expose port
EXPOSE 8000

# Run service
CMD ["python", "main.py"]
```

**Build and Run:**
```bash
docker build -t hausa-stt .
docker run -p 8000:8000 --gpus all hausa-stt
```

## 🔍 Monitoring

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'hausa-stt'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Import metrics and create panels for:
- Request rate and latency percentiles
- Cache hit rate over time
- GPU memory usage
- WER/CER trends
- Concurrency and rejection rate

## 🧪 Testing

**Test Health:**
```bash
curl http://localhost:8000/health
```

**Test Transcription:**
```bash
# Create test audio (requires sox)
sox -n -r 16000 -c 1 test.wav synth 3 sine 440

# Transcribe
curl -X POST "http://localhost:8000/transcribe" -F "file=@test.wav"
```

## 📊 Performance Optimization

### GPU Memory Management
- Models loaded with float16 on GPU
- Dynamic chunk sizing based on available memory
- CUDA cache cleared before/after inference
- Memory fraction set to 0.95 for headroom

### Chunking Strategy
- **Hausa/Yoruba**: 20-30s chunks with 2s overlap
- **Multilingual**: VAD-based chunks (10-18s)
- Dynamic sizing based on GPU memory

### Caching
- LRU cache with SHA256 keys
- Default size: 10 items
- Separate cache per language (multilingual)

## 🔧 Troubleshooting

### Model Loading Issues
```python
# Check if model loads
from services import asr
from shared.utils.logging import setup_logger
logger = setup_logger("test")
asr.load_model(logger)
```

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
- Reduce `MAX_CONCURRENCY`
- Reduce cache size
- Use smaller chunk sizes
- Ensure `torch.cuda.empty_cache()` is called

## 📝 Differences from Modal Version

| Feature | Modal | FastAPI |
|---------|-------|---------|
| Deployment | Modal cloud | Any server/container |
| Model Caching | Container persistence | Process-level globals |
| Concurrency | `@app.function` | `asyncio.Semaphore` |
| GPU Allocation | Modal config | CUDA device selection |
| Scaling | Auto-scaling | Manual/K8s HPA |
| Cold Starts | Yes | No (if kept running) |

## 🤝 Contributing

1. Follow existing code structure
2. Maintain feature parity with Modal version
3. Add tests for new features
4. Update metrics and logging

## 📄 License

Same as original Modal implementation.

## 🙏 Acknowledgments

- CLEAR-Global for w2v-bert models
- Meta for SeamlessM4T
- Original Modal implementation authors
