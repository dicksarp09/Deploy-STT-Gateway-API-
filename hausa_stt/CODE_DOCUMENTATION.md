# Hausa STT Service - Code Documentation

Complete code documentation for the Hausa Speech-to-Text service.

## 📁 Project Structure

```
hausa_stt/
├── main.py                    # FastAPI application entry point
├── services/
│   ├── __init__.py           # Package marker
│   └── asr.py                # ASR model and transcription logic
├── routers/
│   ├── __init__.py           # Package marker
│   └── transcribe.py         # API endpoint definitions
└── requirements.txt          # Python dependencies
```

---

## 📄 main.py

**Purpose:** FastAPI application initialization and configuration.

### Imports
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
```
- `FastAPI`: Main application class
- `CORSMiddleware`: Enables cross-origin requests
- `asynccontextmanager`: Manages startup/shutdown events

### Key Components

#### 1. `lifespan()` Context Manager
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Hausa STT service...")
    transcribe.init_semaphore()
    asr.load_model(logger)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hausa STT service...")
```

**What it does:**
- **Startup Phase**: 
  - Logs service start
  - Initializes concurrency control semaphore
  - Preloads the ASR model into memory (optional but recommended)
- **Shutdown Phase**: 
  - Logs service shutdown
  - Cleans up resources

**Why it's important:** Ensures model is loaded once at startup, avoiding cold starts on first request.

#### 2. FastAPI App Creation
```python
app = FastAPI(
    title="Hausa STT API",
    version="3.0",
    description="Hausa Speech-to-Text API using CLEAR-Global w2v-bert model",
    lifespan=lifespan
)
```

**What it does:**
- Creates the main FastAPI application
- Sets API metadata (title, version, description)
- Attaches lifespan events for startup/shutdown

#### 3. CORS Middleware
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**What it does:**
- Allows requests from any origin (`*`)
- Enables cookies/authentication headers
- Permits all HTTP methods (GET, POST, etc.)
- Allows all request headers

**Security Note:** In production, restrict `allow_origins` to specific domains.

#### 4. Router Registration
```python
app.include_router(transcribe.router, tags=["transcription"])
```

**What it does:**
- Registers all endpoints from `transcribe.py`
- Tags them as "transcription" in API docs

#### 5. Root Endpoint
```python
@app.get("/")
async def root():
    return {
        "service": "Hausa STT API",
        "version": "3.0",
        "model": asr.MODEL_ID,
        "language": asr.LANGUAGE,
        "endpoints": {...}
    }
```

**What it does:**
- Provides service information at root URL
- Lists available endpoints
- Shows model and language configuration

#### 6. Main Execution Block
```python
if __name__ == "__main__":
    import uvicorn
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
```

**What it does:**
- Sets CUDA memory allocation strategy for better GPU memory management
- Runs the FastAPI app using Uvicorn ASGI server
- Listens on all interfaces (`0.0.0.0`) on port 8000
- Disables auto-reload for production stability

---

## 📄 services/asr.py

**Purpose:** Core ASR functionality - model loading, caching, and transcription.

### Global Variables

```python
LANGUAGE = "hausa"
MODEL_ID = "CLEAR-Global/w2v-bert-2.0-hausa_100_400h_yourtts"
_asr_pipe = None
_transcription_cache = TranscriptionCache(max_size=10)
```

**What they do:**
- `LANGUAGE`: Identifies the service language for metrics
- `MODEL_ID`: HuggingFace model identifier
- `_asr_pipe`: Global variable for process-level model caching (replaces Modal container persistence)
- `_transcription_cache`: LRU cache for storing recent transcriptions

### Functions

#### 1. `get_device()`
```python
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
```

**What it does:**
- Checks if CUDA-capable GPU is available
- Returns `"cuda"` if GPU found, otherwise `"cpu"`

**Why it's important:** Determines where to load the model (GPU for speed, CPU as fallback).

#### 2. `load_model(logger)`
```python
def load_model(logger: logging.Logger):
    global _asr_pipe
    
    if _asr_pipe is not None:
        return  # Already loaded
    
    device = get_device()
    
    # Clear CUDA cache
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Try float16 first
    try:
        if device == "cuda":
            _asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                device=0,
                torch_dtype=torch.float16,
                model_kwargs={"low_cpu_mem_usage": True, "torch_dtype": torch.float16}
            )
            if hasattr(_asr_pipe.model, 'half'):
                _asr_pipe.model = _asr_pipe.model.half()
        else:
            _asr_pipe = pipeline("automatic-speech-recognition", model=MODEL_ID, device=-1)
    except Exception as e:
        # Fallback to float32
        logger.warning(f"Could not load with float16: {e}. Using default precision.")
        _asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device=0 if device == "cuda" else -1,
            model_kwargs={"low_cpu_mem_usage": True}
        )
    
    # Optimize for inference
    if device == "cuda":
        _asr_pipe.model.eval()
        torch.cuda.empty_cache()
```

**What it does:**
1. **Checks if model already loaded** - Avoids reloading on subsequent calls
2. **Clears GPU memory** - Frees up space before loading
3. **Sets memory fraction** - Reserves 95% of GPU memory, leaving 5% headroom
4. **Tries float16 loading** - Half-precision for memory efficiency (2x less memory)
5. **Converts model to half precision** - If not already done
6. **Fallback to float32** - If float16 fails (some GPUs don't support it)
7. **Sets eval mode** - Disables dropout/batch norm for inference
8. **Final cache clear** - Ensures clean state

**Why float16?** Uses half the memory, allowing larger batch sizes or longer audio chunks.

#### 3. `transcribe_audio(audio_bytes, logger)`

**Purpose:** Main transcription function with caching, chunking, and metrics.

##### Step 1: Request Setup
```python
request_id = str(uuid.uuid4())
start_time = time.time()
request_logger = logging.LoggerAdapter(logger, {"request_id": request_id})
```

**What it does:**
- Generates unique ID for request tracing
- Records start time for latency calculation
- Creates logger with request context

##### Step 2: Cache Check
```python
cached_result = _transcription_cache.get(audio_bytes)
if cached_result:
    latency = time.time() - start_time
    cached_result["cache_hit"] = True
    
    # Update metrics
    cache_hits_total.labels(language=LANGUAGE).inc()
    requests_duration_seconds.labels(language=LANGUAGE).observe(latency)
    requests_total.labels(language=LANGUAGE, status="success").inc()
    
    return cached_result
```

**What it does:**
- Checks if this exact audio was transcribed before (using SHA256 hash)
- If found, updates metrics and returns cached result immediately
- Avoids expensive model inference for duplicate requests

**Performance impact:** Cache hits return in <10ms vs. seconds for inference.

##### Step 3: Cache Miss Handling
```python
cache_misses_total.labels(language=LANGUAGE).inc()

if _asr_pipe is None:
    load_model(logger)
```

**What it does:**
- Records cache miss in metrics
- Ensures model is loaded (lazy loading if not preloaded)

##### Step 4: Audio Processing
```python
audio_array = load_audio(audio_bytes)
audio_array = detect_and_remove_silence(audio_array, threshold_db=-40, sample_rate=16000)
```

**What it does:**
- **`load_audio()`**: Converts audio bytes to numpy array using ffmpeg
  - Resamples to 16kHz
  - Converts to mono
  - Normalizes to float32 [-1.0, 1.0]
- **`detect_and_remove_silence()`**: Removes silent segments
  - Calculates RMS energy in 25ms frames
  - Converts to dB scale
  - Removes frames below threshold
  - Reduces processing time by 20-40%

##### Step 5: GPU Memory Management
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

**What it does:**
- **`empty_cache()`**: Frees unused cached memory
- **`synchronize()`**: Waits for all GPU operations to complete

**Why it's important:** Prevents memory fragmentation, reduces OOM errors.

##### Step 6: Dynamic Chunking
```python
chunk_length_seconds = get_optimal_chunk_size()
overlap_seconds = 2
sample_rate = 16000

total_duration = len(audio_array) / sample_rate
```

**What it does:**
- **`get_optimal_chunk_size()`**: Returns 20-30s based on available GPU memory
  - <5GB available → 20s chunks
  - 5-10GB → 25s chunks
  - >10GB → 30s chunks
- **Overlap**: 2-second overlap prevents cutting words at boundaries
- **Duration calculation**: Determines if chunking is needed

##### Step 7: Transcription Logic
```python
if total_duration <= chunk_length_seconds:
    # Short audio - process directly
    with torch.inference_mode():
        result = _asr_pipe(audio_array)
    full_text = result["text"]
    chunks_processed = 1
else:
    # Long audio - chunk processing
    chunks = split_audio_into_chunks(
        audio_array,
        chunk_length_seconds=chunk_length_seconds,
        overlap_seconds=overlap_seconds,
        sample_rate=sample_rate
    )
    
    transcribed_chunks = []
    for i, chunk in enumerate(chunks):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            chunk_result = _asr_pipe(chunk)
        
        transcribed_chunks.append(chunk_result["text"])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    full_text = " ".join(transcribed_chunks)
    chunks_processed = len(chunks)
```

**What it does:**
- **Short audio path**: Processes entire audio in one pass
- **Long audio path**: 
  1. Splits into overlapping chunks
  2. Processes each chunk separately
  3. Clears GPU cache before/after each chunk
  4. Joins transcriptions with spaces

**`torch.inference_mode()`**: Disables gradient calculation for faster inference.

##### Step 8: Metrics Collection
```python
sys_metrics = get_system_metrics()

result_dict = {
    "text": full_text,
    "latency_sec": latency,
    "cache_hit": False,
    "chunks_processed": chunks_processed,
    "cache_stats": _transcription_cache.stats(),
    "wer": 0.0,
    "cer": 0.0,
    "gpu_memory_used_mb": sys_metrics["gpu_memory_mb"],
}

# Update Prometheus metrics
requests_duration_seconds.labels(language=LANGUAGE).observe(latency)
requests_total.labels(language=LANGUAGE, status="success").inc()
chunks_processed_gauge.labels(language=LANGUAGE).set(chunks_processed)
gpu_memory_used_mb.labels(language=LANGUAGE).set(sys_metrics["gpu_memory_mb"])
cpu_usage_percent.labels(language=LANGUAGE).set(sys_metrics["cpu_percent"])
memory_usage_mb.labels(language=LANGUAGE).set(sys_metrics["memory_mb"])
```

**What it does:**
- Collects system metrics (CPU, RAM, GPU usage)
- Creates result dictionary with all metadata
- Updates Prometheus metrics for monitoring
- WER/CER set to 0.0 (requires ground truth to calculate)

##### Step 9: Caching and Return
```python
_transcription_cache.put(audio_bytes, result_dict)
return result_dict
```

**What it does:**
- Stores result in cache for future requests
- Returns complete result dictionary

#### 4. `get_cache_stats()`
```python
def get_cache_stats() -> dict:
    return _transcription_cache.stats()
```

**What it does:**
- Returns cache statistics (hits, misses, hit rate, size)
- Used by `/accuracy/stats` endpoint

---

## 📄 routers/transcribe.py

**Purpose:** API endpoint definitions and request handling.

### Global Variables

```python
router = APIRouter()
MAX_CONCURRENCY = 4
transcription_semaphore = None
```

**What they do:**
- `router`: FastAPI router for grouping endpoints
- `MAX_CONCURRENCY`: Maximum simultaneous transcription requests
- `transcription_semaphore`: asyncio.Semaphore for concurrency control

### Functions

#### 1. `init_semaphore()`
```python
def init_semaphore():
    global transcription_semaphore
    if transcription_semaphore is None:
        transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
```

**What it does:**
- Initializes semaphore with max concurrent requests
- Called during app startup
- Prevents creating multiple semaphores

**Why 4 concurrent requests?** Balances throughput with GPU memory constraints.

#### 2. `health_endpoint()`
```python
@router.get("/health")
async def health_endpoint():
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "language": LANGUAGE,
    })
```

**What it does:**
- Kubernetes liveness probe endpoint
- Returns 200 OK if service is running
- Includes timestamp and language identifier

**Kubernetes usage:** `livenessProbe: httpGet: path: /health`

#### 3. `ready_endpoint()`
```python
@router.get("/ready")
async def ready_endpoint():
    return JSONResponse({
        "status": "ready",
        "timestamp": time.time(),
        "language": LANGUAGE,
    })
```

**What it does:**
- Kubernetes readiness probe endpoint
- Indicates service is ready to accept traffic
- Could be enhanced to check model loading status

**Kubernetes usage:** `readinessProbe: httpGet: path: /ready`

#### 4. `metrics_endpoint()`
```python
@router.get("/metrics")
async def metrics_endpoint():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

**What it does:**
- Returns Prometheus metrics in text format
- Includes all counters, gauges, histograms
- Scraped by Prometheus server

**Example output:**
```
stt_requests_total{language="hausa",status="success"} 42
stt_latency_seconds_sum{language="hausa"} 105.3
stt_cache_hits_total{language="hausa"} 15
```

#### 5. `transcribe_endpoint(file)`

**Purpose:** Main transcription endpoint with concurrency control.

##### Step 1: Setup
```python
logger = setup_logger("hausa_stt")
request_id = str(uuid.uuid4())
init_semaphore()
```

**What it does:**
- Creates logger instance
- Generates request ID
- Ensures semaphore is initialized

##### Step 2: Concurrency Check
```python
if transcription_semaphore.locked():
    # Max concurrency reached
    rejected_requests_total.labels(language=LANGUAGE).inc()
    logger.warning("Request rejected due to concurrency limit")
    requests_total.labels(language=LANGUAGE, status="rejected").inc()
    raise HTTPException(status_code=503, detail="Service busy...")
```

**What it does:**
- Checks if semaphore is fully locked (all slots taken)
- If busy, returns 503 Service Unavailable
- Updates rejection metrics
- Logs warning

**Why reject?** Prevents GPU OOM by limiting concurrent inference.

##### Step 3: Request Processing
```python
async with transcription_semaphore:
    active_requests.labels(language=LANGUAGE).inc()
    start_time = time.time()
    
    audio_bytes = await file.read()
    result = transcribe_audio(audio_bytes, logger)
    
    latency = time.time() - start_time
    
    logger.info("Transcription request completed", extra={"request_id": request_id, "latency_seconds": latency})
    
    active_requests.labels(language=LANGUAGE).dec()
    return JSONResponse(result)
```

**What it does:**
1. **Acquires semaphore slot** - Blocks if all slots taken
2. **Increments active requests** - For monitoring
3. **Reads audio file** - Async to not block event loop
4. **Calls transcription** - Main processing
5. **Logs completion** - With latency
6. **Decrements active requests** - Releases slot
7. **Returns JSON response** - With all metadata

**Error handling:**
```python
except HTTPException:
    raise  # Re-raise HTTP exceptions
except Exception as e:
    requests_total.labels(language=LANGUAGE, status="error").inc()
    logger.error(f"Transcription error: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

**What it does:**
- Passes through HTTP exceptions (503, etc.)
- Catches all other exceptions
- Updates error metrics
- Logs error with details
- Returns 500 Internal Server Error

#### 6. `accuracy_stats_endpoint()`
```python
@router.get("/accuracy/stats")
async def accuracy_stats_endpoint():
    cache_stats = get_cache_stats()
    return JSONResponse({
        "cache_stats": cache_stats,
        "note": "WER/CER requires reference transcriptions (ground truth).",
    })
```

**What it does:**
- Returns cache statistics
- Explains WER/CER limitation
- Useful for monitoring cache effectiveness

---

## 🔄 Request Flow

```
1. Client sends POST /transcribe with audio file
   ↓
2. transcribe_endpoint() receives request
   ↓
3. Check semaphore - reject if busy (503)
   ↓
4. Acquire semaphore slot
   ↓
5. Read audio bytes
   ↓
6. transcribe_audio() called
   ↓
7. Check cache - return if hit
   ↓
8. Load model if not loaded
   ↓
9. Process audio (load, silence removal)
   ↓
10. Chunk if needed
   ↓
11. Run inference on each chunk
   ↓
12. Collect metrics
   ↓
13. Cache result
   ↓
14. Return JSON response
   ↓
15. Release semaphore slot
```

---

## 📊 Key Metrics

| Metric | Type | Purpose |
|--------|------|---------|
| `stt_requests_total` | Counter | Total requests by status |
| `stt_latency_seconds` | Histogram | Request latency distribution |
| `stt_active_requests` | Gauge | Current concurrent requests |
| `stt_cache_hits_total` | Counter | Cache effectiveness |
| `stt_chunks_processed` | Gauge | Audio chunking stats |
| `stt_gpu_memory_used_mb` | Gauge | GPU memory usage |
| `stt_rejected_requests_total` | Counter | Concurrency limit hits |

---

## 🔧 Configuration

### Adjustable Parameters

**In `services/asr.py`:**
- `max_size=10` - Cache size (line 287)
- `threshold_db=-40` - Silence threshold (line 354)
- `torch.cuda.set_per_process_memory_fraction(0.95)` - GPU memory limit (line 469)

**In `routers/transcribe.py`:**
- `MAX_CONCURRENCY = 4` - Concurrent requests (line 151)

**In `main.py`:**
- `port=8000` - Service port (line 90)
- `host="0.0.0.0"` - Listen address (line 90)

---

## 🐛 Common Issues

### 1. GPU Out of Memory
**Solution:** Reduce `MAX_CONCURRENCY` or chunk size

### 2. Slow First Request
**Cause:** Model loading on first request  
**Solution:** Preload in startup (already implemented)

### 3. Cache Not Working
**Cause:** Different audio encoding  
**Solution:** Cache uses raw bytes hash, works across formats

### 4. High Latency
**Check:**
- GPU availability (`nvidia-smi`)
- Chunk size (larger = faster but more memory)
- Silence removal effectiveness
