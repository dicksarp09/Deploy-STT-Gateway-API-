# STT Services Port Configuration

## Port Assignments

| Service | Port | URL |
|---------|------|-----|
| **Hausa STT** | 8001 | http://localhost:8001 |
| **Yoruba STT** | 8002 | http://localhost:8002 |
| **Multilingual STT** | 8003 | http://localhost:8003 |

## Running All Services Simultaneously

### Terminal 1 - Hausa STT
```bash
cd hausa_stt
python main.py
# Service running on http://0.0.0.0:8001
```

### Terminal 2 - Yoruba STT
```bash
cd yoruba_stt
python main.py
# Service running on http://0.0.0.0:8002
```

### Terminal 3 - Multilingual STT
```bash
cd multilingual_stt
python main.py
# Service running on http://0.0.0.0:8003
```

## Testing Each Service

### Hausa STT (Port 8001)
```bash
# Health check
curl http://localhost:8001/health

# Transcribe
curl -X POST http://localhost:8001/transcribe -F "file=@audio.wav"

# Metrics
curl http://localhost:8001/metrics
```

### Yoruba STT (Port 8002)
```bash
# Health check
curl http://localhost:8002/health

# Transcribe
curl -X POST http://localhost:8002/transcribe -F "file=@audio.wav"

# Metrics
curl http://localhost:8002/metrics
```

### Multilingual STT (Port 8003)
```bash
# Health check
curl http://localhost:8003/health

# Transcribe
curl -X POST http://localhost:8003/transcribe \
  -F "file=@audio.wav" \
  -F "source_lang=fuv"

# Metrics
curl http://localhost:8003/metrics
```

## Prometheus Configuration

Update your `prometheus.yml` to scrape all three services:

```yaml
scrape_configs:
  - job_name: 'hausa-stt'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'

  - job_name: 'yoruba-stt'
    static_configs:
      - targets: ['localhost:8002']
    metrics_path: '/metrics'

  - job_name: 'multilingual-stt'
    static_configs:
      - targets: ['localhost:8003']
    metrics_path: '/metrics'
```

## Docker Compose

Updated port mappings for docker-compose.yml:

```yaml
services:
  hausa-stt:
    ports:
      - "8001:8001"  # Changed from 8000:8000
    
  yoruba-stt:
    ports:
      - "8002:8002"  # Changed from 8001:8000
    
  multilingual-stt:
    ports:
      - "8003:8003"  # Changed from 8002:8000
```

## Nginx Reverse Proxy

If using Nginx as reverse proxy:

```nginx
# Hausa STT
location /hausa/ {
    proxy_pass http://localhost:8001/;
}

# Yoruba STT
location /yoruba/ {
    proxy_pass http://localhost:8002/;
}

# Multilingual STT
location /multilingual/ {
    proxy_pass http://localhost:8003/;
}
```

## Firewall Rules (GCP)

```bash
# Allow all three ports
gcloud compute firewall-rules create allow-stt-services \
    --allow=tcp:8001,tcp:8002,tcp:8003 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow STT API traffic"
```

## Environment Variables

If you need to override ports via environment variables, update main.py:

```python
import os

port = int(os.getenv("PORT", 8001))  # Default to 8001 for Hausa
uvicorn.run("main:app", host="0.0.0.0", port=port)
```

Then run with:
```bash
PORT=9001 python main.py  # Custom port
```

## API Documentation

Each service has its own Swagger UI:

- **Hausa:** http://localhost:8001/docs
- **Yoruba:** http://localhost:8002/docs
- **Multilingual:** http://localhost:8003/docs

## Summary

✅ **Hausa STT** → Port **8001**  
✅ **Yoruba STT** → Port **8002**  
✅ **Multilingual STT** → Port **8003**  

All services can now run simultaneously without port conflicts!
