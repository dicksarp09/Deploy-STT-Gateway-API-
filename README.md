# STT Gateway - Multilingual Speech-to-Text Service

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/stt-gateway)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-ready multilingual Speech-to-Text gateway that intelligently routes requests to language-specific backend services. Built with FastAPI and deployed on Modal, this gateway provides unified API access to Hausa, Yoruba, and Igbo transcription services.

## Features

- **🌍 Multilingual Support**: Hausa, Yoruba, and Igbo languages with easy extensibility
- **🔄 Intelligent Routing**: Automatic request routing to language-specific backend services
- **🛡️ Production-Ready Resilience**: Circuit breaker pattern, retry logic with exponential backoff, and rate limiting
- **📊 Comprehensive Observability**: Real-time metrics, latency percentiles, and structured logging
- **⚡ High Performance**: Keep-warm instances for fast response times
- **🔒 Robust Validation**: File size limits, format checking, and parameter validation

## Quick Start

### Prerequisites

- Python 3.11+
- Modal account ([sign up here](https://modal.com))
- Modal CLI installed: `pip install modal`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stt-gateway.git
cd stt-gateway
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Authenticate with Modal:
```bash
modal token new
```

### Deployment

Deploy the gateway with a single command:
```bash
modal deploy gateway_stt_modal.py
```

After deployment, Modal will provide a public HTTPS endpoint URL.

## API Reference

### Base URL
```
https://your-gateway-url
```

### Endpoints

#### POST `/transcribe`
Transcribe audio files to text.

**Request:**
```bash
curl -X POST "https://your-gateway-url/transcribe" \
  -F "file=@audio.wav" \
  -F "language=hausa"
```

**Parameters:**
- `file` (required): Audio file (WAV, MP3, M4A, OGG, FLAC, WebM)
- `language` (required): Target language (`hausa`, `yoruba`, or `igbo`)

**Response:**
```json
{
  "request_id": "req_1234567890",
  "language": "hausa",
  "routed_to": "https://voicebreeze--hausa-stt-endpoint-api.modal.run/transcribe",
  "transcription": "Sannu, yaya kuke?",
  "backend_response": {...},
  "latency_seconds": 2.345,
  "timestamp": "2025-11-26T14:00:00"
}
```

#### GET `/health`
Check gateway health and capabilities.

**Request:**
```bash
curl "https://your-gateway-url/health"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "supported_languages": ["hausa", "yoruba", "igbo"],
  "timestamp": "2025-11-26T14:00:00"
}
```

#### GET `/languages`
Get supported languages and their backend endpoints.

**Request:**
```bash
curl "https://your-gateway-url/languages"
```

**Response:**
```json
{
  "hausa": "https://voicebreeze--hausa-stt-endpoint-api.modal.run/transcribe",
  "yoruba": "https://voicebreeze--yoruba-stt-endpoint-api.modal.run/transcribe",
  "igbo": "https://voicebreeze--multilingual-speech-translation-endpoint-api.modal.run/transcribe"
}
```

#### GET `/metrics`
Retrieve comprehensive gateway metrics and observability data.

**Request:**
```bash
curl "https://your-gateway-url/metrics"
```

**Response:**
```json
{
  "total_requests": 1523,
  "total_errors": 12,
  "error_rate": 0.0079,
  "uptime_seconds": 86400,
  "latency_p50": 2.1,
  "latency_p90": 4.5,
  "latency_p95": 6.2,
  "latency_p99": 12.8,
  "language_traffic": {
    "hausa": 645,
    "yoruba": 523,
    "igbo": 355
  },
  "backend_errors": {
    "hausa": 3,
    "yoruba": 5,
    "igbo": 4
  },
  "requests_per_minute": 1.06,
  "circuit_breaker": {
    "open_circuits": [],
    "failures": {}
  }
}
```

## Architecture

### System Overview

The gateway implements a microservices architecture pattern where a central gateway routes requests to specialized backend services based on the requested language.

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  STT Gateway    │
│  (This Service) │
└────┬───┬───┬────┘
     │   │   │
     ▼   ▼   ▼
┌─────┐ ┌─────┐ ┌─────┐
│Hausa│ │Yoruba│ │Igbo │
│ STT │ │ STT  │ │ STT │
└─────┘ └─────┘ └─────┘
```

### Backend Services

| Language | Backend Service | Endpoint URL |
|----------|----------------|--------------|
| Hausa | hausa-stt-endpoint-api | https://voicebreeze--hausa-stt-endpoint-api.modal.run/transcribe |
| Yoruba | yoruba-stt-endpoint-api | https://voicebreeze--yoruba-stt-endpoint-api.modal.run/transcribe |
| Igbo | multilingual-speech-translation-endpoint-api | https://voicebreeze--multilingual-speech-translation-endpoint-api.modal.run/transcribe |

## Reliability & Resilience

### Circuit Breaker Pattern
Prevents cascading failures by opening the circuit after consecutive failures.

- **Failure Threshold**: 5 consecutive failures
- **Timeout**: 60 seconds
- **Recovery**: Automatic after timeout period

### Retry Logic with Exponential Backoff
Automatically retries failed requests with increasing delays.

- **Max Retries**: 3 attempts
- **Initial Delay**: 1.0 second
- **Backoff Strategy**: Exponential (2^attempt)
- **Max Delay**: 4.0 seconds

**Example retry sequence:**
```
Request → Fail → Wait 1s → Retry → Fail → Wait 2s → Retry → Fail → Wait 4s → Final result
```

### Rate Limiting
Protects backend services from overload.

- **Max Requests**: 10 requests per client
- **Time Window**: 60 seconds (1 minute)
- **Scope**: Per client IP address
- **Response Code**: 429 Too Many Requests

### Timeout Handling
- **Request Timeout**: 300 seconds (5 minutes)
- Suitable for processing audio files up to 10 minutes

## Validation Rules

| Rule | Requirement | Error Response |
|------|-------------|----------------|
| File Size | 500 MB maximum | 400 Bad Request with file size details |
| Audio Duration | 10 minutes maximum | 400 Bad Request if duration likely exceeds limit |
| Audio Format | WAV, MP3, M4A, OGG, FLAC, WebM | 400 Bad Request for unsupported formats |
| Language Parameter | Must be 'hausa', 'yoruba', or 'igbo' | 400 Bad Request with list of supported languages |
| Missing Language | Language parameter is required | 400 Bad Request indicating missing parameter |

## Observability & Monitoring

### Structured Logging
All routing decisions, errors, and important events are logged with structured data:
- Request details (language, file size, client IP)
- Routing decisions and target backend service
- Retry attempts with delays
- Circuit breaker state changes
- Backend response status
- Final metrics (latency, status, request ID)

### Latency Percentiles

| Metric | Description | Use Case |
|--------|-------------|----------|
| p50 (Median) | 50% of requests complete within this time | Typical user experience |
| p90 | 90% of requests complete within this time | Most users' experience |
| p95 | 95% of requests complete within this time | Performance SLA monitoring |
| p99 | 99% of requests complete within this time | Outlier detection |

### Alert Triggers

| Alert Trigger | Log Message | Recommended Action |
|---------------|-------------|-------------------|
| Circuit Breaker Opens | 🚨 ALERT: Circuit breaker opened for {service} | Backend service is unavailable. Investigate service health. |
| All Retries Failed | 🚨 ALERT: All {N} attempts failed for {service} | Backend consistently failing. Check service logs. |
| Service Unavailable | 🚨 ALERT: Service {service} is unavailable | Wait for circuit timeout or fix backend. |
| Gateway Error | 🚨 ALERT: Unexpected error in request {request_id} | Review stack trace and fix bug. |

## Configuration

### Modal Deployment Configuration

```python
Parameter           Value                Purpose
-----------------   ------------------   --------------------------------
Modal App Name      stt-gateway          Unique identifier
Python Version      3.11                 Latest stable Python
Base Image          debian_slim          Minimal image for fast cold starts
Keep Warm           1 instance           Maintains warm container
Container Timeout   300 seconds          Idle timeout
Request Timeout     600 seconds          Maximum processing time
```

### Environment Variables

Create a `.env` file (optional):
```bash
# CORS settings
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Logging level
LOG_LEVEL=INFO
```

## Adding New Languages

Adding support for a new language is simple. Update the `LANGUAGE_ENDPOINTS` dictionary in `gateway_stt_modal.py`:

```python
LANGUAGE_ENDPOINTS = {
    "hausa": "https://voicebreeze--hausa-stt-endpoint-api.modal.run/transcribe",
    "yoruba": "https://voicebreeze--yoruba-stt-endpoint-api.modal.run/transcribe",
    "igbo": "https://voicebreeze--multilingual-speech-translation-endpoint-api.modal.run/transcribe",
    "swahili": "https://your-new-backend-endpoint.modal.run/transcribe"  # Add new language
}
```

No other code changes are required!

## Development

### Local Testing

Run the gateway locally for development:
```bash
modal serve gateway_stt_modal.py
```

This starts a local development server with hot reload.

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black gateway_stt_modal.py

# Type checking
mypy gateway_stt_modal.py

# Linting
flake8 gateway_stt_modal.py
```

## Troubleshooting

### Common Issues

**Issue: Circuit breaker keeps opening**
- Check backend service health at the direct endpoint URL
- Review backend service logs in Modal dashboard
- Verify network connectivity between services

**Issue: High latency (p99 > 30s)**
- Check audio file sizes (larger files take longer)
- Monitor backend service performance
- Consider scaling backend services

**Issue: Rate limit errors (429)**
- Increase rate limit in configuration if legitimate traffic
- Implement exponential backoff in client
- Distribute load across multiple client IPs if possible

**Issue: "Language not supported" errors**
- Verify language parameter is lowercase
- Check supported languages via `/languages` endpoint
- Ensure language is in LANGUAGE_ENDPOINTS dictionary

## Performance Tips

1. **File Size**: Keep audio files under 100 MB for best performance
2. **Audio Format**: WAV and FLAC provide best transcription quality
3. **Batch Processing**: Use multiple concurrent requests for batch transcription
4. **Caching**: Implement client-side caching for repeated transcriptions


Made with ❤️ by [Your Team Name]
