# Yoruba STT Service - Code Documentation

Complete code documentation for the Yoruba Speech-to-Text service.

## 📁 Project Structure

```
yoruba_stt/
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

## 🔄 Architecture Overview

The Yoruba STT service has **identical architecture** to the Hausa STT service, with only the model changed:

| Component | Hausa | Yoruba |
|-----------|-------|--------|
| **Model** | `CLEAR-Global/w2v-bert-2.0-hausa_100_400h_yourtts` | `CLEAR-Global/w2v-bert-2.0-yoruba_naijavoices_1h` |
| **Language** | `"hausa"` | `"yoruba"` |
| **Architecture** | Identical | Identical |

---

## 📄 Key Differences from Hausa

### services/asr.py
```python
# Lines 23-24
LANGUAGE = "yoruba"  # vs "hausa"
MODEL_ID = "CLEAR-Global/w2v-bert-2.0-yoruba_naijavoices_1h"
```

### main.py
```python
# Line 28
app = FastAPI(title="Yoruba STT API")  # vs "Hausa STT API"

# Line 80
logger = setup_logger("yoruba_stt")  # vs "hausa_stt"
```

### routers/transcribe.py
```python
# Line 95
logger = setup_logger("yoruba_stt")  # vs "hausa_stt"
```

---

## 📚 Complete Documentation

Since the code is **100% identical** to Hausa STT (except model/language), see the comprehensive documentation here:

**[Hausa STT CODE_DOCUMENTATION.md](../hausa_stt/CODE_DOCUMENTATION.md)**

This document covers:
- ✅ All function explanations
- ✅ Request flow diagrams
- ✅ Code walkthroughs
- ✅ Metrics descriptions
- ✅ Configuration options
- ✅ Troubleshooting guide

---

## 🎯 Yoruba Model Specifics

**Model:** `CLEAR-Global/w2v-bert-2.0-yoruba_naijavoices_1h`

**Training Data:**
- NaijaVoices dataset
- 1 hour of Yoruba speech
- Nigerian Yoruba dialect

**Strengths:**
- Excellent for Nigerian Yoruba
- Handles tonal variations
- Good with code-switching

**Usage:**
```bash
curl -X POST "http://localhost:8000/transcribe" -F "file=@yoruba_audio.wav"
```

---

## 📊 Metrics

All metrics use `language="yoruba"` label:
```
stt_requests_total{language="yoruba"}
stt_latency_seconds{language="yoruba"}
stt_cache_hits_total{language="yoruba"}
```

---

## 🔧 Configuration

Same as Hausa:
- `MAX_CONCURRENCY = 4`
- `cache max_size = 10`
- `threshold_db = -40`

---

## 💡 Summary

Yoruba STT is a **direct clone** of Hausa STT with only the model changed. For detailed code explanations, refer to [Hausa CODE_DOCUMENTATION.md](../hausa_stt/CODE_DOCUMENTATION.md).
