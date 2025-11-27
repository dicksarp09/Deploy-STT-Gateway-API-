# Multilingual STT Service - Code Documentation

Complete code documentation for the Multilingual Speech-to-Text service using SeamlessM4Tv2.

## 📁 Project Structure

```
multilingual_stt/
├── main.py                    # FastAPI application entry point
├── services/
│   ├── __init__.py           # Package marker
│   └── asr.py                # SeamlessM4T model and transcription logic
├── routers/
│   ├── __init__.py           # Package marker
│   └── transcribe.py         # API endpoint definitions
└── requirements.txt          # Python dependencies (includes jiwer, torchaudio)
```

---

## 🎯 Key Differences from Hausa/Yoruba

| Feature | Hausa/Yoruba | Multilingual |
|---------|--------------|--------------|
| **Model** | w2v-bert (ASR only) | SeamlessM4Tv2 (ASR + Translation) |
| **Output** | Single language | Native + English |
| **VAD** | Basic silence removal | Silero VAD with chunking |
| **Hallucination** | None | Detection + retry logic |
| **WER/CER** | Placeholder (0.0) | Calculated with jiwer |
| **Language Param** | Fixed | User-specified |
| **Complexity** | Simple | Advanced |

---

## 📄 main.py

**Purpose:** FastAPI application initialization (similar to Hausa/Yoruba).

**Key Differences:**
```python
# Line 28
app = FastAPI(
    title="Multilingual STT API",
    description="Multilingual Speech-to-Text API using SeamlessM4Tv2 model"
)

# Line 56: Supported languages listed
return {
    "supported_languages": [
        "fuv (Fulfulde)", "yor (Yoruba)", "ibo (Igbo)",
        "hau (Hausa)", "and many more..."
    ]
}
```

See [Hausa CODE_DOCUMENTATION.md](../hausa_stt/CODE_DOCUMENTATION.md) for general FastAPI setup details.

---

## 📄 services/asr.py

**Purpose:** SeamlessM4T model loading and dual transcription (native + English).

### Global Variables

```python
LANGUAGE = "multilingual"
MODEL_ID = "facebook/seamless-m4t-v2-large"

class ModelState:
    model = None
    processor = None

_transcription_cache = TranscriptionCache(max_size=10)
```

**What they do:**
- `MODEL_ID`: Facebook's SeamlessM4T v2 Large model
- `ModelState`: Class-based state (vs global variable) for model/processor
- Cache includes language in key: `get_key(audio_bytes, lang)`

### Key Functions

#### 1. `load_model(logger)`

**Differences from Hausa:**
```python
# Load both processor and model
ModelState.processor = SeamlessM4TProcessor.from_pretrained(MODEL_ID)
ModelState.model = SeamlessM4Tv2Model.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)
```

**What it does:**
- Loads **processor** (tokenizer + feature extractor)
- Loads **model** (encoder-decoder architecture)
- Uses float16 on GPU (model is ~10GB)
- No fallback to float32 (SeamlessM4T requires float16)

#### 2. `run_inference(vad_chunks, tgt_lang, source_lang, logger)`

**Purpose:** Run inference on VAD chunks for a specific target language.

**Parameters:**
- `vad_chunks`: List of audio chunks from Silero VAD
- `tgt_lang`: Target language code ("eng", "fuv", "yor", etc.)
- `source_lang`: Source language (for metrics labeling)
- `logger`: Logger instance

**Step-by-Step:**

##### Step 1: Process Each Chunk
```python
for i, chunk in enumerate(vad_chunks):
    if len(chunk) == 0:
        continue
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(chunk).float()
    
    # Prepare inputs
    inputs = ModelState.processor(
        audios=audio_tensor,
        sampling_rate=16000,
        return_tensors="pt"
    )
```

**What it does:**
- Skips empty chunks
- Converts numpy array to PyTorch tensor
- Processes audio through feature extractor
- Returns input tensors for model

##### Step 2: Language-Specific Parameters
```python
if source_lang == "yor":
    num_beams = 8
    repetition_penalty = 1.5
    no_repeat_ngram = 5
elif source_lang == "ibo":
    num_beams = 4
    repetition_penalty = 1.2
    no_repeat_ngram = 4
else:
    num_beams = 4
    repetition_penalty = 1.2
    no_repeat_ngram = 4
```

**What it does:**
- **Yoruba**: Stronger penalties (prone to hallucinations)
- **Igbo**: Default parameters (works well)
- **Others**: Default parameters

**Why different?** Each language has different hallucination tendencies based on training data.

##### Step 3: Generate with Anti-Hallucination Params
```python
with torch.inference_mode():
    generated_output = ModelState.model.generate(
        **inputs,
        tgt_lang=tgt_lang,           # Target language
        generate_speech=False,       # Text only, no audio
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram,
        num_beams=num_beams,
        return_dict_in_generate=True
    )

output_tokens = generated_output.sequences
text = ModelState.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
```

**What it does:**
- Runs beam search decoding
- `tgt_lang`: Specifies output language
- `generate_speech=False`: Critical - ensures text output only
- `repetition_penalty`: Penalizes repeated tokens
- `no_repeat_ngram_size`: Blocks repeated n-grams
- `num_beams`: Beam search width (higher = better quality, slower)
- Decodes tokens to text

##### Step 4: Hallucination Detection and Retry
```python
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
                repetition_penalty=2.0,  # Much stronger
                no_repeat_ngram_size=3,
                num_beams=2,
                return_dict_in_generate=True
            )
            output_tokens = generated_output.sequences
            text = ModelState.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Retry failed: {e}")
```

**What it does:**
- Checks for hallucination patterns (see `detect_hallucination()` in `shared/utils/audio.py`)
- If detected, updates metrics
- Retries with **stronger penalties**:
  - `repetition_penalty=2.0` (vs 1.2-1.5)
  - `num_beams=2` (faster retry)
- If retry fails, uses original output

**Hallucination patterns:**
- Repeated words >5 times
- Repeated 4-grams >3 times

##### Step 5: Combine Chunks
```python
transcribed_chunks.append(text)

# After all chunks
return " ".join(transcribed_chunks), len(vad_chunks)
```

**What it does:**
- Collects all chunk transcriptions
- Joins with spaces
- Returns full text and chunk count

#### 3. `transcribe_audio(audio_bytes, source_lang, reference_text, logger)`

**Purpose:** Main transcription function with dual output.

**Unique Features:**

##### Cache with Language
```python
cached_result = _transcription_cache.get(audio_bytes, source_lang)
```
- Cache key includes language
- Same audio, different language = different cache entry

##### VAD Processing
```python
raw_audio_array = load_audio(audio_bytes)
vad_chunks = process_audio_with_vad(raw_audio_array)
```
- Uses Silero VAD instead of simple silence removal
- Creates 10-18 second chunks based on speech activity
- More intelligent than fixed-size chunking

##### Dual Transcription
```python
# 1. Native Transcription (ASR)
native_text, chunks_processed = run_inference(vad_chunks, source_lang, source_lang, request_logger)

# 2. English Translation (S2TT)
english_text, _ = run_inference(vad_chunks, "eng", source_lang, request_logger)
```

**What it does:**
- First pass: Transcribes to source language
- Second pass: Translates to English
- Both use same VAD chunks
- Returns both in result

##### WER/CER Calculation
```python
if reference_text:
    try:
        import jiwer
        wer = jiwer.wer(reference_text, native_text)
        cer = jiwer.cer(reference_text, native_text)
    except Exception as e:
        request_logger.error(f"Error calculating WER/CER: {e}")
```

**What it does:**
- If ground truth provided, calculates accuracy metrics
- Uses `jiwer` library (industry standard)
- WER: Word Error Rate
- CER: Character Error Rate

##### Result Dictionary
```python
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
```

**Differences from Hausa/Yoruba:**
- Two text fields: `native_text` and `english_text`
- Separate timing: `preprocessing_sec` and `inference_sec`
- Real WER/CER values (if reference provided)

---

## 📄 routers/transcribe.py

**Purpose:** API endpoints with language parameter support.

### Key Differences

#### `transcribe_endpoint()` Parameters
```python
@router.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    source_lang: str = Form(default="fuv"),
    reference_text: Optional[str] = Form(default=None)
):
```

**What's different:**
- `source_lang`: User specifies source language (default: "fuv" for Fulfulde)
- `reference_text`: Optional ground truth for WER/CER calculation
- Uses `Form()` instead of just `File()` for multipart form data

#### Request Processing
```python
audio_bytes = await file.read()
result = transcribe_audio(audio_bytes, source_lang, reference_text, logger)
```

**What it does:**
- Passes language and reference text to transcription function
- Returns dual transcription result

---

## 🔄 Request Flow

```
1. Client sends POST /transcribe
   - file: audio.wav
   - source_lang: "fuv"
   - reference_text: "optional ground truth"
   ↓
2. Check semaphore (reject if busy)
   ↓
3. Check cache (with language key)
   ↓
4. Load model if not loaded
   ↓
5. Load audio with ffmpeg
   ↓
6. Apply Silero VAD → chunks
   ↓
7. For each chunk:
   - Prepare inputs
   - Generate (native language)
   - Check hallucination
   - Retry if needed
   ↓
8. Combine native chunks
   ↓
9. Repeat for English translation
   ↓
10. Calculate WER/CER if reference provided
   ↓
11. Update metrics
   ↓
12. Cache result
   ↓
13. Return JSON with native + English
```

---

## 📊 Additional Metrics

Multilingual service has extra metrics:

```python
preprocessing_duration_seconds  # Audio preprocessing time
inference_duration_seconds      # Model inference time
hallucination_events_total      # Detected hallucinations
transcription_errors_total      # Inference errors
average_audio_duration_seconds  # Audio length distribution
```

---

## 🔧 Configuration

| Parameter | Location | Default | Purpose |
|-----------|----------|---------|---------|
| `source_lang` | Request param | "fuv" | Source language |
| `num_beams` | Dynamic | 4-8 | Beam search width |
| `repetition_penalty` | Dynamic | 1.2-2.0 | Hallucination control |
| `no_repeat_ngram_size` | Dynamic | 3-5 | N-gram blocking |

---

## 🎯 Supported Languages

SeamlessM4T supports 100+ languages. Common ones:
- `fuv` - Fulfulde
- `yor` - Yoruba
- `ibo` - Igbo
- `hau` - Hausa
- `eng` - English
- `fra` - French
- `spa` - Spanish

Full list: https://huggingface.co/facebook/seamless-m4t-v2-large

---

## 🚀 Usage Example

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "source_lang=fuv" \
  -F "reference_text=Optional ground truth text"
```

**Response:**
```json
{
  "native_text": "Transcription in Fulfulde...",
  "english_text": "Translation to English...",
  "latency_sec": 4.2,
  "preprocessing_sec": 0.8,
  "inference_sec": 3.4,
  "cache_hit": false,
  "chunks_processed": 3,
  "wer": 0.12,
  "cer": 0.06,
  "gpu_memory_used_mb": 9216.0
}
```

---

## 🐛 Troubleshooting

### Hallucinations
**Symptom:** Repeated words/phrases  
**Solution:** Already handled with detection + retry

### Empty Transcription
**Cause:** `generate_speech=True` (wrong)  
**Solution:** Ensure `generate_speech=False`

### Language Not Supported
**Check:** https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages

---

## 💡 Summary

Multilingual STT is **significantly more complex** than Hausa/Yoruba:
- ✅ Dual output (native + English)
- ✅ Advanced VAD with Silero
- ✅ Hallucination detection + retry
- ✅ Real WER/CER calculation
- ✅ Language-specific optimization
- ✅ Larger model (10GB vs 1.2GB)

For basic concepts (FastAPI setup, caching, GPU management), see [Hausa CODE_DOCUMENTATION.md](../hausa_stt/CODE_DOCUMENTATION.md).
