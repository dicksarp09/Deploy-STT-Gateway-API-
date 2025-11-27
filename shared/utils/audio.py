"""
Audio processing utilities for STT services.
Includes audio loading, silence detection, VAD, chunking, and hallucination detection.
"""
import os
import tempfile
import numpy as np
import torch
import ffmpeg
from typing import List


def load_audio(file_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """
    Load audio from bytes using ffmpeg.
    
    Args:
        file_bytes: Audio file bytes
        sr: Target sample rate (default: 16000)
    
    Returns:
        Audio array as float32 numpy array
    """
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_file = fp.name
            fp.write(file_bytes)
            fp.flush()
            
            # Optimized ffmpeg flags for speed
            out, _ = (
                ffmpeg.input(temp_file)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin", "-loglevel", "quiet"], 
                     capture_stdout=True, capture_stderr=True)
            )
        
        audio_array = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        return audio_array
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def detect_and_remove_silence(
    audio_array: np.ndarray, 
    threshold_db: float = -40, 
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Detect and remove silent segments from audio.
    
    Args:
        audio_array: Audio samples
        threshold_db: Silence threshold in dB
        sample_rate: Sample rate of audio
    
    Returns:
        Audio array with silent segments removed
    """
    # Calculate RMS energy
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    energy = []
    for i in range(0, len(audio_array) - frame_length, hop_length):
        frame = audio_array[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        energy.append(rms)
    
    # Convert to dB
    energy = np.array(energy)
    energy_db = 20 * np.log10(energy + 1e-10)
    
    # Find threshold
    max_energy_db = np.max(energy_db)
    threshold_abs = max_energy_db + threshold_db
    
    # Mark non-silent frames
    non_silent = energy_db > threshold_abs
    
    # Convert back to samples
    non_silent_samples = []
    for i, is_non_silent in enumerate(non_silent):
        start_sample = i * hop_length
        end_sample = start_sample + frame_length
        if is_non_silent:
            non_silent_samples.extend(range(start_sample, min(end_sample, len(audio_array))))
    
    if not non_silent_samples:
        return audio_array  # No silence detected, return original
    
    # Extract non-silent samples
    non_silent_samples = sorted(set(non_silent_samples))
    return audio_array[non_silent_samples]


def get_vad_model():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    return model, utils


def process_audio_with_vad(
    audio_array: np.ndarray, 
    sample_rate: int = 16000
) -> List[np.ndarray]:
    """
    Process audio using Silero VAD to remove silence and create chunks.
    
    Args:
        audio_array: Audio samples
        sample_rate: Sample rate (default: 16000)
    
    Returns:
        List of active audio chunks
    """
    try:
        model, utils = get_vad_model()
        (get_speech_timestamps, save_audio, read_audio, 
         VADIterator, collect_chunks) = utils
        
        # Convert to tensor
        wav = torch.from_numpy(audio_array)
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
        
        if not speech_timestamps:
            return [audio_array]  # Return original if no speech detected
        
        # Merge timestamps into chunks of ~10-18 seconds
        merged_chunks = []
        current_chunk = []
        current_duration = 0
        
        for ts in speech_timestamps:
            start = ts['start']
            end = ts['end']
            duration = (end - start) / sample_rate
            
            segment = audio_array[start:end]
            
            if current_duration + duration > 18.0:
                # Flush current chunk
                if current_chunk:
                    merged_chunks.append(np.concatenate(current_chunk))
                    current_chunk = []
                    current_duration = 0
            
            current_chunk.append(segment)
            current_duration += duration
        
        if current_chunk:
            merged_chunks.append(np.concatenate(current_chunk))
        
        return merged_chunks
    except Exception:
        return [audio_array]  # Fallback to original on error


def split_audio_into_chunks(
    audio_array: np.ndarray,
    chunk_length_seconds: int = 30,
    overlap_seconds: int = 2,
    sample_rate: int = 16000
) -> List[np.ndarray]:
    """
    Split audio array into overlapping chunks to avoid cutting words.
    
    Args:
        audio_array: numpy array of audio samples
        chunk_length_seconds: length of each chunk in seconds
        overlap_seconds: overlap between chunks in seconds
        sample_rate: audio sample rate
    
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_length_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    chunks = []
    for i in range(0, len(audio_array), step_samples):
        chunk = audio_array[i:i + chunk_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    
    return chunks


def get_optimal_chunk_size() -> int:
    """
    Dynamically determine chunk size based on available GPU memory.
    
    Returns:
        Optimal chunk length in seconds
    """
    if not torch.cuda.is_available():
        return 30  # Default for CPU
    
    try:
        # Get available GPU memory in GB
        available_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        
        # Adjust chunk size based on memory
        if available_memory < 5:
            return 20  # Smaller chunks for low memory
        elif available_memory < 10:
            return 25
        else:
            return 30  # Standard size
    except Exception:
        return 30  # Fallback to default


def detect_hallucination(text: str) -> bool:
    """
    Detect potential hallucinations in text.
    
    Criteria:
    1. Repeated tokens > 5 times
    2. Large blocks of identical n-grams
    
    Args:
        text: Transcribed text to check
    
    Returns:
        True if hallucination detected, False otherwise
    """
    if not text:
        return False
    
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check for immediate repetition of words
    repeats = 0
    max_repeats = 0
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            repeats += 1
        else:
            max_repeats = max(max_repeats, repeats)
            repeats = 0
    max_repeats = max(max_repeats, repeats)
    
    if max_repeats > 5:
        return True
    
    # Check for n-gram repetition (4-gram check)
    if len(words) > 20:
        ngrams = set()
        for i in range(len(words) - 3):
            ngram = tuple(words[i:i+4])
            if ngram in ngrams:
                # Count occurrences
                count = 0
                for j in range(len(words) - 3):
                    if tuple(words[j:j+4]) == ngram:
                        count += 1
                if count > 3:
                    return True
            ngrams.add(ngram)
    
    return False


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions, N = reference word count
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        WER as float between 0.0 and 1.0
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Simple edit distance approach
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return float(min(wer, 1.0))


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).
    CER = (S + D + I) / N (at character level)
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        CER as float between 0.0 and 1.0
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    
    # Simple edit distance approach
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    
    cer = d[len(ref_chars), len(hyp_chars)] / len(ref_chars)
    return float(min(cer, 1.0))
