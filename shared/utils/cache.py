"""
LRU cache implementation for transcription results.
Provides efficient caching with SHA256-based keys.
"""
import hashlib
from collections import OrderedDict
from typing import Optional


class TranscriptionCache:
    """Simple LRU cache for transcription results."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_key(self, audio_bytes: bytes, lang: str = "") -> str:
        """
        Generate cache key from audio hash and optional language.
        
        Args:
            audio_bytes: Audio file bytes
            lang: Language code (optional, for multilingual services)
        
        Returns:
            SHA256 hash as hex string
        """
        if lang:
            return hashlib.sha256(audio_bytes + lang.encode()).hexdigest()
        return hashlib.sha256(audio_bytes).hexdigest()
    
    def get(self, audio_bytes: bytes, lang: str = "") -> Optional[dict]:
        """
        Get result from cache if exists.
        
        Args:
            audio_bytes: Audio file bytes
            lang: Language code (optional)
        
        Returns:
            Cached result dict or None if not found
        """
        key = self.get_key(audio_bytes, lang)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, audio_bytes: bytes, result: dict, lang: str = "") -> None:
        """
        Store result in cache.
        
        Args:
            audio_bytes: Audio file bytes
            result: Transcription result to cache
            lang: Language code (optional)
        """
        key = self.get_key(audio_bytes, lang)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[key] = result
    
    def stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, total, hit_rate, and size
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }
