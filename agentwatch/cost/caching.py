from typing import Optional
from dataclasses import dataclass
import hashlib

@dataclass
class CacheHit:
    prompt_hash: str
    response_text: str
    framework: str

class SemanticCacheManager:
    """
    Manages semantic caching of agent prompts to save API costs.
    """
    def __init__(self):
        # In-memory dictionary for the simplest pass of Behavior 1
        self._cache = {}

    async def store(self, prompt: str, response_text: str, framework: str) -> None:
        """
        Stores the response for a given prompt.
        """
        prompt_hash = self._hash_prompt(prompt)
        self._cache[prompt_hash] = CacheHit(
            prompt_hash=prompt_hash,
            response_text=response_text,
            framework=framework
        )

    async def search(self, prompt: str) -> Optional[CacheHit]:
        """
        Searches the cache for an exact match or semantically similar prompt.
        """
        prompt_hash = self._hash_prompt(prompt)
        return self._cache.get(prompt_hash)

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
