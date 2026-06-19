import logging
from functools import wraps

logger = logging.getLogger(__name__)

_original_async_create = None

def patch_openai(semantic_cache):
    """
    Patches the OpenAI AsyncClient.chat.completions.create method to use the semantic cache.
    """
    global _original_async_create
    try:
        from openai.resources.chat.completions import AsyncCompletions
    except ImportError:
        logger.warning("openai package not found, skipping patch_openai")
        return

    if _original_async_create is not None:
        return

    _original_async_create = AsyncCompletions.create

    @wraps(_original_async_create)
    async def cached_create(self, *args, **kwargs):
        import os
        # Determine TTL
        global_ttl_env = os.getenv("AGENTWATCH_CACHE_TTL_DAYS")
        global_ttl = None if global_ttl_env is None else int(global_ttl_env)
        
        ttl = global_ttl
        
        extra_body = kwargs.get("extra_body", {})
        if "agentwatch_metadata" in extra_body:
            override_ttl = extra_body["agentwatch_metadata"].get("cache_ttl_days")
            if override_ttl is not None:
                ttl = int(override_ttl)
                
        messages = kwargs.get("messages", [])
        if messages and ttl != 0:
            # Simple heuristic: use the content of the last message as the prompt
            prompt = messages[-1].get("content", "")
            
            # Use specific TTL if provided
            original_ttl = semantic_cache.ttl_days
            semantic_cache.ttl_days = ttl
            
            # Check cache
            cached_response = await semantic_cache.get(prompt)
            semantic_cache.ttl_days = original_ttl
            
            if cached_response:
                logger.info("Semantic cache hit for OpenAI request.")
                # We must return a structure that looks like an OpenAI response
                from openai.types.chat import ChatCompletion, ChatCompletionMessage
                from openai.types.chat.chat_completion import Choice
                
                return ChatCompletion(
                    id="chatcmpl-cached",
                    choices=[
                        Choice(
                            finish_reason="stop",
                            index=0,
                            message=ChatCompletionMessage(
                                content=cached_response,
                                role="assistant",
                            ),
                        )
                    ],
                    created=0,
                    model=kwargs.get("model", "unknown"),
                    object="chat.completion",
                )

        # Cache miss, call original
        response = await _original_async_create(self, *args, **kwargs)
        
        # Save to cache
        if messages and hasattr(response, "choices") and response.choices and ttl != 0:
            prompt = messages[-1].get("content", "")
            assistant_message = getattr(response.choices[0].message, "content", None)
            if assistant_message:
                await semantic_cache.set(query=prompt, response=assistant_message, metadata={"framework": "openai"})
        
        return response

    AsyncCompletions.create = cached_create

def unpatch_openai():
    """
    Restores the original OpenAI AsyncClient.chat.completions.create method.
    """
    global _original_async_create
    try:
        from openai.resources.chat.completions import AsyncCompletions
    except ImportError:
        return

    if _original_async_create is not None:
        AsyncCompletions.create = _original_async_create
        _original_async_create = None
