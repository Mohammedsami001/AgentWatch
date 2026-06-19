
import pytest

# We'll import SemanticCacheManager once we create it.
# from agentwatch.cost.caching import SemanticCacheManager

@pytest.mark.asyncio
async def test_semantic_cache_manager_store_and_exact_match():
    """
    TDD Behavior 1: SemanticCacheManager can store and exactly match a hashed prompt.
    """
    from agentwatch.cost.caching import SemanticCacheManager
    
    # We use a dummy memory dictionary or a mock session if needed.
    # For now, let's assume the manager can operate on an in-memory dictionary
    # or we mock the DB session.
    manager = SemanticCacheManager()
    
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."
    
    # Store the prompt
    await manager.store(
        prompt=prompt,
        response_text=response,
        framework="openai"
    )
    
    # Search for the exact prompt
    hit = await manager.search(prompt, framework="openai")
    
    assert hit is not None
    assert hit.response_text == response
    assert hit.framework == "openai"

@pytest.mark.asyncio
async def test_semantic_cache_manager_semantic_match(monkeypatch):
    """
    TDD Behavior 2: SemanticCacheManager can semantically match an altered prompt.
    """
    from agentwatch.cost.semantic_cache import SemanticCache
    from agentwatch.memory.engine import EmbeddingProvider

    # Mock the embed method to avoid downloading real models
    async def mock_embed(self, texts):
        return [
            [1.0, 0.0] if "reverse a string" in t else [0.0, 1.0]
            for t in texts
        ]
    
    monkeypatch.setattr(EmbeddingProvider, "embed", mock_embed)
    
    cache = SemanticCache(similarity_threshold=0.90)
    
    original_prompt = "Can you write a python script to reverse a string?"
    response = "Sure! Here is a script: `print(s[::-1])`"
    
    await cache.set(query=original_prompt, response=response, metadata={"framework": "openai"})
    
    # Altered prompt that is semantically similar
    altered_prompt = "Please write a python code to reverse a string."
    
    hit_response = await cache.get(altered_prompt)
    
    assert hit_response == response

@pytest.mark.asyncio
async def test_semantic_cache_manager_ttl(monkeypatch):
    """
    TDD Behavior 3: SemanticCacheManager can expire cache entries after a set TTL.
    """
    import datetime

    from agentwatch.cost.semantic_cache import SemanticCache
    from agentwatch.memory.engine import EmbeddingProvider

    # Mock the embed method
    async def mock_embed(self, texts):
        return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(EmbeddingProvider, "embed", mock_embed)

    cache = SemanticCache(ttl_days=1)
    
    # Store a prompt
    prompt = "What is the capital of France?"
    await cache.set(query=prompt, response="Paris")

    # Override the created_at to be older than the TTL
    cache._cache[0].created_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=2)

    # It should be expired now
    hit = await cache.get(prompt)
    assert hit is None

@pytest.mark.asyncio
async def test_semantic_cache_manager_interception(monkeypatch):
    """
    TDD Behavior 4: Adapter interception catches requests and returns cached responses.
    """
    import openai

    from agentwatch.adapters.interception import patch_openai, unpatch_openai
    from agentwatch.cost.semantic_cache import SemanticCache
    from agentwatch.memory.engine import EmbeddingProvider

    # Mock embed
    async def mock_embed(self, texts):
        return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(EmbeddingProvider, "embed", mock_embed)

    cache = SemanticCache()
    prompt = "What is the largest ocean?"
    await cache.set(query=prompt, response="Pacific Ocean", metadata={"framework": "openai"})

    # Patch OpenAI client
    patch_openai(cache)

    try:
        # Create a mock openai client and call it
        # The patch should intercept this and return the cached response without hitting the network
        client = openai.AsyncClient(api_key="mock")
        
        # We assume the patch intercepts `client.chat.completions.create`
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        assert response.choices[0].message.content == "Pacific Ocean"
    finally:
        unpatch_openai()

@pytest.mark.asyncio
async def test_semantic_cache_manager_config_override(monkeypatch):
    """
    TDD Behavior 5: Per-Session configuration can override global caching toggle.
    """

    import openai

    from agentwatch.adapters.interception import patch_openai, unpatch_openai
    from agentwatch.cost.semantic_cache import SemanticCache
    from agentwatch.memory.engine import EmbeddingProvider
    async def mock_embed(self, texts):
        return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(EmbeddingProvider, "embed", mock_embed)

    # Set global TTL to 0 (disabled)
    monkeypatch.setenv("AGENTWATCH_CACHE_TTL_DAYS", "0")

    cache = SemanticCache()
    prompt = "What is the smallest continent?"
    
    # Even if we set it in cache, it shouldn't hit unless the session overrides
    await cache.set(query=prompt, response="Australia", metadata={"framework": "openai"})
    
    patch_openai(cache)
    try:
        client = openai.AsyncClient(api_key="mock")

        # Mock the underlying network call so it doesn't really fail
        # If cache intercept works properly, we will return Australia, but since TTL is 0,
        # it should miss the cache. However, we'll pass an override to force a hit.
        
        # 1. Without override, should NOT hit cache, but we don't mock network so it would raise an APIConnectionError.
        # So we mock the original method.
        original_create_called = False
        async def mock_network(*args, **kwargs):
            nonlocal original_create_called
            original_create_called = True
            from openai.types.chat import ChatCompletion, ChatCompletionMessage
            from openai.types.chat.chat_completion import Choice
            return ChatCompletion(
                id="net",
                choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(content="Network", role="assistant"))],
                created=0, model="x", object="chat.completion"
            )
        
        # Replace the original with our network mock
        from agentwatch.adapters import interception
        monkeypatch.setattr(interception, "_original_async_create", mock_network)
        
        # This one should hit the network (cache disabled globally)
        resp1 = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        assert resp1.choices[0].message.content == "Network"
        assert original_create_called

        original_create_called = False
        
        # 2. With override, should hit the cache!
        resp2 = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            extra_body={"agentwatch_metadata": {"cache_ttl_days": 1}}
        )
        assert resp2.choices[0].message.content == "Australia"
        assert not original_create_called
    finally:
        unpatch_openai()

@pytest.mark.asyncio
async def test_semantic_cache_manager_db_backend(monkeypatch):
    """
    TDD Behavior 6: SemanticCacheManager integrates with SQLAlchemy/pgvector for scalable storage.
    """
    from unittest.mock import AsyncMock, MagicMock

    from agentwatch.cost.semantic_cache import SemanticCache
    from agentwatch.memory.engine import EmbeddingProvider
    from agentwatch.models.cache import SemanticCacheEntry

    async def mock_embed(self, texts):
        return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(EmbeddingProvider, "embed", mock_embed)

    db_session = MagicMock()
    db_session.execute = AsyncMock()
    db_session.commit = AsyncMock()
    # Mock the execute return value for .get()
    mock_result = MagicMock()
    mock_entry = SemanticCacheEntry(response_text="Blue Whale")
    mock_result.scalars().first.return_value = mock_entry
    db_session.execute.return_value = mock_result

    # Pass the database session to the cache
    cache = SemanticCache(db_session=db_session)
    
    prompt = "What is the largest animal?"
    await cache.set(query=prompt, response="Blue Whale", metadata={"framework": "openai"})
    
    # Verify set called add and commit
    assert db_session.add.called
    assert db_session.commit.called

    hit = await cache.get("What is the biggest animal?")
    assert hit == "Blue Whale"
    
    # Verify execute was called
    assert db_session.execute.called


