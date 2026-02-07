# Quick Reference Guide - 2026 Gold Standard RAG

## For Autonomous AI Agents

This is your quick reference for using the BakkesMod RAG Documentation system as a source of truth.

## Quick Start

### Option 1: Python API (Recommended)

```python
from rag_2026 import build_gold_standard_rag

# Initialize system
rag = build_gold_standard_rag()

# Query the documentation
response = rag.query("How do I get the player's car velocity?")
print(response)

# Access source nodes for citations
for node in response.source_nodes:
    print(f"Source: {node.node.metadata.get('file_name')}")
    print(f"Score: {node.score}")
```

### Option 2: MCP Server (For Claude/VSCode)

```bash
python mcp_rag_server.py
```

### Option 3: Docker

```bash
docker-compose up -d
```

## Common Queries

### Code Examples

**Get player car:**
```python
response = rag.query("How do I get the player's car in BakkesMod?")
```

**Hook events:**
```python
response = rag.query("How do I hook the goal scored event?")
```

**ImGui usage:**
```python
response = rag.query("How do I create a button in ImGui?")
```

**Get ball velocity:**
```python
response = rag.query("How do I get the ball's velocity vector?")
```

### Architecture Questions

**Understanding wrappers:**
```python
response = rag.query("What is the difference between ServerWrapper and CarWrapper?")
```

**Event system:**
```python
response = rag.query("How does the BakkesMod event hooking system work?")
```

## Response Format

Every response includes:

```python
{
    "response": str,          # The answer text
    "source_nodes": [         # List of source documents
        {
            "node": {
                "text": str,  # Source text
                "metadata": {
                    "file_name": str,
                    "file_path": str
                }
            },
            "score": float    # Relevance score (0-1)
        }
    ]
}
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional
DAILY_BUDGET_USD=10.0
LOG_LEVEL=INFO
```

### Quick Configuration Changes

```python
from config import get_config

config = get_config()

# Change LLM provider
config.llm.primary_provider = "gemini"  # or "openai", "anthropic"

# Change embedding model  
config.embedding.model = "text-embedding-3-small"  # faster, cheaper

# Adjust retrieval
config.retriever.vector_top_k = 5  # reduce for speed
config.retriever.rerank_top_n = 3  # reduce for speed

# Cost control
config.cost.daily_budget_usd = 5.0
```

## Cost Control

### Check Current Costs

```python
from cost_tracker import get_tracker

tracker = get_tracker()
print(tracker.get_report(days=7))
```

### Set Budget Alert

```python
from config import get_config

config = get_config()
config.cost.daily_budget_usd = 10.0  # Hard limit
config.cost.alert_threshold_pct = 80.0  # Alert at 80%
```

## Monitoring

### View Metrics (Prometheus)

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `rag_queries_total` - Total queries
- `rag_query_latency_seconds` - Latency
- `rag_daily_cost_usd` - Current cost
- `rag_cache_hits_total` - Cache performance

### View Traces (Phoenix)

Open browser: http://localhost:6006

See:
- LLM call traces
- Token usage
- Latency breakdown
- Error tracking

## Error Handling

### Circuit Breaker States

```
CLOSED -> Normal operation
OPEN -> Provider failing, using fallback
HALF_OPEN -> Testing recovery
```

### Common Errors

**"Circuit breaker OPEN"**
- Provider is experiencing issues
- System automatically using fallback
- Will retry after recovery timeout (60s)

**"Daily budget exceeded"**
- Cost limit reached
- Increase budget or wait until tomorrow
- Check cost report for details

**"Rate limit exceeded"**
- Too many requests per minute
- System will automatically retry
- Consider reducing query frequency

## Performance Tips

### For Speed

```python
# Use faster models
config.llm.primary_model = "gemini-2.0-flash"

# Reduce retrieval
config.retriever.vector_top_k = 5
config.retriever.rerank_top_n = 3

# Enable aggressive caching
config.cache.similarity_threshold = 0.85
```

### For Quality

```python
# Use best models
config.llm.primary_model = "claude-3-5-sonnet"
config.embedding.model = "text-embedding-3-large"

# Increase retrieval
config.retriever.vector_top_k = 20
config.retriever.rerank_top_n = 10

# Disable caching
config.cache.enabled = False
```

### For Cost

```python
# Use cheap models
config.llm.primary_model = "gpt-4o-mini"
config.embedding.model = "text-embedding-3-small"

# Enable caching
config.cache.enabled = True
config.cache.similarity_threshold = 0.9

# Reduce retrieval
config.retriever.vector_top_k = 5
```

## Best Practices

### 1. Always Check Sources

```python
response = rag.query("...")

# Verify sources are relevant
for node in response.source_nodes[:3]:  # Top 3 sources
    print(f"✓ {node.node.metadata['file_name']}: {node.score:.2f}")
```

### 2. Handle Errors Gracefully

```python
try:
    response = rag.query("...")
except Exception as e:
    # System has fallbacks, but handle edge cases
    print(f"Query failed: {e}")
    # Retry or use cached data
```

### 3. Monitor Costs

```python
from cost_tracker import get_tracker

tracker = get_tracker()
daily_cost = tracker.get_daily_cost()

if daily_cost > 5.0:
    print(f"⚠️ Daily cost: ${daily_cost:.2f}")
```

### 4. Cache Similar Queries

The system automatically caches similar queries (>90% similarity). Take advantage:

```python
# These will likely hit cache
rag.query("How do I get player car?")
rag.query("How to get the player's car?")  # Similar -> cache hit
```

### 5. Use Structured Logging

```python
from observability import get_logger

logger = get_logger()

# Log important operations
logger.log_query("User query", metadata={"user_id": "agent_123"})
```

## Integration Examples

### Autonomous Agent Loop

```python
from rag_2026 import build_gold_standard_rag
from cost_tracker import get_tracker

# Initialize
rag = build_gold_standard_rag()
tracker = get_tracker()

while True:
    # Check budget
    if tracker.get_daily_cost() > 10.0:
        print("Budget exceeded, pausing...")
        break
    
    # Get user question
    question = get_user_input()
    
    # Query RAG
    try:
        response = rag.query(question)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        continue
```

### Batch Processing

```python
queries = [
    "How do I get player car?",
    "How do I hook goal event?",
    "How do I use ImGui?",
]

results = []
for query in queries:
    response = rag.query(query)
    results.append({
        "query": query,
        "answer": str(response),
        "sources": len(response.source_nodes)
    })

# Report
for r in results:
    print(f"Q: {r['query']}")
    print(f"A: {r['answer'][:100]}...")
    print(f"Sources: {r['sources']}\n")
```

### Plugin Development Assistant

```python
def get_code_example(task: str) -> str:
    """Get code example for BakkesMod task."""
    response = rag.query(f"Show me code example for: {task}")
    
    # Extract code from response
    code = extract_code_blocks(str(response))
    return code

# Use in plugin development
velocity_code = get_code_example("getting ball velocity")
print(velocity_code)
```

## Troubleshooting

### Problem: Slow Queries

**Solution:**
1. Check Phoenix traces for bottleneck
2. Reduce retrieval top-k values
3. Use faster LLM model
4. Enable caching

### Problem: High Costs

**Solution:**
1. Check cost report: `tracker.get_report()`
2. Identify expensive operations
3. Enable caching: `config.cache.enabled = True`
4. Use cheaper models: `config.llm.primary_model = "gpt-4o-mini"`

### Problem: Irrelevant Answers

**Solution:**
1. Check source nodes for relevance
2. Increase retrieval top-k
3. Enable reranking
4. Use better embedding model

### Problem: System Unavailable

**Solution:**
1. Check circuit breaker state
2. Verify API keys are valid
3. Check provider status pages
4. Wait for recovery timeout (60s)

## Quick Commands

```bash
# Test configuration
python config.py

# Check costs
python cost_tracker.py

# Test observability
python observability.py

# Test resilience
python resilience.py

# Run RAG system
python rag_2026.py "Your question here"

# Deploy with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Health check
curl http://localhost:8000/metrics
```

## Support

- **Documentation**: See `docs/2026-gold-standard-architecture.md`
- **Deployment**: See `docs/deployment-guide.md`
- **Issues**: https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/issues
