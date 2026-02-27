# Deployment Guide - 2026 Gold Standard RAG

This guide covers deploying the BakkesMod RAG Documentation system for production use.

> **Platform Note:** Optimized for **Windows 11** as the primary platform. Docker deployment supports all platforms.

## Prerequisites

- **Windows 11** (primary platform) or Linux/Mac (for development)
- Python 3.12+
- Docker and Docker Compose (for containerized deployment)
- API keys for:
  - OpenAI
  - Anthropic
  - Google (Gemini)

## Local Development

### 1. Clone and Setup

**Windows:**
```cmd
git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
cd BakkesMod-RAG-Documentation

REM Create virtual environment
python -m venv venv
venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
cd BakkesMod-RAG-Documentation

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

**Windows:**
```cmd
copy .env.example .env
```

**Linux/Mac:**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Set budget
DAILY_BUDGET_USD=10.0

# Optional: Configure logging
LOG_LEVEL=INFO
```

### 3. Run System

```bash
# Start the interactive CLI
python interactive_rag.py

# Or start the web GUI
python rag_gui.py
```

### 4. Access Observability

If Phoenix and Prometheus are installed:

- **Phoenix UI**: http://localhost:6006
- **Prometheus Metrics**: http://localhost:8000/metrics

## Docker Deployment

### 1. Build Image

```bash
docker build -t bakkesmod-rag:2026 .
```

### 2. Run Container

```bash
docker run -d \
  --name bakkesmod-rag \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -e GOOGLE_API_KEY=${GOOGLE_API_KEY} \
  -p 6006:6006 \
  -p 8000:8000 \
  -v $(pwd)/docs:/app/docs:ro \
  -v rag_storage:/app/rag_storage \
  -v logs:/app/logs \
  bakkesmod-rag:2026
```

### 3. Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Production Deployment

### Cloud Platforms

#### AWS ECS

```bash
# 1. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag bakkesmod-rag:2026 <account>.dkr.ecr.us-east-1.amazonaws.com/bakkesmod-rag:2026
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/bakkesmod-rag:2026

# 2. Create task definition with environment variables
# 3. Deploy to ECS service
```

#### Google Cloud Run

```bash
# 1. Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/bakkesmod-rag

# 2. Deploy
gcloud run deploy bakkesmod-rag \
  --image gcr.io/PROJECT_ID/bakkesmod-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY,ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name bakkesmod-rag --location eastus

# Deploy container
az container create \
  --resource-group bakkesmod-rag \
  --name bakkesmod-rag \
  --image bakkesmod-rag:2026 \
  --environment-variables OPENAI_API_KEY=$OPENAI_API_KEY ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --ports 6006 8000 \
  --dns-name-label bakkesmod-rag
```

### Kubernetes

Create `k8s-deployment.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
type: Opaque
stringData:
  openai-key: YOUR_KEY
  anthropic-key: YOUR_KEY
  google-key: YOUR_KEY
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bakkesmod-rag
spec:
  replicas: 2
  selector:
    matchLabels:
      app: bakkesmod-rag
  template:
    metadata:
      labels:
        app: bakkesmod-rag
    spec:
      containers:
      - name: rag
        image: bakkesmod-rag:2026
        ports:
        - containerPort: 6006
          name: phoenix
        - containerPort: 8000
          name: metrics
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: anthropic-key
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: google-key
        volumeMounts:
        - name: storage
          mountPath: /app/rag_storage
        - name: docs
          mountPath: /app/docs
          readOnly: true
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: rag-storage-pvc
      - name: docs
        configMap:
          name: rag-docs
---
apiVersion: v1
kind: Service
metadata:
  name: bakkesmod-rag
spec:
  selector:
    app: bakkesmod-rag
  ports:
  - name: phoenix
    port: 6006
    targetPort: 6006
  - name: metrics
    port: 8000
    targetPort: 8000
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes | - | Anthropic API key |
| `GOOGLE_API_KEY` | Yes | - | Google API key |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DAILY_BUDGET_USD` | No | None | Daily spending limit |

### Advanced Configuration

Edit `config.py` to customize:

- **Embedding model**: `embedding.model`
- **Primary LLM**: `llm.primary_model`
- **Retrieval settings**: `retriever.*`
- **Cost limits**: `cost.daily_budget_usd`
- **Rate limiting**: `production.requests_per_minute`
- **Circuit breaker**: `production.failure_threshold`

## Monitoring

### Metrics

Access Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `rag_queries_total` - Total queries
- `rag_query_latency_seconds` - Latency histogram
- `rag_llm_tokens_total` - Token usage
- `rag_daily_cost_usd` - Current daily cost

### Phoenix Tracing

Access Phoenix UI at http://localhost:6006 for:
- LLM call traces
- Token usage analysis
- Latency breakdown
- Error tracking

### Structured Logs

Logs are JSON-formatted and can be ingested into:
- **CloudWatch Logs** (AWS)
- **Cloud Logging** (GCP)
- **Application Insights** (Azure)
- **Elasticsearch/Kibana**
- **Datadog/New Relic**

Example log entry:

```json
{
  "timestamp": "2026-02-06T16:00:00.000Z",
  "level": "INFO",
  "event": "query",
  "query": "How do I get player velocity?",
  "latency_ms": 1234,
  "num_sources": 5,
  "cost_usd": 0.023
}
```

## Health Checks

### HTTP Health Check

The system exposes a Prometheus metrics endpoint that can be used for health checks:

```bash
# Check if metrics endpoint is responding
curl -f http://localhost:8000/metrics || exit 1
```

### Docker Health Check

Included in Dockerfile:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/metrics')"
```

## Scaling

### Horizontal Scaling

The RAG system is stateless (except for the index storage) and can be scaled horizontally:

1. **Shared storage**: Mount the `rag_storage` directory as a shared volume
2. **Load balancer**: Use NGINX or cloud load balancers
3. **Cache**: Consider Redis for distributed semantic caching

### Vertical Scaling

For better performance:

- **CPU**: 4+ cores for parallel retrieval
- **Memory**: 8GB+ for large indices
- **Storage**: SSD for faster index loading

## Troubleshooting

### High Costs

1. Check cost report:
   ```python
   from cost_tracker import get_tracker
   print(get_tracker().get_report())
   ```

2. Reduce budget:
   ```env
   DAILY_BUDGET_USD=5.0
   ```

3. Enable aggressive caching:
   ```python
   # In config.py
   cache.similarity_threshold = 0.85  # Lower = more cache hits
   ```

### Slow Queries

1. Check Phoenix traces for bottlenecks
2. Reduce retrieval top-k:
   ```python
   retriever.vector_top_k = 5
   retriever.kg_top_k = 5
   retriever.bm25_top_k = 5
   ```

3. Use faster models:
   ```python
   llm.primary_model = "gemini-2.0-flash"  # Fastest
   ```

### Circuit Breaker Open

1. Check logs for errors
2. Verify API keys are valid
3. Check provider status pages
4. Wait for recovery timeout (default: 60s)
5. Manually reset if needed

### Out of Memory

1. Reduce checkpoint interval:
   ```python
   storage.checkpoint_interval = 100
   ```

2. Process documents in smaller batches
3. Increase container memory limits

## Security

### API Key Management

**Never commit API keys to git!**

Use:
- Environment variables
- Secret managers (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
- Kubernetes secrets
- HashiCorp Vault

### Network Security

- **TLS/SSL**: Use HTTPS for all external access
- **Firewall**: Restrict access to metrics endpoint
- **Authentication**: Add authentication layer (e.g., API keys, OAuth)

### Data Privacy

- Documentation is stored locally
- No data sent to external services except LLM APIs
- Consider on-premise LLMs for sensitive data

## Backup and Recovery

### Backup Index

```bash
# Backup RAG storage
tar -czf rag_storage_backup_$(date +%Y%m%d).tar.gz rag_storage/

# Backup to S3
aws s3 cp rag_storage_backup_*.tar.gz s3://your-bucket/backups/
```

### Restore Index

```bash
# Download from S3
aws s3 cp s3://your-bucket/backups/rag_storage_backup_20260206.tar.gz .

# Extract
tar -xzf rag_storage_backup_20260206.tar.gz
```

### Rebuild from Scratch

If index is corrupted:

```bash
# Remove existing index
rm -rf rag_storage/

# Rebuild
python -m bakkesmod_rag.comprehensive_builder
```

## Performance Tuning

### Optimal Configuration

For **cost-optimized**:
```python
embedding.model = "text-embedding-3-large"
llm.primary_model = "claude-3-5-sonnet"
llm.kg_model = "gpt-4o-mini"
cache.enabled = True
cache.similarity_threshold = 0.9
```

For **speed-optimized**:
```python
embedding.model = "text-embedding-3-small"
llm.primary_model = "gemini-2.0-flash"
llm.kg_model = "gemini-2.0-flash"
retriever.vector_top_k = 5
retriever.rerank_top_n = 3
```

For **quality-optimized**:
```python
embedding.model = "text-embedding-3-large"
llm.primary_model = "claude-3-5-sonnet"
llm.kg_model = "gpt-4o"
retriever.vector_top_k = 20
retriever.rerank_top_n = 10
cache.enabled = False
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/issues
- Documentation: See `docs/2026-gold-standard-architecture.md`
