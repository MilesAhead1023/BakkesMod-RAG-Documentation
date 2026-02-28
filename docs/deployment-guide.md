# Deployment Guide - 2026 Gold Standard RAG

This guide covers deploying the BakkesMod RAG Documentation system for production use.

> **Platform Note:** Optimized for **Windows 11** as the primary platform. Docker deployment supports all platforms.

## Table of Contents

1. [Windows Executable (Easiest)](#windows-executable)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Health Checks](#health-checks)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)
11. [Security](#security)
12. [Backup and Recovery](#backup-and-recovery)
13. [Performance Tuning](#performance-tuning)

---

## Windows Executable (NiceGUI Native App)

**Easiest option for Windows users - No Python installation required!**

The native desktop app uses NiceGUI and opens as a native window (no browser needed). It has 7 tabs: Query, Code Gen, Settings, and more.

### For End Users

1. **Download the pre-built executable:**
   - Get `BakkesModRAG.zip` from [GitHub Releases](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/releases)
   - Extract to your desired location

2. **Configure API keys:**
   ```cmd
   copy .env.example .env
   notepad .env
   ```
   Add your OpenAI, Anthropic, and Google API keys

3. **Run:**
   ```cmd
   BakkesModRAG.exe
   ```

See [exe-user-guide.md](exe-user-guide.md) for detailed instructions.

### For Developers (Building from Source)

Run the NiceGUI app in development mode:

```cmd
python nicegui_app.py
```

Build a standalone executable:

```cmd
pyinstaller --clean --noconfirm nicegui_app.spec
```

Output: `dist/BakkesModRAG/BakkesModRAG.exe` (COLLECT/directory mode).

See [build-exe-guide.md](build-exe-guide.md) for detailed build instructions.

---

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

**Windows (launches Gradio web GUI at localhost:7860):**
```cmd
start_gui.bat
```

**Linux/Mac (using launch scripts):**
```bash
chmod +x start_gui.sh
./start_gui.sh
```

The GUI will be available at `http://localhost:7860`.

**Or launch manually:**
```bash
# Start the native desktop app (NiceGUI)
python nicegui_app.py

# Start the interactive CLI
python interactive_rag.py

# Or start the Gradio web GUI (used by Docker)
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

### Docker Compose (GUI-specific)

Create `docker-compose.gui.yml`:

```yaml
version: '3.8'

services:
  rag-gui:
    build: .
    container_name: bakkesmod-rag-gui
    ports:
      - "7860:7860"
    env_file:
      - .env
    volumes:
      - ./docs_bakkesmod_only:/app/docs_bakkesmod_only:ro
      - rag-storage:/app/rag_storage
      - cache-data:/app/.cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  rag-storage:
  cache-data:
```

Launch:
```bash
docker-compose -f docker-compose.gui.yml up -d
```

## Production Deployment

### Reverse Proxy (Nginx)

Install nginx and configure a reverse proxy for production:

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/bakkesmod-rag`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;

    # Proxy to Gradio app
    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

Enable and start:
```bash
sudo ln -s /etc/nginx/sites-available/bakkesmod-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### Reverse Proxy (Caddy)

Caddy handles automatic HTTPS:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

Create `/etc/caddy/Caddyfile`:

```
your-domain.com {
    reverse_proxy localhost:7860

    # Automatic HTTPS
    encode gzip
}
```

Start Caddy:
```bash
sudo systemctl start caddy
sudo systemctl enable caddy
```

### Systemd Service

Create `/etc/systemd/system/bakkesmod-rag.service`:

```ini
[Unit]
Description=BakkesMod RAG GUI Application
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/bakkesmod-rag
Environment="PATH=/opt/bakkesmod-rag/venv/bin"
EnvironmentFile=/opt/bakkesmod-rag/.env
ExecStart=/opt/bakkesmod-rag/venv/bin/python rag_gui.py
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/bakkesmod-rag/.cache /opt/bakkesmod-rag/rag_storage

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable bakkesmod-rag
sudo systemctl start bakkesmod-rag
sudo systemctl status bakkesmod-rag
```

### Cloud Platforms

#### AWS EC2

1. Launch EC2 Instance:
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t3.medium or larger
   - Security group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

2. Connect and setup:
```bash
ssh ubuntu@your-instance-ip

sudo apt update
sudo apt install -y python3-pip python3-venv nginx

git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
cd BakkesMod-RAG-Documentation

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env  # Add API keys
```

3. Configure nginx (see Reverse Proxy section above)

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

#### Heroku

1. Create `Procfile`:
```
web: python rag_gui.py --server-port=$PORT --server-name=0.0.0.0
```

2. Deploy:
```bash
heroku create bakkesmod-rag
heroku config:set OPENAI_API_KEY=xxx
heroku config:set ANTHROPIC_API_KEY=xxx
git push heroku main
```

#### Digital Ocean App Platform

1. Create `app.yaml`:
```yaml
name: bakkesmod-rag
services:
  - name: web
    github:
      repo: MilesAhead1023/BakkesMod-RAG-Documentation
      branch: main
    run_command: python rag_gui.py --server-port=8080
    envs:
      - key: OPENAI_API_KEY
        scope: RUN_TIME
        value: ${OPENAI_API_KEY}
      - key: ANTHROPIC_API_KEY
        scope: RUN_TIME
        value: ${ANTHROPIC_API_KEY}
```

2. Deploy via CLI or the Digital Ocean web console.

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

### Cost Monitoring

Track API costs with token counting:
```python
import tiktoken

def estimate_cost(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(text))

    # Cost per 1K tokens (example)
    cost_per_1k = {
        "gpt-4": 0.03,
        "claude-sonnet-4-5": 0.015,
        "text-embedding-3-small": 0.00002
    }

    return (tokens / 1000) * cost_per_1k.get(model, 0)
```

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

### Port Already in Use

```bash
# Find process using port 7860
lsof -i :7860
kill -9 <PID>

# Or change the port in rag_gui.py
```

### High Costs

1. Check cost report:
   ```python
   from bakkesmod_rag.cost_tracker import get_tracker
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
   llm.primary_model = "gemini-2.5-flash"  # Fastest
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

**Best Practices:**
- Rotate keys regularly (every 90 days)
- Use separate keys for dev/staging/production
- Monitor API usage for anomalies
- Set spending limits on provider dashboards

**Using AWS Secrets Manager:**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

secrets = get_secret('bakkesmod-rag/api-keys')
os.environ['OPENAI_API_KEY'] = secrets['openai_key']
os.environ['ANTHROPIC_API_KEY'] = secrets['anthropic_key']
```

### Authentication

**Add basic authentication to the Gradio GUI:**

Modify `rag_gui.py`:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=("username", "password"),  # Add authentication
    share=False
)
```

For advanced OAuth, use the nginx `auth_request` module or oauth2-proxy.

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

### Recommended Server Specs

- **Development**: 2 vCPU, 4GB RAM
- **Production (< 100 users)**: 4 vCPU, 8GB RAM
- **Production (> 100 users)**: 8+ vCPU, 16GB RAM

### Caching Strategy

1. **Enable semantic caching** (default: 92% similarity)
2. **Adjust TTL** based on doc update frequency
3. **Pre-cache** common questions

### Optimal Configuration

For **cost-optimized**:
```python
embedding.model = "text-embedding-3-large"
llm.primary_model = "claude-sonnet-4-5"
llm.kg_model = "gpt-4o-mini"
cache.enabled = True
cache.similarity_threshold = 0.9
```

For **speed-optimized**:
```python
embedding.model = "text-embedding-3-small"
llm.primary_model = "gemini-2.5-flash"
llm.kg_model = "gemini-2.5-flash"
retriever.vector_top_k = 5
retriever.rerank_top_n = 3
```

For **quality-optimized**:
```python
embedding.model = "text-embedding-3-large"
llm.primary_model = "claude-sonnet-4-5"
llm.kg_model = "gpt-4o"
retriever.vector_top_k = 20
retriever.rerank_top_n = 10
cache.enabled = False
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/issues
- Documentation: See `CLAUDE.md` for architecture and system design
- GUI Guide: See [gui-user-guide.md](gui-user-guide.md)
- Debug Guide: See [debug-guide.md](debug-guide.md)

---

**Last Updated**: February 2026
