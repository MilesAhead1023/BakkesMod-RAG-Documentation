# BakkesMod RAG GUI - Deployment Guide

## Overview

This guide covers deploying the BakkesMod RAG GUI application for various environments.

## Table of Contents

1. [Windows Executable (Easiest)](#windows-executable)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Security Considerations](#security-considerations)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Windows Executable

**Easiest option for Windows users - No Python installation required!**

### For End Users

1. **Download the pre-built executable:**
   - Get `BakkesMod_RAG_GUI.zip` from [GitHub Releases](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/releases)
   - Extract to your desired location

2. **Configure API keys:**
   ```cmd
   copy .env.example .env
   notepad .env
   ```
   Add your OpenAI, Anthropic, and Google API keys

3. **Run:**
   ```cmd
   BakkesMod_RAG_GUI.exe
   ```

See [EXE_USER_GUIDE.md](EXE_USER_GUIDE.md) for detailed instructions.

### For Developers (Building from Source)

Build your own executable:

```cmd
build_exe.bat
```

See [BUILD_EXE_GUIDE.md](BUILD_EXE_GUIDE.md) for detailed build instructions.

---

## Local Development

### Quick Start

**Prerequisites:**
- Python 3.8+
- pip package manager
- API keys (OpenAI, Anthropic)

**Steps:**

1. **Clone and setup:**
   ```cmd
   git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
   cd BakkesMod-RAG-Documentation
   ```

2. **Configure environment:**
   ```cmd
   copy .env.example .env
   REM Edit .env and add your API keys
   ```

3. **Launch GUI:**
   
   **Windows:**
   ```cmd
   start_gui.bat
   ```
   
   **Linux/Mac:**
   ```bash
   chmod +x start_gui.sh
   ./start_gui.sh
   ```

The GUI will be available at `http://localhost:7860`

### Manual Installation

If you prefer manual setup:

**Windows:**
```cmd
REM Create virtual environment
python -m venv venv
venv\Scripts\activate

REM Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Launch
python rag_gui.py
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch
python rag_gui.py
```

---

## Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t bakkesmod-rag-gui:latest .

# Or with custom Dockerfile
docker build -f Dockerfile.gui -t bakkesmod-rag-gui:latest .
```

### Run Container

```bash
# Run with environment file
docker run -d \
  --name bakkesmod-rag \
  -p 7860:7860 \
  --env-file .env \
  bakkesmod-rag-gui:latest

# Run with environment variables
docker run -d \
  --name bakkesmod-rag \
  -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  bakkesmod-rag-gui:latest
```

### Docker Compose

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
      - rag-storage:/app/rag_storage_bakkesmod
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

### Custom Dockerfile for GUI

Create `Dockerfile.gui`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rag_gui.py .
COPY cache_manager.py .
COPY query_rewriter.py .
COPY code_generator.py .
COPY code_validator.py .
COPY code_templates.py .
COPY docs_bakkesmod_only ./docs_bakkesmod_only
COPY templates ./templates

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:7860 || exit 1

# Run application
CMD ["python", "rag_gui.py"]
```

---

## Production Deployment

### Prerequisites

- Server with public IP or domain
- SSL certificate (Let's Encrypt recommended)
- Reverse proxy (nginx, Caddy, or Traefik)
- API keys for LLM providers
- Monitoring tools (optional but recommended)

### Option 1: Nginx Reverse Proxy

**Install nginx:**
```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

**Configure nginx:**

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
    
    # Rate limiting (optional)
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

**Enable and start:**
```bash
sudo ln -s /etc/nginx/sites-available/bakkesmod-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### Option 2: Caddy Reverse Proxy

**Install Caddy:**
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

**Configure Caddy:**

Create `/etc/caddy/Caddyfile`:

```
your-domain.com {
    reverse_proxy localhost:7860
    
    # Automatic HTTPS
    encode gzip
    
    # Rate limiting (requires plugin)
    # rate_limit {
    #     zone static 60r/m
    # }
}
```

**Start Caddy:**
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
ReadWritePaths=/opt/bakkesmod-rag/.cache /opt/bakkesmod-rag/rag_storage_bakkesmod

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable bakkesmod-rag
sudo systemctl start bakkesmod-rag
sudo systemctl status bakkesmod-rag
```

---

## Cloud Deployment

### AWS EC2

**1. Launch EC2 Instance:**
- AMI: Ubuntu 22.04 LTS
- Instance type: t3.medium or larger
- Security group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

**2. Connect and setup:**
```bash
ssh ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv nginx

# Clone repository
git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
cd BakkesMod-RAG-Documentation

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure .env
cp .env.example .env
nano .env  # Add API keys

# Setup as systemd service (see above)
```

**3. Configure nginx** (see Production Deployment section)

### Google Cloud Run

**1. Build container:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/bakkesmod-rag
```

**2. Deploy:**
```bash
gcloud run deploy bakkesmod-rag \
  --image gcr.io/PROJECT_ID/bakkesmod-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=xxx,ANTHROPIC_API_KEY=xxx
```

### Heroku

**1. Create `Procfile`:**
```
web: python rag_gui.py --server-port=$PORT --server-name=0.0.0.0
```

**2. Deploy:**
```bash
heroku create bakkesmod-rag
heroku config:set OPENAI_API_KEY=xxx
heroku config:set ANTHROPIC_API_KEY=xxx
git push heroku main
```

### Digital Ocean App Platform

**1. Create `app.yaml`:**
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

**2. Deploy via CLI or web console**

---

## Security Considerations

### API Key Management

**Best Practices:**
- ✅ Use environment variables, never hardcode
- ✅ Rotate keys regularly (every 90 days)
- ✅ Use separate keys for dev/staging/production
- ✅ Monitor API usage for anomalies
- ✅ Set spending limits on provider dashboards
- ❌ Never commit `.env` files to git
- ❌ Never expose keys in logs or error messages

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

**Add basic authentication:**

Modify `rag_gui.py`:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=("username", "password"),  # Add authentication
    share=False
)
```

**Add OAuth (advanced):**

Use nginx `auth_request` module or oauth2-proxy.

### Rate Limiting

**Application-level:**

Add to `rag_gui.py`:
```python
from functools import wraps
from time import time
from collections import defaultdict

rate_limit_data = defaultdict(list)

def rate_limit(max_calls=10, time_window=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            calls = rate_limit_data[func.__name__]
            calls[:] = [t for t in calls if now - t < time_window]
            
            if len(calls) >= max_calls:
                return "Rate limit exceeded. Please try again later."
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=10, time_window=60)
def query_rag(self, query, use_cache):
    # ... existing code
```

**Network-level:**

Configure nginx (see Production Deployment section).

### HTTPS/TLS

**Always use HTTPS in production!**

Free SSL certificates: [Let's Encrypt](https://letsencrypt.org/)

```bash
sudo certbot --nginx -d your-domain.com
```

### Content Security Policy

Add to nginx config:
```nginx
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;
```

---

## Monitoring & Maintenance

### Application Monitoring

**1. Logging:**

Configure structured logging in `rag_gui.py`:
```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_gui.log'),
        logging.StreamHandler()
    ]
)
```

**2. Metrics:**

Track key metrics:
- Query count
- Average response time
- Cache hit rate
- Error rate
- API costs

**3. Health Checks:**

Add health endpoint:
```python
def health_check():
    return {
        "status": "healthy",
        "query_engine": "online" if self.query_engine else "offline",
        "cache": "online" if self.cache else "offline"
    }

# In Gradio interface
with gr.Row():
    health_btn = gr.Button("Health Check")
    health_output = gr.JSON()
    health_btn.click(fn=app.health_check, outputs=health_output)
```

### Cost Monitoring

Track API costs:
```python
import tiktoken

def estimate_cost(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(text))
    
    # Cost per 1K tokens (example)
    cost_per_1k = {
        "gpt-4": 0.03,
        "claude-3-5-sonnet": 0.015,
        "text-embedding-3-small": 0.00002
    }
    
    return (tokens / 1000) * cost_per_1k.get(model, 0)
```

### Backup & Recovery

**Backup indices:**
```bash
# Backup RAG storage
tar -czf rag_storage_backup_$(date +%Y%m%d).tar.gz rag_storage_bakkesmod/

# Backup cache
tar -czf cache_backup_$(date +%Y%m%d).tar.gz .cache/
```

**Automated backups:**
```bash
# Add to crontab (daily at 2 AM)
0 2 * * * /opt/bakkesmod-rag/backup.sh
```

### Updates & Maintenance

**Update dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**Rebuild indices:**
```bash
# If documentation is updated
rm -rf rag_storage_bakkesmod/
python rag_gui.py  # Will rebuild on startup
```

**Monitor logs:**
```bash
# System service logs
sudo journalctl -u bakkesmod-rag -f

# Application logs
tail -f rag_gui.log
```

---

## Troubleshooting

### Common Issues

**1. Port already in use:**
```bash
# Find process using port 7860
lsof -i :7860
kill -9 <PID>

# Or change port in rag_gui.py
```

**2. Out of memory:**
```bash
# Check memory usage
free -h

# Increase swap (if needed)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**3. Slow responses:**
- Check API rate limits
- Verify cache is working
- Monitor network latency
- Consider upgrading server

**4. High API costs:**
- Enable semantic caching
- Reduce `similarity_top_k`
- Use cheaper embedding models
- Monitor query patterns

---

## Performance Optimization

### Caching Strategy

1. **Enable semantic caching** (default: 92% similarity)
2. **Adjust TTL** based on doc update frequency
3. **Pre-cache** common questions

### Resource Allocation

**Recommended server specs:**
- **Development**: 2 vCPU, 4GB RAM
- **Production (< 100 users)**: 4 vCPU, 8GB RAM
- **Production (> 100 users)**: 8+ vCPU, 16GB RAM

### Database Optimization

If using Neo4j for knowledge graph:
- Configure memory settings
- Enable query caching
- Index frequently queried properties

---

## Support & Resources

- **Documentation**: [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/issues)
- **BakkesMod**: [bakkesmod.com](https://bakkesmod.com/)

---

**Last Updated**: February 8, 2026
