# 🚀 Production Deployment Guide

## Deployment Options

### 1. Local Development (Current Setup)
**Use for**: Learning, testing, local development
**URL**: http://localhost:8000 and http://localhost:8080
**Command**: `python dev.py run-all`

---

### 2. Docker Containerization (Recommended)

#### Prerequisites
- Docker installed
- Docker Compose installed

#### Create Dockerfile

**`Dockerfile`**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["python", "dev.py", "run-all"]
```

#### Create docker-compose.yml

**`docker-compose.yml`**:
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
      - BACKEND_PORT=8000
    command: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - backend
    volumes:
      - ./django_app:/app/django_app
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
      - DJANGO_SETTINGS_MODULE=omni_flow.settings
    command: sh -c "cd django_app && python manage.py migrate && python manage.py runserver 0.0.0.0:8080"

volumes:
  data:
  models:
```

#### Deploy with Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

---

### 3. Cloud Deployment

#### AWS EC2

**1. Launch EC2 Instance**:
- Image: Ubuntu 22.04 LTS
- Instance: t3.large (8GB RAM recommended)
- Storage: 100GB EBS
- Security: Allow ports 8000, 8080

**2. SSH and Setup**:
```bash
ssh -i key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3.11 python3-pip python3-venv git

# Clone repository
git clone https://github.com/your-repo/OmniFlow-Sales-AI.git
cd OmniFlow-Sales-AI

# Setup environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service
```

**3. Create Systemd Service** (`/etc/systemd/system/omniflow-backend.service`):
```ini
[Unit]
Description=OmniFlow FastAPI Backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/OmniFlow-Sales-AI
ExecStart=/home/ubuntu/OmniFlow-Sales-AI/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**4. Start Service**:
```bash
sudo systemctl daemon-reload
sudo systemctl start omniflow-backend
sudo systemctl enable omniflow-backend
sudo systemctl status omniflow-backend
```

#### Google Cloud Run

**1. Create Dockerfile** (see above)

**2. Build and Push**:
```bash
gcloud auth configure-docker
docker tag omniflow-api gcr.io/YOUR_PROJECT_ID/omniflow-api:latest
docker push gcr.io/YOUR_PROJECT_ID/omniflow-api:latest
```

**3. Deploy**:
```bash
gcloud run deploy omniflow-backend \
  --image gcr.io/YOUR_PROJECT_ID/omniflow-api:latest \
  --platform managed \
  --region us-central1 \
  --port 8000 \
  --memory 2Gi \
  --timeout 3600
```

#### Heroku

**1. Create `Procfile`**:
```
web: sh -c 'cd django_app && python manage.py migrate && python manage.py runserver 0.0.0.0:$PORT'
release: python manage.py migrate
```

**2. Create `.env`** (local testing):
```
DJANGO_DEBUG=False
ALLOWED_HOSTS=your-app.herokuapp.com
```

**3. Deploy**:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

## Production Configuration

### Environment Variables

Create `.env` file:
```bash
# Django
DEBUG=False
DJANGO_DEBUG=False
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com,localhost

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/omniflow

# FastAPI
BACKEND_URL=http://backend:8000
CORS_ORIGINS=https://yourdomain.com

# ML Settings
MODEL_PATH=/models/champion.joblib
DATA_PATH=/data/raw/

# Logging
LOG_LEVEL=INFO
```

### Database Setup

For production, use PostgreSQL instead of SQLite:

```bash
# Install psycopg2
pip install psycopg2-binary django-environ

# Create database
createdb omniflow

# Run migrations
cd django_app
python manage.py migrate
```

### Security Hardening

**1. Update Django Settings**:
```python
# django_app/omni_flow/settings.py

DEBUG = False
SECRET_KEY = os.getenv('SECRET_KEY')
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost').split(',')

# HTTPS
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000

# CORS
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ORIGINS', '').split(',')
```

**2. Update FastAPI Settings**:
```python
# api/main.py

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:8080').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.get("/protected")
async def protected(credentials: HTTPAuthCredentials = Depends(security)):
    # Verify JWT token
    pass
```

**3. Setup HTTPS** (nginx reverse proxy):
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

---

## Monitoring & Logging

### Application Monitoring

**Prometheus Integration** (optional):

```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('omniflow_requests', 'Total requests', ['endpoint', 'method'])
request_duration = Histogram('omniflow_request_duration', 'Request duration (s)', ['endpoint'])

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    request_count.labels(endpoint=request.url.path, method=request.method).inc()
    request_duration.labels(endpoint=request.url.path).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging Configuration

**Centralized Logging** (e.g., ELK Stack):

```python
import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Now all logs are JSON formatted for centralized systems
```

### Health Checks

```bash
# Manual health check
curl http://localhost:8000/health

# Automated monitoring
* Set up alerts if health endpoint returns non-200
* Monitor disk space for data/models directories
* Alert if drift detection fails
```

---

## Scaling Strategy

### Vertical Scaling
- Increase EC2/GCP instance size
- Increase CPU allocation for XGBoost (`n_jobs=-1` uses all cores)
- Add GPU: `tree_method='gpu_hist'` in XGBoost

### Horizontal Scaling

**Job Queue** (replace in-memory store):

```python
# Use Redis for distributed job store
import redis

redis_client = redis.Redis(host='redis', port=6379)

def _update_job(job_id: str, **updates):
    key = f"job:{job_id}"
    redis_client.hset(key, mapping=updates)
    redis_client.expire(key, 86400)  # 24hr expiry
```

**Load Balancing**:
```nginx
upstream omniflow_backend {
    server backend-1:8000;
    server backend-2:8000;
    server backend-3:8000;
}

server {
    listen 8000;
    location / {
        proxy_pass http://omniflow_backend;
    }
}
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily model backup
0 2 * * * tar czf /backup/models_$(date +\%Y\%m\%d).tar.gz /app/models/

# Weekly data backup
0 3 * * 0 pg_dump omniflow | gzip > /backup/db_$(date +\%Y\%m\%d).sql.gz

# Upload to S3
aws s3 sync /backup s3://my-backup-bucket/omniflow/
```

### Model Versioning

```python
# Store model versions
# models/
#   ├── champion.joblib (current)
#   ├── challenger.joblib (testing)
#   └── archive/
#       ├── v1_2024_01_15.joblib
#       ├── v2_2024_02_20.joblib

def save_model_version(model, version_id):
    version_path = f"models/archive/v{version_id}_{datetime.now().isoformat()}.joblib"
    joblib.dump(model, version_path)
```

---

## Performance Optimization

### Caching

```python
from functools import lru_cache
import redis

# In-memory cache for reference stats
@lru_cache(maxsize=1)
def get_reference_stats():
    with open("models/reference_stats.json") as f:
        return json.load(f)

# Redis cache for predictions
redis_cache = redis.Redis()

def cache_prediction(key, prediction, ttl=3600):
    redis_cache.setex(key, ttl, json.dumps(prediction))
```

### Database Optimization

```python
# Add indexes for frequent queries
# PostgreSQL:
CREATE INDEX idx_date ON sales_data(date);
CREATE INDEX idx_store_date ON sales_data(store_nbr, date);

# Django ORM:
class Sales(models.Model):
    date = models.DateTimeField(db_index=True)
    store_nbr = models.IntegerField(db_index=True)
    class Meta:
        indexes = [
            models.Index(fields=['date', 'store_nbr']),
        ]
```

---

## Troubleshooting Production Issues

### Out of Memory
```bash
# Check memory usage
free -h
ps aux | grep python | head -5

# Reduce model size
# - Reduce n_estimators in XGBoost
# - Use simpler preprocessing
# - Archive old data
```

### High Latency
```bash
# Monitor request times
curl -w "@time.txt" http://localhost:8000/predict

# Profile hot spots
python -m cProfile -s cumulative api/main.py > profile.txt

# Scale horizontally with load balancer
```

### Model Stale
```bash
# Monitor drift detection frequency
# If drift detected:
# 1. Review challenger performance
# 2. Manually promote if better
# 3. Document decision

# Set up automated retraining schedule
0 0 * * * python -c "from src.retrain import schedule_retraining; schedule_retraining()"
```

---

## Rollback Procedure

If issues occur after deployment:

```bash
# 1. Stop current deployment
docker-compose down
# or
systemctl stop omniflow-backend omniflow-frontend

# 2. Restore previous model version
cp models/archive/v1_previous.joblib models/champion.joblib

# 3. Review logs for error
tail -100 /var/log/omniflow.log

# 4. Restart with previous code
git checkout previous-commit
# or
docker-compose up -d --build

# 5. Run smoke tests
curl http://localhost:8000/health
```

