# Deploying FastAPI STT Services on GCP/Vertex AI

This guide covers multiple deployment options for your FastAPI STT services on Google Cloud Platform.

## Deployment Options

1. **Compute Engine VMs** - Full control, GPU support
2. **Vertex AI Custom Prediction** - Managed ML deployment
3. **Cloud Run** - Serverless (CPU only)
4. **GKE (Google Kubernetes Engine)** - Container orchestration with GPU

## Option 1: Compute Engine VMs with GPU (Recommended)

### Step 1: Create GPU-enabled VM

```bash
# Set project and zone
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"  # Choose zone with GPU availability

# Create VM with NVIDIA T4 GPU
gcloud compute instances create hausa-stt-vm \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform
```

### Step 2: SSH into VM

```bash
gcloud compute ssh hausa-stt-vm --zone=$ZONE
```

### Step 3: Install Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get install -y ffmpeg git

# Verify GPU
nvidia-smi

# Clone your repository or upload files
git clone https://github.com/yourusername/fastapi-stt.git
cd fastapi-stt

# Install Python dependencies
cd hausa_stt
pip install -r requirements.txt
```

### Step 4: Create Systemd Service

Create `/etc/systemd/system/hausa-stt.service`:

```ini
[Unit]
Description=Hausa STT FastAPI Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/fastapi-stt/hausa_stt
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hausa-stt
sudo systemctl start hausa-stt
sudo systemctl status hausa-stt
```

### Step 5: Configure Firewall

```bash
# Allow HTTP traffic
gcloud compute firewall-rules create allow-hausa-stt \
    --allow=tcp:8000 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow Hausa STT API traffic"
```

### Step 6: Access Service

```bash
# Get external IP
gcloud compute instances describe hausa-stt-vm \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Test
curl http://EXTERNAL_IP:8000/health
```

---

## Option 2: Vertex AI Custom Prediction

### Step 1: Create Docker Image

Create `Dockerfile` in `hausa_stt/`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared utilities
COPY ../shared /app/shared

# Copy service files
COPY . /app/hausa_stt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/hausa_stt/requirements.txt

# Set environment
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/transcribe

WORKDIR /app/hausa_stt

# Expose port
EXPOSE 8080

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Step 2: Build and Push to Artifact Registry

```bash
# Enable APIs
gcloud services enable artifactregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Create repository
gcloud artifacts repositories create stt-models \
    --repository-format=docker \
    --location=us-central1

# Configure Docker auth
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build image
cd hausa_stt
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/stt-models/hausa-stt:v1 .

# Push image
docker push us-central1-docker.pkg.dev/$PROJECT_ID/stt-models/hausa-stt:v1
```

### Step 3: Deploy to Vertex AI

```bash
# Create model
gcloud ai models upload \
    --region=us-central1 \
    --display-name=hausa-stt \
    --container-image-uri=us-central1-docker.pkg.dev/$PROJECT_ID/stt-models/hausa-stt:v1 \
    --container-health-route=/health \
    --container-predict-route=/transcribe \
    --container-ports=8080

# Deploy to endpoint
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=hausa-stt-endpoint

# Get model and endpoint IDs
MODEL_ID=$(gcloud ai models list --region=us-central1 --filter="displayName:hausa-stt" --format="value(name)")
ENDPOINT_ID=$(gcloud ai endpoints list --region=us-central1 --filter="displayName:hausa-stt-endpoint" --format="value(name)")

# Deploy model to endpoint with GPU
gcloud ai endpoints deploy-model $ENDPOINT_ID \
    --region=us-central1 \
    --model=$MODEL_ID \
    --display-name=hausa-stt-v1 \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --min-replica-count=1 \
    --max-replica-count=3
```

### Step 4: Test Endpoint

```bash
# Get endpoint URL
ENDPOINT_URL=$(gcloud ai endpoints describe $ENDPOINT_ID \
    --region=us-central1 \
    --format="value(deployedModels[0].dedicatedResources.endpoint)")

# Test
curl -X POST $ENDPOINT_URL/transcribe \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -F "file=@audio.wav"
```

---

## Option 3: Docker Compose (Development/Testing)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  hausa-stt:
    build:
      context: .
      dockerfile: hausa_stt/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  yoruba-stt:
    build:
      context: .
      dockerfile: yoruba_stt/Dockerfile
    ports:
      - "8001:8000"
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  multilingual-stt:
    build:
      context: .
      dockerfile: multilingual_stt/Dockerfile
    ports:
      - "8002:8000"
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
```

Run with:
```bash
docker-compose up -d
```

---

## Option 4: GKE with GPU Node Pool

### Step 1: Create GKE Cluster

```bash
# Create cluster
gcloud container clusters create stt-cluster \
    --zone=us-central1-a \
    --num-nodes=1 \
    --machine-type=n1-standard-4 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5

# Add GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=stt-cluster \
    --zone=us-central1-a \
    --num-nodes=1 \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=3

# Install NVIDIA GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Step 2: Create Kubernetes Manifests

Create `k8s/hausa-stt-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hausa-stt
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hausa-stt
  template:
    metadata:
      labels:
        app: hausa-stt
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
      - name: hausa-stt
        image: us-central1-docker.pkg.dev/PROJECT_ID/stt-models/hausa-stt:v1
        ports:
        - containerPort: 8000
        env:
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "expandable_segments:True"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hausa-stt-service
spec:
  type: LoadBalancer
  selector:
    app: hausa-stt
  ports:
  - port: 80
    targetPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hausa-stt-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hausa-stt
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Step 3: Deploy to GKE

```bash
# Apply manifests
kubectl apply -f k8s/hausa-stt-deployment.yaml

# Get service external IP
kubectl get service hausa-stt-service

# Test
curl http://EXTERNAL_IP/health
```

---

## Monitoring Setup

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hausa-stt'
    static_configs:
      - targets: ['hausa-stt:8000']
    metrics_path: '/metrics'

  - job_name: 'yoruba-stt'
    static_configs:
      - targets: ['yoruba-stt:8000']
    metrics_path: '/metrics'

  - job_name: 'multilingual-stt'
    static_configs:
      - targets: ['multilingual-stt:8000']
    metrics_path: '/metrics'
```

### Cloud Monitoring Integration

```bash
# Install Ops Agent on Compute Engine VM
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install

# Configure to scrape Prometheus metrics
sudo tee /etc/google-cloud-ops-agent/config.yaml << EOF
metrics:
  receivers:
    prometheus:
      type: prometheus
      config:
        scrape_configs:
          - job_name: 'hausa-stt'
            scrape_interval: 30s
            static_configs:
              - targets: ['localhost:8000']
  service:
    pipelines:
      prometheus:
        receivers: [prometheus]
EOF

sudo systemctl restart google-cloud-ops-agent
```

---

## Cost Optimization

### 1. Use Preemptible/Spot VMs

```bash
gcloud compute instances create hausa-stt-vm \
    --preemptible \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    # ... other flags
```

### 2. Auto-scaling Configuration

For GKE, use cluster autoscaler:
```bash
gcloud container clusters update stt-cluster \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=5 \
    --zone=us-central1-a
```

### 3. GPU Sharing (for multiple services)

Run multiple services on same GPU by adjusting `CUDA_VISIBLE_DEVICES` and memory fractions.

---

## Security Best Practices

### 1. Use Cloud IAM

```bash
# Create service account
gcloud iam service-accounts create stt-service-account

# Grant minimal permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stt-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

### 2. Enable HTTPS with Load Balancer

```bash
# Reserve static IP
gcloud compute addresses create hausa-stt-ip --global

# Create SSL certificate
gcloud compute ssl-certificates create hausa-stt-cert \
    --domains=stt.yourdomain.com

# Create load balancer (see GCP docs for full config)
```

### 3. Use VPC and Private IPs

Deploy services in private subnet and use Cloud NAT for outbound traffic.

---

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check driver installation
sudo dmesg | grep -i nvidia
```

### Out of Memory

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Reduce MAX_CONCURRENCY in code
# Reduce model batch size
# Use smaller chunk sizes
```

### Service Not Starting

```bash
# Check logs
sudo journalctl -u hausa-stt -f

# Check port availability
sudo netstat -tulpn | grep 8000

# Verify dependencies
pip list | grep torch
```

---

## Quick Start Script

Save as `deploy-gcp.sh`:

```bash
#!/bin/bash
set -e

PROJECT_ID="your-project-id"
ZONE="us-central1-a"
SERVICE_NAME="hausa-stt"

echo "Creating VM..."
gcloud compute instances create $SERVICE_NAME-vm \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"

echo "Waiting for VM to be ready..."
sleep 60

echo "Copying files..."
gcloud compute scp --recurse ../fastapi-stt $SERVICE_NAME-vm:~ --zone=$ZONE

echo "Installing dependencies..."
gcloud compute ssh $SERVICE_NAME-vm --zone=$ZONE --command="
    sudo apt-get update && sudo apt-get install -y ffmpeg
    cd fastapi-stt/$SERVICE_NAME
    pip install -r requirements.txt
"

echo "Starting service..."
gcloud compute ssh $SERVICE_NAME-vm --zone=$ZONE --command="
    cd fastapi-stt/$SERVICE_NAME
    nohup python main.py > output.log 2>&1 &
"

echo "Getting external IP..."
EXTERNAL_IP=$(gcloud compute instances describe $SERVICE_NAME-vm \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "Service deployed at: http://$EXTERNAL_IP:8000"
echo "Test with: curl http://$EXTERNAL_IP:8000/health"
```

Run with:
```bash
chmod +x deploy-gcp.sh
./deploy-gcp.sh
```

---

## Summary

**Recommended for Production:** Compute Engine VM with GPU + Systemd service
- Full control
- Cost-effective
- Easy to debug
- GPU support

**For Auto-scaling:** GKE with GPU node pools
- Automatic scaling
- High availability
- Container orchestration

**For Managed ML:** Vertex AI Custom Prediction
- Fully managed
- Built-in monitoring
- Auto-scaling
