# 🚀 Deployment Guide

This guide covers different deployment options for the Medical Image Classifier.

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM recommended
- GPU optional (for faster inference)

## 🔧 Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medical-image-classifier.git
cd medical-image-classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Test Installation
```bash
python test_classifier.py
```

### 5. Launch Web Interface
```bash
python app.py
```

## 🌐 Web Deployment Options

### Option 1: Hugging Face Spaces (Recommended)
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Deploy to Spaces
gradio deploy
```

### Option 2: Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t medical-classifier .
docker run -p 7860:7860 medical-classifier
```

### Option 3: Cloud Platforms

#### Google Cloud Run
```bash
gcloud run deploy medical-classifier \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS Lambda (with Serverless)
```yaml
# serverless.yml
service: medical-classifier
provider:
  name: aws
  runtime: python3.9
functions:
  classify:
    handler: lambda_handler.classify
    events:
      - http:
          path: classify
          method: post
```

## 🔒 Production Considerations

### Security
- Add authentication for sensitive deployments
- Use HTTPS in production
- Implement rate limiting
- Validate all inputs

### Performance
- Use GPU instances for faster inference
- Implement caching for repeated requests
- Consider model quantization for smaller deployments
- Use CDN for static assets

### Monitoring
- Add logging and metrics
- Implement health checks
- Monitor resource usage
- Set up alerts for failures

## 📊 Scaling

### Horizontal Scaling
- Use load balancers
- Deploy multiple instances
- Implement session management

### Vertical Scaling
- Increase memory/CPU
- Use GPU acceleration
- Optimize model loading

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use smaller models
   - Increase system memory

2. **Slow Inference**
   - Use GPU if available
   - Optimize image preprocessing
   - Consider model quantization

3. **Model Loading Errors**
   - Check internet connection
   - Verify Hugging Face cache
   - Clear model cache if corrupted

### Debug Mode
```bash
# Enable verbose logging
python app.py --debug

# Test with minimal setup
python test_classifier.py --minimal
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review logs for error messages
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Deploying! 🚀**