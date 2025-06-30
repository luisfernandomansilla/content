# Production Deployment Guide

## 🚀 Environment Configuration

### Development vs Production

Content Creator automatically configures itself based on the `ENVIRONMENT` variable:

#### Development (Default)
```bash
ENVIRONMENT=development
HOST=127.0.0.1
PORT=7860
```

#### Production
```bash
ENVIRONMENT=production
HOST=0.0.0.0
PORT=80
```

### Configuration Files

1. **Copy example configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` for production:**
   ```bash
   # Production settings
   ENVIRONMENT=production
   HOST=0.0.0.0
   PORT=80
   
   # Your actual tokens
   HF_TOKEN=your_real_huggingface_token_here
   CIVITAI_API_TOKEN=your_real_civitai_token_here
   
   # Performance settings
   CLEANUP_TEMP_FILES=true
   AUTO_OPTIMIZE_SETTINGS=true
   ```

## 🔧 FLUX+LoRA Compatibility Fix

### Issue Resolved
- **Problem**: FLUX models with LoRA (like `Flux-NSFW-uncensored`) were generating gradient images instead of actual content
- **Root Cause**: XFormers incompatibility with FLUX+LoRA causing `attn_output` reference errors
- **Solution**: Automatic XFormers disabling for FLUX models

### Technical Details

The system now automatically:
- ✅ Detects FLUX and FLUX+LoRA models
- ✅ Disables XFormers for these models
- ✅ Uses standard attention mechanisms
- ✅ Maintains performance with other optimizations

## ⚡ Pipeline Caching System

### Performance Enhancement
- **Problem**: Models were being loaded from scratch on every generation (2-5 minutes each time)
- **Solution**: Intelligent pipeline caching system
- **Benefit**: First load takes time, subsequent generations are instant

### How It Works

**First Generation:**
```
INFO:src.content_creator.image_generator:🔄 Loading new pipeline for Flux-NSFW-uncensored
[Loading process takes 2-5 minutes]
INFO:src.content_creator.image_generator:✅ Pipeline cached for Flux-NSFW-uncensored
```

**Subsequent Generations:**
```
INFO:src.content_creator.image_generator:📋 Using cached pipeline for Flux-NSFW-uncensored
[Generation starts immediately]
```

### Cache Features
- ✅ **Automatic caching**: Models cached after first load
- ✅ **Memory management**: Efficient GPU memory usage
- ✅ **Multi-model support**: Different models cached separately
- ✅ **Manual clearing**: Cache can be cleared to free memory

### Resolution Optimization
- ✅ **FLUX models**: Dimensions automatically rounded to multiples of 16
- ✅ **Other models**: Dimensions rounded to multiples of 8
- ✅ **No manual adjustment**: System handles compatibility automatically

### Logs to Expect

**Successful FLUX+LoRA Loading:**
```
INFO:src.content_creator.image_generator:✅ Base FLUX model loaded without variant!
INFO:src.content_creator.image_generator:🔄 Loading LoRA weights: lora.safetensors
INFO:src.content_creator.image_generator:✅ LoRA weights loaded successfully!
INFO:src.content_creator.image_generator:🚫 Xformers disabled for FLUX+LoRA model (compatibility)
INFO:src.content_creator.image_generator:🔧 Using standard attention mechanism for better stability
```

## 🌐 Deployment Options

### 1. Direct Python Deployment
```bash
# Set production environment
export ENVIRONMENT=production
export PORT=80
export HOST=0.0.0.0

# Run the application
python main.py
```

### 2. Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Production environment variables
ENV ENVIRONMENT=production
ENV HOST=0.0.0.0
ENV PORT=80

EXPOSE 80

CMD ["python", "main.py"]
```

**Run Docker container:**
```bash
docker build -t content-creator .
docker run -p 80:80 \
  -e HF_TOKEN=your_token \
  -e CIVITAI_API_TOKEN=your_token \
  content-creator
```

### 3. systemd Service (Linux)
```ini
[Unit]
Description=Content Creator AI Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/content-creator
Environment=ENVIRONMENT=production
Environment=HOST=0.0.0.0
Environment=PORT=80
Environment=HF_TOKEN=your_token_here
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔒 Security Considerations

### API Tokens
- Store tokens in `.env` file (automatically ignored by git)
- Use environment variables in production
- Restrict token permissions to minimum required

### Network Security
- Use reverse proxy (nginx/Apache) for HTTPS
- Configure firewall rules
- Consider rate limiting

### Example nginx configuration:
```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🧪 Testing Your Deployment

Run the compatibility tests:
```bash
python test_flux_fix.py
```

Expected output:
```
🚀 Content Creator - Compatibility Tests
========================================================

🧪 Testing FLUX+LoRA Compatibility Fix
==================================================
🔍 Testing model: Flux-NSFW-uncensored
✅ Model found: flux_lora
📋 Base model: black-forest-labs/FLUX.1-dev
🚀 Testing pipeline loading...
✅ Pipeline loaded successfully!
🔧 XFormers properly disabled for FLUX+LoRA

🌐 Testing Production Configuration
==================================================
🔧 Environment: production
🌐 Host: 0.0.0.0
🔌 Port: 80
🎨 Theme: soft
✅ Production configuration looks correct!

📊 Test Results Summary
==============================
✅ Passed: 2/2
🎉 All tests passed! The fixes are working correctly.
```

## 🚨 Troubleshooting

### Common Issues

1. **Port 80 Permission Denied**
   - Run as root: `sudo python main.py`
   - Or use port > 1024: `PORT=8080`

2. **XFormers Still Causing Issues**
   - Check logs for "Xformers disabled" messages
   - Ensure latest code is deployed

3. **Token Authentication Errors**
   - Verify token validity on HuggingFace
   - Check token permissions for model access

4. **Memory Issues**
   - Monitor GPU/CPU memory usage
   - Adjust `MAX_CONCURRENT_GENERATIONS=1`
   - Enable `CLEANUP_TEMP_FILES=true`

### Debug Mode
```bash
DEBUG=true python main.py
```

This will show detailed logs including:
- Model loading process
- Optimization decisions
- Token usage (partially masked)
- Performance metrics

## 📊 Performance Monitoring

### Key Metrics to Monitor
- Model loading time (first time: 2-5 minutes)
- Generation time (30s-2min after warmup)
- Memory usage (GPU/CPU)
- Disk space (model cache growth)

### Optimization Tips
- Keep `AUTO_OPTIMIZE_SETTINGS=true`
- Use SSD storage for model cache
- Ensure adequate GPU memory
- Monitor temperature under load

## 🔄 Updates and Maintenance

### Updating the Application
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Clearing Model Cache
```bash
rm -rf models/models--*/
```

### Backup Important Data
- Configuration files (`.env`)
- Generated outputs (`outputs/`)
- Custom model configurations 