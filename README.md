# Content Creator - AI Video & Image Generation Suite

A comprehensive toolkit for generating high-quality videos and images using state-of-the-art AI models including FLUX, Stable Diffusion, and AnimateDiff.

## 🚀 Features

- **Multi-Model Support**: Supports FLUX, Stable Diffusion, SDXL, AnimateDiff, and custom models
- **Advanced Image Generation**: High-quality image synthesis with multiple checkpoints support
- **Professional Video Creation**: Cutting-edge video generation with temporal consistency
- **Flexible Model Management**: Easy model downloading and switching between platforms
- **Hardware Optimization**: Automatic optimization for different hardware configurations
- **Long Prompt Support**: Advanced handling of prompts that exceed CLIP's 77-token limit
- **Gallery Management**: Built-in file browser and management system

## 🔤 Long Prompt Handling

### The CLIP 77-Token Limitation

Many AI image models use CLIP as their text encoder, which has a hard limit of 77 tokens per prompt. This means that longer, more detailed prompts get truncated, potentially losing important information.

### Our Solutions

1. **Automatic Model Recommendations**: The interface automatically detects long prompts and recommends FLUX models
2. **Smart Prompt Prioritization**: For CLIP-based models, long prompts are automatically prioritized to keep the most important elements
3. **Compel Integration**: Optional advanced prompt processing for better handling of long prompts
4. **Real-time Analysis**: The UI shows estimated token count and warnings for prompts that may be truncated

### Model Recommendations for Long Prompts

**✅ Best for Long Prompts (No Token Limit):**
- FLUX.1-dev
- FLUX.1-schnell  
- Flux-NSFW-uncensored

**⚠️ Limited by CLIP (77 tokens):**
- Stable Diffusion models
- SDXL models  
- Most checkpoint/LoRA models

### Installing Compel (Optional)

For enhanced long prompt support with CLIP-based models, install Compel:

```bash
pip install compel>=2.0.0
```

Compel provides advanced prompt processing that can help work around some CLIP limitations.

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or Apple Silicon Mac
- 8GB+ RAM (16GB+ recommended)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/content-creator.git
   cd content-creator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

4. **Access the interface**: Open your browser to `http://localhost:7860`

## 🔧 Configuration

### API Keys (Optional)

For accessing gated models or additional features:

1. **Hugging Face Token**: Required for some models
   - Get your token from https://huggingface.co/settings/tokens
   - Add it in the Settings tab

2. **CivitAI API Key**: For CivitAI model downloads
   - Get your key from https://civitai.com/user/account
   - Add it in the Settings tab

## 📖 Usage Guide

### Image Generation

1. **Select Models**: Choose a base model and optional checkpoints
2. **Write Your Prompt**: Describe the image you want (long prompts work best with FLUX)
3. **Configure Settings**: Adjust resolution, quality, and style
4. **Generate**: Click "Generate Image" and wait for results

### Video Generation

1. **Choose Video Model**: Select from AnimateDiff, Text2Video-Zero, etc.
2. **Enter Prompt**: Describe the video content
3. **Set Parameters**: Duration, FPS, resolution
4. **Generate**: Create your video

### Prompt Optimization Tips

- **For FLUX models**: Use detailed, descriptive prompts without token concerns
- **For Stable Diffusion**: Keep prompts focused on key elements
- **Quality terms**: Include "masterpiece, best quality, detailed" early in the prompt
- **Character descriptions**: Put character details at the beginning
- **Style modifiers**: Use the style dropdown instead of adding style terms to your prompt

## 🛠️ Model Management

### Supported Platforms

- **Hugging Face**: Thousands of open-source models
- **CivitAI**: Community-created checkpoints and LoRAs
- **Local Files**: Load your own `.safetensors` and `.ckpt` files

### Model Types

- **Base Models**: Foundation models (FLUX, SD, SDXL)
- **Checkpoints**: Fine-tuned models for specific styles
- **LoRAs**: Lightweight adapters for style/character modifications

## 🔍 Gallery & File Management

The built-in gallery provides:
- **Visual Browser**: Preview all generated content
- **File Statistics**: Track storage usage and file counts
- **Bulk Operations**: Clear all files or delete individual items
- **Auto-Refresh**: Gallery updates automatically when new content is generated

## ⚡ Performance Optimization

### Hardware-Specific Features

**Apple Silicon (M1/M2/M3)**:
- Automatic Metal acceleration
- Optimized memory management
- Model recommendations based on available memory

**NVIDIA GPUs**:
- CUDA acceleration
- Mixed precision training
- Memory-efficient attention

**CPU Fallback**:
- Automatic fallback for systems without dedicated GPU
- Optimized for slower but still functional generation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Diffusers library
- Black Forest Labs for FLUX models
- Stability AI for Stable Diffusion
- The open-source AI community

## 📞 Support

- **Issues**: Report bugs or request features on GitHub Issues
- **Discussions**: Join our community discussions
- **Documentation**: Check our wiki for detailed guides

---

**Ready to create amazing AI content? Start generating today!** 🎨✨ 