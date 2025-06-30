# Content Creator

A Python-based video and image generation tool powered by vLLM and Gradio that creates videos and images from text prompts with customizable styles, durations, and reference images.

## Features

- 🎬 **Text-to-Video Generation**: Create videos from descriptive prompts
- 🎨 **Text-to-Image Generation**: Generate high-quality images with multiple models
- 🖼️ **Style Control**: Choose between anime, realistic, and custom styles
- 📷 **Reference Images**: Use one or multiple images to guide video/image style
- ⚙️ **Customizable Parameters**: Control duration, resolution, quality, and advanced settings
- 🖥️ **Cross-Platform**: Automatic hardware detection (Apple Silicon M4, NVIDIA GPUs, CPU fallback)
- 🌐 **Web Interface**: User-friendly Gradio interface with separate tabs
- 🚀 **High Performance**: Optimized with vLLM for efficient inference
- 🔞 **NSFW Support**: Includes uncensored models for mature content generation
- 🤗 **Hugging Face Integration**: Auto-download models on demand
- 🎨 **Civitai Integration**: Access community models from Civitai platform

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space for models
- **Python**: 3.9 - 3.12

### Recommended Hardware
- **Apple Silicon**: MacBook Pro M4 or newer
- **NVIDIA GPUs**: RTX 3060 or better (12GB+ VRAM)
- **RAM**: 32GB+ for high-resolution video generation

## Installation

### Prerequisites

1. **Install Python 3.9-3.12**
   ```bash
   python --version  # Verify Python version
   ```

2. **Install uv (recommended package manager)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/content-creator.git
   cd content-creator
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   uv venv --python 3.12 --seed
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

### Manual Installation

If you prefer using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Launch the Gradio interface:
```bash
python main.py
```

Then open your browser to `http://localhost:7860`

The interface includes three main tabs:
- **🎥 Generate Video**: Create videos from text prompts
- **🎨 Generate Image**: Create images from text prompts  
- **🔍 Browse Models**: Search and download models from Hugging Face and Civitai
- **⚙️ Settings**: Configure API keys and cache management

### API Usage

#### Video Generation
```python
from content_creator import VideoGenerator

generator = VideoGenerator()
video_path = generator.generate(
    prompt="A beautiful sunset over mountains",
    style="realistic",
    duration=10,
    resolution="1080p",
    reference_images=["path/to/reference.jpg"]
)
```

#### Image Generation
```python
from content_creator import ImageGenerator

image_generator = ImageGenerator()
image_path = image_generator.generate(
    prompt="An anime girl with purple hair in a magical forest",
    model_name="Flux-NSFW-uncensored",
    style="Anime",
    resolution="1024p",
    output_format="PNG",
    negative_prompt="blurry, low quality",
    guidance_scale=8.0,
    num_inference_steps=32,
    seed=42
)
```

### Examples

Basic examples are provided in the `examples/` folder:
```bash
# Video generation example
python examples/basic_usage.py

# Image generation example
python examples/image_usage.py

# Civitai integration example
python examples/civitai_usage.py
```

### API Keys Configuration

To access models from different platforms, you may need API keys:

#### Hugging Face API Key
For private models and faster downloads:
```bash
export HUGGINGFACE_API_KEY="your_hf_token_here"
# or
export HF_TOKEN="your_hf_token_here"
```

Get your token from: [Hugging Face Settings](https://huggingface.co/settings/tokens)

#### Civitai API Key
For downloading Civitai models:
```bash
export CIVITAI_API_KEY="your_civitai_key_here"
# or
export CIVITAI_TOKEN="your_civitai_key_here"
```

Get your API key from: [Civitai Account Settings](https://civitai.com/user/account)

> **Note**: Some models require authentication. Set the appropriate API keys to access private or gated models.

## Model Management

### Local Model Caching

Content Creator automatically downloads and caches AI models locally to improve performance and reduce download times on subsequent uses.

#### Cache Location
Models are stored in the `models/` directory within the repository:
```
content-creator/
├── models/                       # Local model cache (excluded from Git)
│   ├── README.md                # Information about cached models
│   ├── models--black-forest-labs--FLUX.1-dev/
│   ├── models--stabilityai--stable-diffusion-xl-base-1.0/
│   └── models--Heartsync--Flux-NSFW-uncensored/
```

#### Benefits
- ✅ **One-time download**: Models are downloaded once and reused
- ✅ **Faster startup**: No re-downloading when restarting the application
- ✅ **Offline usage**: Access cached models without internet connection
- ✅ **Local storage**: Models stay within your project directory

#### Cache Management
The application automatically manages the model cache, but you can manually:

**View cache status** (in Settings tab):
- See which models are cached
- Check total cache size
- View individual model sizes

**Clear cache** (to free disk space):
```bash
# Remove all cached models
rm -rf models/models--*/

# Remove specific model
rm -rf models/models--black-forest-labs--FLUX.1-dev/

# Keep README.md for reference
```

**Note**: The `models/` directory is excluded from Git via `.gitignore` to prevent committing large model files.

#### Storage Requirements
- **FLUX Models**: ~6-12GB each
- **Stable Diffusion Models**: ~2-8GB each
- **LoRA Models**: ~500MB-2GB each
- **Total**: Plan for 20-50GB depending on usage

## Configuration

### Supported Resolutions
- `480p` (854x480)
- `720p` (1280x720) 
- `1080p` (1920x1080)
- `1440p` (2560x1440)
- `4k` (3840x2160)

### Video Styles
- **Realistic**: Photorealistic video generation
- **Anime**: Animated/cartoon style
- **Custom**: Style guided by reference images

### Duration Limits
- **Minimum**: 1 second
- **Maximum**: 60 seconds (adjustable in config)

### Output Formats

#### Video Formats
- **MP4**: Standard video format (default)
- **WebM**: Web-optimized format
- **AVI**: Legacy compatibility
- **MOV**: QuickTime format
- **GIF**: Animated image format

#### Image Formats
- **PNG**: High quality with transparency (default)
- **JPEG**: Standard image format
- **WebP**: Modern efficient format
- **TIFF**: Professional format

## Supported Models

### Video Generation Models
- **AnimateDiff**: Animation-focused video generation (8GB VRAM)
- **Stable Video Diffusion**: High-quality video from images (12GB VRAM)
- **Text2Video-Zero**: Direct text-to-video generation (6GB VRAM)
- **VideoCrafter**: Creative video generation (10GB VRAM)
- **CogVideoX**: Advanced video model (14GB VRAM)

### Image Generation Models
- **FLUX.1-dev**: High-quality text-to-image generation (12GB VRAM)
- **FLUX.1-schnell**: Fast text-to-image generation (8GB VRAM)
- **Flux-NSFW-uncensored** ⚠️: Uncensored FLUX model (12GB VRAM)
- **Stable Diffusion XL**: High-resolution image generation (8GB VRAM)
- **Stable Diffusion 2.1**: Popular text-to-image model (6GB VRAM)
- **Midjourney Style**: Midjourney-style generation (6GB VRAM)
- **DreamShaper**: Versatile image generation (6GB VRAM)
- **Realistic Vision**: Photorealistic image generation (6GB VRAM)

> ⚠️ **Content Warning**: Models marked with ⚠️ may generate NSFW content. Use responsibly and in accordance with your local laws and community guidelines.

## Hardware Detection

The application automatically detects your hardware and optimizes settings:

- **Apple Silicon (M4)**: Uses Metal Performance Shaders optimization
- **NVIDIA GPUs**: Leverages CUDA acceleration with vLLM
- **CPU Fallback**: Uses CPU inference for systems without compatible GPUs

## Project Structure

```
content-creator/
├── src/
│   ├── content_creator/
│   │   ├── __init__.py
│   │   ├── generator.py          # Video generation logic
│   │   ├── image_generator.py    # Image generation logic
│   │   ├── models/               # Model management
│   │   │   ├── model_manager.py  # Model downloading and caching
│   │   │   ├── huggingface_client.py  # HF API integration
│   │   │   └── civitai_client.py # Civitai API integration
│   │   ├── utils/                # Utility functions
│   │   │   ├── hardware.py       # Hardware detection
│   │   │   ├── image_utils.py    # Image processing
│   │   │   └── video_utils.py    # Video processing
│   │   └── config.py             # Configuration settings
├── tests/                        # Unit tests
├── examples/                     # Example scripts and outputs
│   ├── basic_usage.py           # Video generation examples
│   └── image_usage.py           # Image generation examples
├── docs/                         # Additional documentation
├── outputs/                      # Generated content storage
├── requirements.txt              # Python dependencies
├── main.py                       # Gradio web interface
└── README.md
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Performance Tips

- **Memory Management**: Close unused applications when generating high-resolution videos
- **Storage**: Ensure sufficient disk space for temporary files during generation
- **Batch Processing**: Use the CLI for multiple video generation tasks

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce video resolution or duration
   - Close other applications
   - Consider upgrading hardware

2. **Slow Generation**
   - Ensure GPU drivers are updated
   - Check hardware detection is working correctly
   - Reduce video complexity

3. **Model Download Issues**
   - Check internet connection
   - Verify sufficient disk space
   - Try clearing model cache

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [vLLM](https://docs.vllm.ai/) - Fast LLM inference engine
- [Gradio](https://www.gradio.app/) - Web interface framework
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers
- [Civitai](https://civitai.com/) - Community AI model platform

## Support

For issues and questions:
- Open an [Issue](https://github.com/yourusername/content-creator/issues)
- Check our [FAQ](docs/FAQ.md)
- Join our [Discord](https://discord.gg/your-discord) community 