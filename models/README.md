# Models Directory

This directory contains the AI models downloaded automatically by Content Creator.

## Purpose
- **Automatic caching**: Models are downloaded once and stored here to avoid re-downloading
- **Local storage**: Keeps models within the repository for easy access
- **Performance**: Faster startup times after initial download

## Contents
When you use the application, models will be downloaded automatically to subdirectories like:
- `models--black-forest-labs--FLUX.1-dev/` (FLUX models)
- `models--stabilityai--stable-diffusion-xl-base-1.0/` (Stable Diffusion models)
- `models--Heartsync--Flux-NSFW-uncensored/` (LoRA models)

## Size Warning
⚠️ **Large files**: AI models can be several GB each. The `models/` directory is excluded from Git via `.gitignore`.

## Manual Cleanup
To free disk space, you can safely delete model subdirectories. They will be re-downloaded when needed.

```bash
# Remove all cached models
rm -rf models/models--*/

# Remove specific model
rm -rf models/models--black-forest-labs--FLUX.1-dev/
``` 