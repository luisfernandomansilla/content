"""
Basic tests for Content Creator
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.content_creator import VideoGenerator, Config, HardwareDetector
from src.content_creator.config import config

class TestConfig:
    """Test configuration"""
    
    def test_config_initialization(self):
        """Test that config initializes properly"""
        assert config.DEFAULT_MODEL in config.SUPPORTED_MODELS
        assert config.DEFAULT_STYLE in config.VIDEO_STYLES
        assert config.DEFAULT_RESOLUTION in config.RESOLUTIONS
        assert config.DEFAULT_OUTPUT_FORMAT in config.OUTPUT_FORMATS
    
    def test_resolution_mapping(self):
        """Test resolution mapping"""
        width, height = config.get_resolution("720p")
        assert width == 1280
        assert height == 720


class TestHardwareDetector:
    """Test hardware detection"""
    
    def test_hardware_detector_initialization(self):
        """Test hardware detector initializes"""
        detector = HardwareDetector()
        assert detector.hardware_type in ["apple_silicon", "nvidia_gpu", "amd_gpu", "cpu"]
        assert detector.system_info is not None
    
    def test_device_info(self):
        """Test device info retrieval"""
        detector = HardwareDetector()
        info = detector.get_device_info()
        
        assert "hardware_type" in info
        assert "system_info" in info
        assert "optimizations" in info


class TestVideoGenerator:
    """Test video generator"""
    
    def test_generator_initialization(self):
        """Test generator initializes"""
        generator = VideoGenerator()
        assert generator.model_manager is not None
        assert generator.hardware_detector is not None
    
    def test_get_available_models(self):
        """Test getting available models"""
        generator = VideoGenerator()
        models = generator.get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert config.DEFAULT_MODEL in models
    
    def test_input_validation(self):
        """Test input validation"""
        generator = VideoGenerator()
        
        # Test empty prompt
        assert not generator._validate_inputs("", "AnimateDiff", "Realistic", 10, "720p")
        
        # Test invalid duration
        assert not generator._validate_inputs("test", "AnimateDiff", "Realistic", 0, "720p")
        assert not generator._validate_inputs("test", "AnimateDiff", "Realistic", 1000, "720p")
        
        # Test invalid style
        assert not generator._validate_inputs("test", "AnimateDiff", "InvalidStyle", 10, "720p")
        
        # Test invalid resolution
        assert not generator._validate_inputs("test", "AnimateDiff", "Realistic", 10, "InvalidRes")
        
        # Test valid inputs
        assert generator._validate_inputs("test prompt", "AnimateDiff", "Realistic", 10, "720p")


if __name__ == "__main__":
    pytest.main([__file__]) 