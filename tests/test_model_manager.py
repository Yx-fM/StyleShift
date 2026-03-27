"""Tests for model management utilities."""

import pytest
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

from style_shift.utils.model_manager import ModelManager


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    return str(tmp_path / "cache")


@pytest.fixture
def model_manager(temp_cache_dir):
    """Create a ModelManager instance with temporary cache."""
    return ModelManager(cache_dir=temp_cache_dir)


@pytest.fixture
def mock_response():
    """Create a mock HTTP response."""
    response = MagicMock()
    response.iter_content.return_value = [b"fake_model_data"]
    response.raise_for_status = MagicMock()
    return response


class TestModelManagerInit:
    """Tests for ModelManager initialization."""
    
    def test_creates_default_cache_dir(self):
        """Test that default cache directory is created."""
        mm = ModelManager()
        assert mm.cache_dir.exists()
    
    def test_creates_custom_cache_dir(self, temp_cache_dir):
        """Test that custom cache directory is created."""
        mm = ModelManager(cache_dir=temp_cache_dir)
        assert Path(temp_cache_dir).exists()
    
    def test_has_default_models(self, model_manager):
        """Test that default models are registered."""
        models = model_manager.list_available_models()
        assert len(models) > 0
        assert 'decoder_anime' in models


class TestGetModelPath:
    """Tests for get_model_path()."""
    
    def test_returns_correct_path(self, model_manager):
        """Test that get_model_path returns correct path."""
        path = model_manager.get_model_path('decoder_anime')
        assert str(path).endswith('decoder_anime.pth')


class TestIsModelCached:
    """Tests for is_model_cached()."""
    
    def test_returns_false_for_uncached_model(self, model_manager):
        """Test that uncached model returns False."""
        assert not model_manager.is_model_cached('decoder_anime')
    
    def test_returns_true_for_cached_model(self, model_manager, tmp_path):
        """Test that cached model returns True."""
        # Create a fake model file
        model_file = model_manager.get_model_path('decoder_anime')
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()
        
        assert model_manager.is_model_cached('decoder_anime')


class TestDownloadModel:
    """Tests for download_model()."""
    
    def test_downloads_model(self, model_manager, mock_response, tmp_path):
        """Test that download_model downloads and saves model."""
        model_name = 'decoder_anime'
        
        # Update expected hash to match mock data
        import hashlib
        expected_hash = hashlib.sha256(b"fake_model_data").hexdigest()
        model_manager.models_info[model_name]['hash'] = expected_hash
        
        with patch('requests.get', return_value=mock_response):
            model_path = model_manager.download_model(model_name)
        
        assert model_path.exists()
    
    def test_returns_cached_path_if_exists(self, model_manager):
        """Test that download_model returns cached path if model exists."""
        model_name = 'decoder_anime'
        model_file = model_manager.get_model_path(model_name)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()
        
        # Should not make HTTP request
        with patch('requests.get') as mock_get:
            model_path = model_manager.download_model(model_name)
            mock_get.assert_not_called()
        
        assert model_path == model_file
    
    def test_force_redownloads(self, model_manager, mock_response):
        """Test that force=True re-downloads model."""
        model_name = 'decoder_anime'
        model_file = model_manager.get_model_path(model_name)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()
        
        # Update expected hash
        import hashlib
        expected_hash = hashlib.sha256(b"fake_model_data").hexdigest()
        model_manager.models_info[model_name]['hash'] = expected_hash
        
        with patch('requests.get', return_value=mock_response):
            model_manager.download_model(model_name, force=True)
        
        # File should be overwritten
        assert model_file.exists()
    
    def test_invalid_model_name_raises_error(self, model_manager):
        """Test that invalid model name raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            model_manager.download_model('nonexistent_model')
    
    def test_download_failure_cleans_up(self, model_manager, tmp_path):
        """Test that failed download cleans up partial files."""
        model_name = 'decoder_anime'
        model_path = model_manager.get_model_path(model_name)
        
        # Simulate download failure with requests.RequestException
        import requests
        
        def raise_request_exception(*args, **kwargs):
            raise requests.exceptions.RequestException("Network error")
        
        with patch('requests.get', side_effect=raise_request_exception):
            with pytest.raises(RuntimeError, match="Network error"):
                model_manager.download_model(model_name)
        
        # Partial file should be cleaned up
        assert not model_path.exists()
    
    def test_hash_verification_failure_removes_file(self, model_manager, mock_response):
        """Test that hash verification failure removes corrupted file."""
        model_name = 'decoder_anime'
        
        # Set wrong expected hash
        model_manager.models_info[model_name]['hash'] = 'wrong_hash'
        
        with patch('requests.get', return_value=mock_response):
            with pytest.raises(RuntimeError, match="Hash verification failed"):
                model_manager.download_model(model_name, verify_hash=True)
        
        # File should be removed
        assert not model_manager.get_model_path(model_name).exists()


class TestComputeHash:
    """Tests for _compute_hash()."""
    
    def test_computes_sha256(self, model_manager, tmp_path):
        """Test that _compute_hash computes SHA256 correctly."""
        test_file = tmp_path / "test.txt"
        test_data = b"test data"
        test_file.write_bytes(test_data)
        
        expected_hash = hashlib.sha256(test_data).hexdigest()
        actual_hash = model_manager._compute_hash(test_file)
        
        assert actual_hash == expected_hash


class TestListAvailableModels:
    """Tests for list_available_models()."""
    
    def test_returns_list(self, model_manager):
        """Test that list_available_models returns a list."""
        models = model_manager.list_available_models()
        assert isinstance(models, list)
    
    def test_contains_default_models(self, model_manager):
        """Test that list contains default models."""
        models = model_manager.list_available_models()
        assert 'decoder_anime' in models
        assert 'decoder_vangogh' in models
        assert 'decoder_monet' in models


class TestGetModelInfo:
    """Tests for get_model_info()."""
    
    def test_returns_dict(self, model_manager):
        """Test that get_model_info returns a dictionary."""
        info = model_manager.get_model_info('decoder_anime')
        assert isinstance(info, dict)
    
    def test_contains_metadata(self, model_manager):
        """Test that info contains expected metadata."""
        info = model_manager.get_model_info('decoder_anime')
        assert 'url' in info
        assert 'hash' in info
        assert 'description' in info
    
    def test_invalid_model_raises_error(self, model_manager):
        """Test that invalid model name raises KeyError."""
        with pytest.raises(KeyError):
            model_manager.get_model_info('nonexistent')


class TestClearCache:
    """Tests for clear_cache()."""
    
    def test_removes_all_models(self, model_manager, tmp_path):
        """Test that clear_cache removes all model files."""
        # Create fake model files
        for model_name in ['decoder_anime', 'decoder_vangogh']:
            model_file = model_manager.get_model_path(model_name)
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.touch()
        
        model_manager.clear_cache()
        
        # All files should be removed
        assert len(list(model_manager.cache_dir.glob("*.pth"))) == 0


class TestGetCacheSize:
    """Tests for get_cache_size()."""
    
    def test_returns_zero_for_empty_cache(self, model_manager):
        """Test that get_cache_size returns 0 for empty cache."""
        size = model_manager.get_cache_size()
        assert size == 0
    
    def test_returns_correct_size(self, model_manager, tmp_path):
        """Test that get_cache_size returns correct size."""
        # Create fake model file with known size
        model_file = model_manager.get_model_path('decoder_anime')
        model_file.parent.mkdir(parents=True, exist_ok=True)
        test_data = b"x" * 1024  # 1KB
        model_file.write_bytes(test_data)
        
        size = model_manager.get_cache_size()
        assert size == 1024


class TestIntegration:
    """Integration tests for ModelManager."""
    
    def test_download_and_verify_workflow(self, model_manager, mock_response, tmp_path):
        """Test complete download and verify workflow."""
        model_name = 'decoder_anime'
        
        # Update expected hash
        import hashlib
        expected_hash = hashlib.sha256(b"fake_model_data").hexdigest()
        model_manager.models_info[model_name]['hash'] = expected_hash
        
        # Download
        with patch('requests.get', return_value=mock_response):
            path = model_manager.download_model(model_name)
        
        # Verify file exists
        assert path.exists()
        
        # Verify is_cached returns True
        assert model_manager.is_model_cached(model_name)
        
        # Verify get_cache_size is correct
        size = model_manager.get_cache_size()
        assert size > 0
        
        # Clear cache
        model_manager.clear_cache()
        
        # Verify cache is empty
        assert not model_manager.is_model_cached(model_name)
        assert model_manager.get_cache_size() == 0
