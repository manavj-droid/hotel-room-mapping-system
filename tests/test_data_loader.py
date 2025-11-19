"""Unit tests for data loader module."""

import pytest
from pathlib import Path
from src.data_loader import DataLoader


@pytest.fixture
def sample_json_file(tmp_path):
    """Create a sample JSON file for testing."""
    test_file = tmp_path / "test.json"
    test_file.write_text('{"hotels": [{"hotel_id": "H1", "name": "Test Hotel"}]}')
    return test_file


def test_load_json(sample_json_file):
    """Test loading JSON file."""
    data = DataLoader.load_json(sample_json_file)
    assert "hotels" in data
    assert len(data["hotels"]) == 1


def test_load_hotelbeds_hotels(sample_json_file):
    """Test loading Hotelbeds hotels."""
    hotels = DataLoader.load_hotelbeds_hotels(sample_json_file)
    assert len(hotels) == 1
    assert hotels[0]["hotel_id"] == "H1"


def test_load_json_file_not_found():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        DataLoader.load_json(Path("/nonexistent/file.json"))


def test_load_json_invalid_json(tmp_path):
    """Test error handling for invalid JSON."""
    test_file = tmp_path / "invalid.json"
    test_file.write_text("invalid json content")
    with pytest.raises(Exception):
        DataLoader.load_json(test_file)
