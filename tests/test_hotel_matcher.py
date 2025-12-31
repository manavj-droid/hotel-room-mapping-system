"""Unit tests for hotel matcher module."""

import pytest
from src.hotel_matcher import HotelMatcher


@pytest.fixture
def sample_hotels():
    """Create sample hotel data."""
    hotelbeds = [
        {
            "hotel_id": "HB1",
            "name": "Grand Hotel Paris",
            "name_normalized": "grand hotel paris",
        }
    ]
    axisdata = [
        {
            "hotel_id": "AD1",
            "name": "Grand Hotel Paris",
            "name_normalized": "grand hotel paris",
        },
        {
            "hotel_id": "AD2",
            "name": "Small Inn",
            "name_normalized": "small inn",
        },
    ]
    return hotelbeds, axisdata


def test_calculate_similarity():
    """Test similarity calculation."""
    score = HotelMatcher.calculate_similarity("test", "test")
    assert score == 1.0

    score = HotelMatcher.calculate_similarity("test", "best")
    assert 0 < score < 1


def test_match_by_name(sample_hotels):
    """Test hotel matching by name."""
    hotelbeds, axisdata = sample_hotels
    match = HotelMatcher.match_by_name(hotelbeds[0], axisdata, threshold=0.8)
    assert match is not None
    assert match["hotel_id"] == "AD1"


def test_match_by_name_no_match(sample_hotels):
    """Test when no suitable match is found."""
    hotelbeds, axisdata = sample_hotels
    no_match_hotel = {
        "hotel_id": "HB2",
        "name": "Completely Different",
        "name_normalized": "completely different",
    }
    match = HotelMatcher.match_by_name(no_match_hotel, axisdata, threshold=0.9)
    assert match is None


def test_match_hotels(sample_hotels):
    """Test matching multiple hotels."""
    hotelbeds, axisdata = sample_hotels
    matches = HotelMatcher.match_hotels(hotelbeds, axisdata)
    assert len(matches) == 1
    assert matches[0][1] is not None
