
"""Unit tests for room matcher module."""

import pytest
from src.room_matcher import RoomMatcher


@pytest.fixture
def room_matcher():
    """Create room matcher instance."""
    return RoomMatcher()


@pytest.fixture
def sample_rooms():
    """Create sample room data."""
    hotelbeds_rooms = [
        {"code": "DBL01", "name": "Double Room", "description": "Standard double bed room"},
        {"code": "SGL01", "name": "Single Room", "description": "Single bed room"},
    ]
    enriched_rooms = [
        {"code": "ER01", "name": "Twin Beds", "description": "Standard double bed room"},
        {"code": "ER02", "name": "Solo", "description": "Single bed room"},
    ]
    return hotelbeds_rooms, enriched_rooms


def test_cosine_similarity(room_matcher):
    """Test cosine similarity calculation."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = room_matcher.cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(1.0)

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = room_matcher.cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(0.0)


def test_cosine_similarity_empty(room_matcher):
    """Test cosine similarity with empty vectors."""
    similarity = room_matcher.cosine_similarity([], [1.0, 0.0])
    assert similarity == 0.0


def test_match_room_no_description(room_matcher, sample_rooms):
    """Test matching room without description."""
    hotelbeds_rooms, enriched_rooms = sample_rooms
    room_no_desc = {"code": "TEST", "name": "Test"}
    match = room_matcher.match_room(room_no_desc, enriched_rooms)
    assert match is None


def test_match_rooms(room_matcher, sample_rooms):
    """Test matching multiple rooms."""
    hotelbeds_rooms, enriched_rooms = sample_rooms
    matches = room_matcher.match_rooms(hotelbeds_rooms, enriched_rooms)
    assert len(matches) == 2
