"""Integration tests for the complete pipeline."""

import pytest
import json
from pathlib import Path
from src.data_loader import DataLoader
from src.data_normalizer import DataNormalizer
from src.hotel_matcher import HotelMatcher
from src.room_enricher import RoomEnricher
from src.output_builder import OutputBuilder


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data files for testing."""
    hotelbeds_file = tmp_path / "hotelbeds_hotels.json"
    axisdata_file = tmp_path / "axisdata_hotels.json"
    rooms_file = tmp_path / "rooms.json"

    hotelbeds_data = {
        "hotels": [
            {
                "hotel_id": "HB001",
                "name": "Luxury Hotel Paris",
                "rooms": [
                    {"code": "DBL01", "name": "Double Room", "description": "Modern double room"}
                ],
            }
        ]
    }

    axisdata_data = {
        "hotels": [
            {"hotel_id": "AD001", "name": "Luxury Hotel Paris"}
        ]
    }

    rooms_data = {
        "room_types": [
            {"code": "DBL", "name": "Double Room", "description": "Modern double room"}
        ]
    }

    hotelbeds_file.write_text(json.dumps(hotelbeds_data))
    axisdata_file.write_text(json.dumps(axisdata_data))
    rooms_file.write_text(json.dumps(rooms_data))

    return hotelbeds_file, axisdata_file, rooms_file


def test_integration_pipeline(sample_data):
    """Test the complete integration pipeline."""
    hotelbeds_file, axisdata_file, rooms_file = sample_data

    # Load data
    hotelbeds_hotels = DataLoader.load_hotelbeds_hotels(hotelbeds_file)
    axisdata_hotels = DataLoader.load_axisdata_hotels(axisdata_file)
    room_types = DataLoader.load_room_types(rooms_file)

    assert len(hotelbeds_hotels) == 1
    assert len(axisdata_hotels) == 1
    assert len(room_types) == 1

    # Normalize data
    hotelbeds_hotels = DataNormalizer.clean_hotels(hotelbeds_hotels)
    axisdata_hotels = DataNormalizer.clean_hotels(axisdata_hotels)
    room_types = DataNormalizer.clean_rooms(room_types)

    # Match hotels
    hotel_matches = HotelMatcher.match_hotels(hotelbeds_hotels, axisdata_hotels)
    assert len(hotel_matches) == 1
    assert hotel_matches[0][1] is not None

    # Enrich rooms
    enriched_rooms = RoomEnricher.enrich_rooms(room_types)
    assert len(enriched_rooms) == 1
    assert "room_class" in enriched_rooms[0]

    # Build output
    all_room_matches = [[]]
    output = OutputBuilder.build_output(hotel_matches, all_room_matches)

    assert "metadata" in output
    assert "mappings" in output
    assert output["metadata"]["matched_hotels"] >= 0
