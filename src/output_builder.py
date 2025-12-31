"""Output builder module for constructing final mapping results."""

from typing import Dict, List, Any, Tuple
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)


class OutputBuilder:
    """Build and structure the final hotel-room mapping output."""

    @staticmethod
    def build_hotel_mapping(
        hotel_match: Tuple[Dict[str, Any], Dict[str, Any] | None],
        room_matches: List[Tuple[Dict[str, Any], Dict[str, Any] | None]],
    ) -> Dict[str, Any]:
        """Build mapping result for a single hotel."""
        hotelbeds_hotel, axisdata_hotel = hotel_match

        hotel_mapping = {
            "hotelbeds_hotel_id": hotelbeds_hotel.get("hotel_id"),
            "hotelbeds_hotel_name": hotelbeds_hotel.get("name"),
            "axisdata_hotel_id": axisdata_hotel.get("hotel_id") if axisdata_hotel else None,
            "axisdata_hotel_name": axisdata_hotel.get("name") if axisdata_hotel else None,
            "matched": axisdata_hotel is not None,
            "rooms": [],
            "confidence": 0.0,
        }

        # Add room mappings
        total_rooms = 0
        matched_rooms = 0

        for hb_room, enriched_room in room_matches:
            total_rooms += 1
            if enriched_room:
                matched_rooms += 1

            room_mapping = {
                "hotelbeds_room_code": hb_room.get("code"),
                "hotelbeds_room_name": hb_room.get("name"),
                "room_class": hb_room.get("room_class"),
                "matched_room": enriched_room.get("code") if enriched_room else None,
                "confidence": 0.8 if enriched_room else 0.0,
            }
            hotel_mapping["rooms"].append(room_mapping)

        # Calculate overall confidence
        if total_rooms > 0:
            hotel_mapping["confidence"] = matched_rooms / total_rooms

        return hotel_mapping

    @staticmethod
    def build_output(
        hotel_matches: List[Tuple[Dict[str, Any], Dict[str, Any] | None]],
        all_room_matches: List[List[Tuple[Dict[str, Any], Dict[str, Any] | None]]],
    ) -> Dict[str, Any]:
        """Build complete output structure."""
        mappings = []

        for i, hotel_match in enumerate(hotel_matches):
            room_matches = all_room_matches[i] if i < len(all_room_matches) else []
            mapping = OutputBuilder.build_hotel_mapping(hotel_match, room_matches)
            mappings.append(mapping)

        output = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_hotels": len(mappings),
                "matched_hotels": sum(1 for m in mappings if m["matched"]),
                "total_rooms": sum(len(m["rooms"]) for m in mappings),
            },
            "mappings": mappings,
        }

        logger.info(
            f"Built output with {output['metadata']['matched_hotels']} "
            f"matched hotels out of {output['metadata']['total_hotels']}"
        )
        return output

    @staticmethod
    def validate_output(output: Dict[str, Any]) -> bool:
        """Validate output structure."""
        required_keys = {"metadata", "mappings"}
        if not all(key in output for key in required_keys):
            logger.error("Invalid output structure")
            return False

        logger.info("Output validation passed")
        return True
