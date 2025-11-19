
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# ========================================================================
# CONFIGURATION - EDIT THESE PATHS AS NEEDED
# ========================================================================

# Set your file paths here
ROOM_TYPES_PATH = Path("data/inputs/hotelbeds_room_types.json")
HOTELS_PATH = Path("data/inputs/hotelbeds_hotels.json")
OUTPUT_PATH = Path("data/outputs/enriched_hotelbeds_hotels.json")

# Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = logging.INFO

# Set to True to print sample output
SHOW_SAMPLE = True

# ========================================================================


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ========================================================================
# DATA MODELS
# ========================================================================


@dataclass
class RoomTypeInfo:
    """Room type information from catalog."""
    code: str
    code_description: str
    type_category: str
    occupancies: List[Dict[str, Any]]


class RoomEnricherException(Exception):
    """Custom exception for room enrichment errors."""
    pass


# ========================================================================
# ROOM TYPE LOADER
# ========================================================================


class RoomTypeCatalog:
    """Loads and manages the room type catalog."""

    def __init__(self):
        self.catalog: Dict[str, RoomTypeInfo] = {}
        self.stats = {
            'total_loaded': 0,
            'invalid_records': 0,
            'duplicates': 0,
        }

    def load_from_json(self, file_path: Path) -> Tuple[Dict[str, RoomTypeInfo], List[str]]:
        """
        Load room type catalog from JSON file.
        
        Expected format:
        [
            {
                "_id": {"$oid": "..." or "oid": "..."},
                "code": "APT-B1-10",
                "codeDescription": "APARTMENT basement level 1",
                "type": "APT",
                "occupancies": [...]
            },
            ...
        ]
        
        Args:
            file_path: Path to room types JSON file
            
        Returns:
            Tuple of (catalog_dict, error_messages)
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise RoomEnricherException(f"Room types file not found: {file_path}")
        
        logger.info(f"Loading room types catalog from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RoomEnricherException(f"Invalid JSON in room types file: {e}")
        except Exception as e:
            raise RoomEnricherException(f"Error loading room types file: {e}")
        
        if not isinstance(data, list):
            raise RoomEnricherException(f"Room types data must be a list, got {type(data).__name__}")
        
        errors = []
        
        for idx, room_type_raw in enumerate(data):
            try:
                room_type_info = self._parse_room_type(room_type_raw, idx)
                
                # Check for duplicates
                if room_type_info.code in self.catalog:
                    self.stats['duplicates'] += 1
                    logger.warning(
                        f"Duplicate room code '{room_type_info.code}' at index {idx}, "
                        f"overwriting previous entry"
                    )
                
                self.catalog[room_type_info.code] = room_type_info
                self.stats['total_loaded'] += 1
            
            except Exception as e:
                error_msg = f"Failed to parse room type at index {idx}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                self.stats['invalid_records'] += 1
        
        logger.info(
            f"Loaded {len(self.catalog)} room types from catalog. "
            f"Duplicates: {self.stats['duplicates']}, Invalid: {self.stats['invalid_records']}"
        )
        
        return self.catalog, errors

    @staticmethod
    def _parse_room_type(room_type_raw: Any, idx: int) -> RoomTypeInfo:
        """Parse a single room type record."""
        if not isinstance(room_type_raw, dict):
            raise ValueError(f"Room type must be dict, got {type(room_type_raw).__name__}")
        
        # Required fields
        code = room_type_raw.get('code', '').strip()
        if not code:
            raise ValueError("Missing or empty 'code' field")
        
        code_description = room_type_raw.get('codeDescription', '').strip()
        if not code_description:
            raise ValueError("Missing or empty 'codeDescription' field")
        
        type_category = room_type_raw.get('type', '').strip()
        if not type_category:
            raise ValueError("Missing or empty 'type' field")
        
        occupancies = room_type_raw.get('occupancies', [])
        if not isinstance(occupancies, list):
            raise ValueError(f"'occupancies' must be list, got {type(occupancies).__name__}")
        
        return RoomTypeInfo(
            code=code,
            code_description=code_description,
            type_category=type_category,
            occupancies=occupancies,
        )

    def get_room_type(self, code: str) -> Optional[RoomTypeInfo]:
        """Get room type info by code."""
        return self.catalog.get(code)

    def get_code_description(self, code: str) -> Optional[str]:
        """Get code description by code, or return None if not found."""
        room_type = self.catalog.get(code)
        if room_type:
            return room_type.code_description
        return None


# ========================================================================
# HOTELBEDS ROOM ENRICHER
# ========================================================================


class HotelbedRoomEnricher:
    """Enriches Hotelbeds room data with type meanings."""

    def __init__(self, room_type_catalog: RoomTypeCatalog):
        """
        Initialize enricher with room type catalog.
        
        Args:
            room_type_catalog: Loaded RoomTypeCatalog instance
        """
        self.catalog = room_type_catalog
        self.stats = {
            'enriched': 0,
            'missing_roomcode': 0,
            'missing_catalog': 0,
        }

    def enrich_room(self, room: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single Hotelbeds room with type meaning.
        
        Uses roomCode to lookup in catalog and populate codeDescription.
        
        Args:
            room: Raw room dict from Hotelbeds
            
        Returns:
            Enriched room dict with codeDescription populated
        """
        room_copy = room.copy()
        
        # Get roomCode from room (unique per room type)
        room_code = room.get('roomCode', '').strip()
        
        if not room_code:
            self.stats['missing_roomcode'] += 1
            logger.warning(f"Room missing roomCode: {room.get('id', 'unknown')}")
            return room_copy
        
        # Look up in catalog using roomCode
        code_description = self.catalog.get_code_description(room_code)
        
        if code_description:
            room_copy['codeDescription'] = code_description
            self.stats['enriched'] += 1
            logger.debug(f"✓ Enriched roomCode '{room_code}' → '{code_description}'")
        else:
            self.stats['missing_catalog'] += 1
            logger.debug(f"✗ roomCode '{room_code}' not found in catalog")
        
        return room_copy

    def enrich_hotel(self, hotel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich all rooms in a Hotelbeds hotel.
        
        Args:
            hotel: Raw hotel dict from Hotelbeds
            
        Returns:
            Enriched hotel dict
        """
        hotel_copy = hotel.copy()
        enriched_rooms = []
        
        for room in hotel.get('rooms', []):
            enriched_room = self.enrich_room(room)
            enriched_rooms.append(enriched_room)
        
        hotel_copy['rooms'] = enriched_rooms
        
        logger.debug(f"Enriched {len(enriched_rooms)} rooms for hotel {hotel.get('hotelCode', 'unknown')}")
        
        return hotel_copy

    def enrich_hotels(self, hotels: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Enrich all Hotelbeds hotels.
        
        Args:
            hotels: List of raw hotel dicts
            
        Returns:
            Tuple of (enriched_hotels, error_messages)
        """
        enriched_hotels = []
        errors = []
        
        for idx, hotel in enumerate(hotels):
            try:
                enriched_hotel = self.enrich_hotel(hotel)
                enriched_hotels.append(enriched_hotel)
            except Exception as e:
                error_msg = f"Failed to enrich hotel #{idx} ({hotel.get('hotelCode', 'unknown')}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(
            f"Enriched {len(enriched_hotels)}/{len(hotels)} Hotelbeds hotels. "
            f"Rooms enriched: {self.stats['enriched']}, "
            f"Missing catalog: {self.stats['missing_catalog']}, "
            f"Missing roomCode: {self.stats['missing_roomcode']}"
        )
        
        return enriched_hotels, errors


# ========================================================================
# ORCHESTRATOR
# ========================================================================


class RoomEnricher:
    """Main orchestrator for room enrichment process."""

    def __init__(self, room_types_path: Path):
        """
        Initialize enricher with room types catalog.
        
        Args:
            room_types_path: Path to hotelbeds_room_types.json
        """
        self.catalog = RoomTypeCatalog()
        self.catalog.load_from_json(room_types_path)
        self.hotelbeds_enricher = HotelbedRoomEnricher(self.catalog)

    def enrich_hotelbeds_hotels(self, hotels: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Enrich Hotelbeds hotels with room type descriptions.
        
        Args:
            hotels: List of raw Hotelbeds hotel dicts
            
        Returns:
            Tuple of (enriched_hotels, error_messages)
        """
        return self.hotelbeds_enricher.enrich_hotels(hotels)

    def get_catalog_stats(self) -> Dict[str, int]:
        """Get catalog loading statistics."""
        return self.catalog.stats.copy()

    def get_enrichment_stats(self) -> Dict[str, int]:
        """Get enrichment statistics."""
        return self.hotelbeds_enricher.stats.copy()


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================


def save_enriched_hotels(hotels: List[Dict[str, Any]], output_path: Path):
    """
    Save enriched hotels to JSON file.
    
    Args:
        hotels: List of enriched hotel dicts
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hotels, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(hotels)} enriched hotels to {output_path}")


def print_enrichment_sample(enriched_hotels: List[Dict[str, Any]]):
    """Print sample enriched hotel data."""
    print("\n" + "=" * 80)
    print("ROOM ENRICHMENT SAMPLE")
    print("=" * 80)
    
    if enriched_hotels:
        hotel = enriched_hotels[0]
        print(f"\nHotel: {hotel.get('name')} ({hotel.get('hotelCode')})")
        print(f"Place ID: {hotel.get('placeId')}")
        print(f"Total Rooms: {len(hotel.get('rooms', []))}\n")
        
        # Show first 5 rooms
        for idx, room in enumerate(hotel.get('rooms', [])[:5]):
            room_code = room.get('roomCode', 'N/A')
            code_desc = room.get('codeDescription', '❌ NOT ENRICHED')
            room_name = room.get('name', 'N/A')
            
            # Add visual indicator
            status = "✓" if code_desc != '❌ NOT ENRICHED' else "✗"
            
            print(f"Room #{idx + 1}: {status}")
            print(f"  roomCode: {room_code}")
            print(f"  roomName: {room_name}")
            print(f"  codeDescription: {code_desc}")
            if room.get('occupancies'):
                occ = room['occupancies']
                print(f"  occupancy: min={occ.get('minOccupancy')}, max={occ.get('maxOccupancy')}")
            print()
    
    print("=" * 80)


# ========================================================================
# MAIN EXECUTION
# ========================================================================


def main():
    """Main entry point - no CLI arguments required."""
    
    try:
        # Step 1: Load raw Hotelbeds hotels from JSON file
        logger.info("=" * 80)
        logger.info("STEP 1: Loading raw Hotelbeds hotels")
        logger.info("=" * 80)
        
        if not HOTELS_PATH.exists():
            logger.error(f"Hotelbeds hotels file not found: {HOTELS_PATH}")
            logger.error(f"Please check the path at the top of this script")
            return 1
        
        try:
            with open(HOTELS_PATH, 'r', encoding='utf-8') as f:
                hotelbeds_raw = json.load(f)
            
            if not isinstance(hotelbeds_raw, list):
                logger.error(f"Expected list, got {type(hotelbeds_raw).__name__}")
                return 1
            
            logger.info(f"Loaded {len(hotelbeds_raw)} Hotelbeds hotels from {HOTELS_PATH}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in hotelbeds file: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error loading hotelbeds hotels: {e}")
            return 1
        
        # Step 2: Enrich Hotelbeds rooms with type descriptions
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Enriching Hotelbeds rooms with type catalog")
        logger.info("=" * 80)
        
        try:
            enricher = RoomEnricher(room_types_path=ROOM_TYPES_PATH)
            enriched_hotelbeds, enrich_errors = enricher.enrich_hotelbeds_hotels(hotelbeds_raw)
            
            # Print statistics
            print(f"\n--- Catalog Statistics ---")
            catalog_stats = enricher.get_catalog_stats()
            for key, value in catalog_stats.items():
                print(f"  {key}: {value}")
            
            print(f"\n--- Enrichment Statistics ---")
            enrich_stats = enricher.get_enrichment_stats()
            for key, value in enrich_stats.items():
                print(f"  {key}: {value}")
            
            total_attempts = (enrich_stats['enriched'] + enrich_stats['missing_catalog'] + 
                            enrich_stats['missing_roomcode'])
            enrichment_rate = (enrich_stats['enriched'] / total_attempts * 100) if total_attempts > 0 else 0
            print(f"  Enrichment Rate: {enrichment_rate:.1f}%")
            
            # Print sample if enabled
            if SHOW_SAMPLE:
                print_enrichment_sample(enriched_hotelbeds)
            
            # Save enriched hotels
            save_enriched_hotels(enriched_hotelbeds, OUTPUT_PATH)
            
            # Print errors if any
            if enrich_errors:
                print(f"\n--- Enrichment Errors ({len(enrich_errors)}) ---")
                for error in enrich_errors[:5]:  # Show first 5 errors
                    print(f"  ⚠ {error}")
                if len(enrich_errors) > 5:
                    print(f"  ... and {len(enrich_errors) - 5} more errors")
            
            logger.info("\n" + "=" * 80)
            logger.info("ENRICHMENT COMPLETE")
            logger.info("=" * 80)
            logger.info(f"✓ Output saved to: {OUTPUT_PATH}")
            
            return 0
        
        except RoomEnricherException as e:
            logger.error(f"CRITICAL: {e}")
            return 1
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR: {e}", exc_info=True)
            return 1
    
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())