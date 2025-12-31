import unicodedata
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from data_loader import load_all_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ========================================================================
# DATA MODELS FOR NORMALIZED DATA
# ========================================================================

@dataclass
class NormalizedOccupancy:
    """
    Normalized occupancy representation.
    Standardized format for both Hotelbeds and AxisData.
    """
    min_occupancy: int = 0
    max_occupancy: int = 0
    min_adults: int = 0
    max_adults: int = 0
    min_children: int = 0
    max_children: int = 0
    max_child_age: Optional[int] = None
    min_infants: Optional[int] = None
    max_infants: Optional[int] = None
    max_infant_age: Optional[int] = None
    standard_occupancy: Optional[int] = None

@dataclass
class NormalizedRoom:
    """
    Normalized room representation.
    
    For semantic matching, we need:
    - For Hotelbeds: room_code, room_name, codeDescription (from enrichment)
    - For AxisData: room_code, description, codeDescription (if available)
    """
    room_id: str
    room_code: str
    room_name: str
    room_description: str
    code_description: Optional[str] = None
    occupancy: Optional[NormalizedOccupancy] = None  # ✓ UPDATED: Now NormalizedOccupancy object
    provider_code: Optional[str] = None
    normalized_description: Optional[str] = None
    normalized_code_description: Optional[str] = None

@dataclass
class NormalizedHotel:
    """Normalized hotel representation."""
    hotel_id: str
    place_id: str
    hotel_code: str
    hotel_name: str
    normalized_hotel_name: str
    rooms: List[NormalizedRoom] = field(default_factory=list)
    source: str = "unknown"

# ========================================================================
# STRING NORMALIZATION UTILITIES
# ========================================================================

class StringNormalizer:
    """Utility class for string normalization operations."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode text using NFKD decomposition.
        Converts accented characters to their base form.
        
        Example: 'Apartamëntos' -> 'Apartamientos' (ë -> e)
        """
        if not isinstance(text, str):
            return ""
        normalized = unicodedata.normalize('NFKD', text)
        normalized = ''.join(
            char for char in normalized 
            if unicodedata.category(char) != 'Mn'
        )
        return normalized

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace: remove leading/trailing, collapse multiple spaces.
        """
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def normalize_case(text: str) -> str:
        """Convert to lowercase for consistent comparison."""
        if not isinstance(text, str):
            return ""
        return text.lower()

    @staticmethod
    def remove_special_characters(text: str, keep_chars: str = "") -> str:
        """Remove special characters, keep alphanumeric + spaces + custom chars."""
        if not isinstance(text, str):
            return ""
        pattern = r'[^a-z0-9\s' + re.escape(keep_chars) + ']'
        text = re.sub(pattern, '', text)
        return text

    @staticmethod
    def normalize_full(text: str, remove_special: bool = False) -> str:
        """
        Apply full normalization: unicode → whitespace → lowercase → optionally remove special chars.
        """
        text = StringNormalizer.normalize_unicode(text)
        text = StringNormalizer.normalize_whitespace(text)
        text = StringNormalizer.normalize_case(text)
        if remove_special:
            text = StringNormalizer.remove_special_characters(text)
        return text

    @staticmethod
    def extract_key_terms(text: str) -> List[str]:
        """
        Extract key terms from text (words > 3 characters, excluding common words).
        Useful for understanding room types without full semantic analysis.
        """
        stop_words = {
            'and', 'or', 'the', 'a', 'an', 'is', 'are', 'with', 'to', 'of', 
            'for', 'in', 'on', 'at', 'by', 'from', 'as', 'has', 'have', 'be',
            'bed', 'room', 'apartment', 'suite', 'view', 'balcony'
        }
        text = StringNormalizer.normalize_full(text)
        words = re.findall(r'\w{4,}', text)
        key_terms = [w for w in words if w not in stop_words]
        return list(set(key_terms))

# ========================================================================
# OCCUPANCY NORMALIZATION
# ========================================================================

class OccupancyNormalizer:
    """Normalizes occupancy data from different providers."""

    @staticmethod
    def normalize_hotelbeds_occupancy(occupancy_data: Optional[Dict[str, Any]]) -> Optional[NormalizedOccupancy]:
        """
        Normalize Hotelbeds occupancy data to standard format.
        
        Hotelbeds format (integers):
        {
            "minOccupancy": 1,
            "maxOccupancy": 4,
            "minAdult": 1,
            "maxAdults": 4,
            "maxChildren": 2
        }
        
        Args:
            occupancy_data: Raw occupancy dict from Hotelbeds
            
        Returns:
            NormalizedOccupancy object or None if no data
        """
        if not occupancy_data or not isinstance(occupancy_data, dict):
            return None
        
        return NormalizedOccupancy(
            min_occupancy=occupancy_data.get('minOccupancy', 0),
            max_occupancy=occupancy_data.get('maxOccupancy', 0),
            min_adults=occupancy_data.get('minAdult', 0),
            max_adults=occupancy_data.get('maxAdults', 0),
            min_children=0,  # Hotelbeds doesn't provide min_children
            max_children=occupancy_data.get('maxChildren', 0),
            max_child_age=None,  # Not provided by Hotelbeds
            min_infants=None,
            max_infants=None,
            max_infant_age=None,
            standard_occupancy=None,
        )

    @staticmethod
    def normalize_axisdata_occupancy(occupancy_data: Optional[Dict[str, str]]) -> Optional[NormalizedOccupancy]:
        """
        Normalize AxisData occupancy data to standard format.
        
        AxisData format (strings):
        {
            "MaxOccupancy": "4",
            "MinOccupancy": "1",
            "MinAdultOccupancy": "1",
            "MaxAdultOccupancy": "4",
            "MinChildOccupancy": "0",
            "MaxChildOccupancy": "2",
            "MaxChildAge": "2",
            "InfantOccupancy": "1",
            "MinInfantOccupancy": "0",
            "MaxInfantOccupancy": "1",
            "StandardOccupancy": "1"
        }
        
        Args:
            occupancy_data: Raw occupancy dict from AxisData (values are strings)
            
        Returns:
            NormalizedOccupancy object or None if no data
        """
        if not occupancy_data or not isinstance(occupancy_data, dict):
            return None
        
        def safe_int(value: Any, default: int = 0) -> int:
            """Safely convert string to int."""
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        return NormalizedOccupancy(
            min_occupancy=safe_int(occupancy_data.get('MinOccupancy')),
            max_occupancy=safe_int(occupancy_data.get('MaxOccupancy')),
            min_adults=safe_int(occupancy_data.get('MinAdultOccupancy')),
            max_adults=safe_int(occupancy_data.get('MaxAdultOccupancy')),
            min_children=safe_int(occupancy_data.get('MinChildOccupancy')),
            max_children=safe_int(occupancy_data.get('MaxChildOccupancy')),
            max_child_age=safe_int(occupancy_data.get('MaxChildAge')) if occupancy_data.get('MaxChildAge') else None,
            min_infants=safe_int(occupancy_data.get('MinInfantOccupancy')) if occupancy_data.get('MinInfantOccupancy') else None,
            max_infants=safe_int(occupancy_data.get('MaxInfantOccupancy')) if occupancy_data.get('MaxInfantOccupancy') else None,
            max_infant_age=safe_int(occupancy_data.get('MaxInfantAge')) if occupancy_data.get('MaxInfantAge') else None,
            standard_occupancy=safe_int(occupancy_data.get('StandardOccupancy')) if occupancy_data.get('StandardOccupancy') else None,
        )

# ========================================================================
# ROOM NORMALIZATION
# ========================================================================

class RoomNormalizer:
    """Normalizes room data from different providers."""

    @staticmethod
    def normalize_hotelbeds_room(room_data: Dict[str, Any], hotel_code: str) -> NormalizedRoom:
        """
        Normalize a Hotelbeds room to standard format.
        
        Hotelbeds includes codeDescription from the enrichment step.
        
        Args:
            room_data: Raw room data from Hotelbeds (enriched with codeDescription)
            hotel_code: Parent hotel code (for context)
            
        Returns:
            NormalizedRoom object
        """
        room_id = room_data.get('id', '')
        room_name = room_data.get('name', 'Unknown Room')
        
        # Extract roomCode from input data (strict - no fallback)
        room_code = room_data.get('roomCode', '')
        
        # Extract provider code from providers (for provider_code field only)
        provider_code = ''
        providers = room_data.get('providers', [])
        if providers and isinstance(providers, list) and len(providers) > 0 and isinstance(providers[0], dict):
            provider_code = providers[0].get('code', '')
        
        # Extract codeDescription (populated by room enricher)
        code_description = room_data.get('codeDescription', '')
        
        # For Hotelbeds, room_name is the semantic information (+ codeDescription if available)
        room_description = room_name
        normalized_description = StringNormalizer.normalize_full(room_description, remove_special=True)
        
        # Normalize codeDescription if present
        normalized_code_description = None
        if code_description:
            normalized_code_description = StringNormalizer.normalize_full(code_description, remove_special=True)
        
        # ✓ NEW: Normalize occupancy data
        occupancy = OccupancyNormalizer.normalize_hotelbeds_occupancy(
            room_data.get('occupancies')
        )
        
        normalized_room = NormalizedRoom(
            room_id=room_id,
            room_code=room_code,
            room_name=room_name,
            room_description=room_description,
            code_description=code_description,
            provider_code=provider_code,
            normalized_description=normalized_description,
            normalized_code_description=normalized_code_description,
            occupancy=occupancy,  # ✓ NEW
        )
        
        logger.debug(
            f"Normalized Hotelbeds room: room_code={room_code}, "
            f"provider_code={provider_code}, name={room_name}, "
            f"code_desc={code_description}, normalized_desc={normalized_description}, "
            f"occupancy={occupancy}"
        )
        
        return normalized_room

    @staticmethod
    def normalize_axisdata_room(room_data: Dict[str, Any], hotel_code: str) -> NormalizedRoom:
        """
        Normalize an AxisData room to standard format.
        
        AxisData provides:
        - room.name: Room type name
        - room.code: Room code
        - room.description: Full descriptive text
        - room.codeDescription: Room type description (if available)
        - room.occupancies: Occupancy constraints (string values)
        - providers.code: Provider-specific code
        
        For semantic matching, we use the description field + codeDescription.
        
        Args:
            room_data: Raw room data from AxisData
            hotel_code: Parent hotel code (for context)
            
        Returns:
            NormalizedRoom object
        """
        room_id = room_data.get('id', '')
        room_name = room_data.get('name', 'Unknown Room')
        room_code = room_data.get('code', '')
        room_description = room_data.get('description', room_name or 'Unknown Room')
        
        # Extract provider-specific code
        provider_code = ''
        providers = room_data.get('providers', [])
        if providers and isinstance(providers, list) and len(providers) > 0 and isinstance(providers[0], dict):
            provider_code = providers[0].get('code', '')
        
        # Extract codeDescription (if available in AxisData)
        code_description = room_data.get('codeDescription', '')
        
        # Normalize description for semantic matching
        normalized_description = StringNormalizer.normalize_full(room_description, remove_special=True)
        
        # Normalize codeDescription if present
        normalized_code_description = None
        if code_description:
            normalized_code_description = StringNormalizer.normalize_full(code_description, remove_special=True)
        
        # ✓ NEW: Normalize occupancy data
        occupancy = OccupancyNormalizer.normalize_axisdata_occupancy(
            room_data.get('occupancies')
        )
        
        normalized_room = NormalizedRoom(
            room_id=room_id,
            room_code=room_code,
            room_name=room_name or room_description,
            room_description=room_description,
            code_description=code_description,
            provider_code=provider_code,
            normalized_description=normalized_description,
            normalized_code_description=normalized_code_description,
            occupancy=occupancy,  # ✓ NEW
        )
        
        logger.debug(
            f"Normalized AxisData room: code={room_code}, "
            f"name={room_name}, provider_code={provider_code}, "
            f"code_desc={code_description}, "
            f"normalized_desc={normalized_description}, "
            f"occupancy={occupancy}"
        )
        
        return normalized_room

# ========================================================================
# HOTEL NORMALIZATION
# ========================================================================

class HotelNormalizer:
    """Normalizes hotel data from different providers."""

    @staticmethod
    def normalize_hotelbeds_hotel(hotel_data: Dict[str, Any]) -> NormalizedHotel:
        """
        Normalize a Hotelbeds hotel to standard format.
        
        Args:
            hotel_data: Raw hotel data from Hotelbeds (with enriched rooms)
            
        Returns:
            NormalizedHotel object
        """
        # Extract IDs
        hotel_id = hotel_data.get('_id', {}).get('oid', '')
        place_id = hotel_data.get('placeId', '')
        hotel_code = hotel_data.get('hotelCode', '')
        hotel_name = hotel_data.get('name', 'Unknown Hotel')
        
        # Normalize hotel name
        normalized_hotel_name = StringNormalizer.normalize_full(hotel_name, remove_special=False)
        
        # Normalize rooms
        normalized_rooms = []
        rooms_with_occupancy = 0
        for room_data in hotel_data.get('rooms', []):
            try:
                normalized_room = RoomNormalizer.normalize_hotelbeds_room(room_data, hotel_code)
                normalized_rooms.append(normalized_room)
                if normalized_room.occupancy:
                    rooms_with_occupancy += 1
            except Exception as e:
                logger.warning(f"Failed to normalize Hotelbeds room in {hotel_code}: {e}")
                continue
        
        normalized_hotel = NormalizedHotel(
            hotel_id=hotel_id,
            place_id=place_id,
            hotel_code=hotel_code,
            hotel_name=hotel_name,
            normalized_hotel_name=normalized_hotel_name,
            rooms=normalized_rooms,
            source='hotelbeds',
        )
        
        logger.info(
            f"Normalized Hotelbeds hotel: {hotel_code} ({place_id}), "
            f"rooms={len(normalized_rooms)}, rooms_with_occupancy={rooms_with_occupancy}"
        )
        
        return normalized_hotel

    @staticmethod
    def normalize_axisdata_hotel(hotel_data: Dict[str, Any]) -> NormalizedHotel:
        """
        Normalize an AxisData hotel to standard format.
        
        Args:
            hotel_data: Raw hotel data from AxisData
            
        Returns:
            NormalizedHotel object
        """
        # Extract IDs
        hotel_id = hotel_data.get('_id', {}).get('oid', '')
        place_id = hotel_data.get('placeId', '')
        hotel_code = hotel_data.get('hotelCode', '')
        hotel_name = hotel_data.get('name', 'Unknown Hotel')
        
        # Normalize hotel name
        normalized_hotel_name = StringNormalizer.normalize_full(hotel_name, remove_special=False)
        
        # Normalize rooms
        normalized_rooms = []
        rooms_with_occupancy = 0
        for room_data in hotel_data.get('rooms', []):
            try:
                normalized_room = RoomNormalizer.normalize_axisdata_room(room_data, hotel_code)
                normalized_rooms.append(normalized_room)
                if normalized_room.occupancy:
                    rooms_with_occupancy += 1
            except Exception as e:
                logger.warning(f"Failed to normalize AxisData room in {hotel_code}: {e}")
                continue
        
        normalized_hotel = NormalizedHotel(
            hotel_id=hotel_id,
            place_id=place_id,
            hotel_code=hotel_code,
            hotel_name=hotel_name,
            normalized_hotel_name=normalized_hotel_name,
            rooms=normalized_rooms,
            source='axisdata',
        )
        
        logger.info(
            f"Normalized AxisData hotel: {hotel_code} ({place_id}), "
            f"rooms={len(normalized_rooms)}, rooms_with_occupancy={rooms_with_occupancy}"
        )
        
        return normalized_hotel

# ========================================================================
# DATA NORMALIZER (ORCHESTRATOR)
# ========================================================================

class DataNormalizer:
    """
    Orchestrates normalization of raw hotel and room data.
    This is the main class used by the pipeline.
    """

    def __init__(self):
        """Initialize data normalizer."""
        self.string_normalizer = StringNormalizer()
        self.room_normalizer = RoomNormalizer()
        self.hotel_normalizer = HotelNormalizer()
        self.occupancy_normalizer = OccupancyNormalizer()

    def normalize_hotelbeds_hotels(
        self,
        raw_hotels: List[Dict[str, Any]],
    ) -> Tuple[List[NormalizedHotel], List[str]]:
        """
        Normalize a list of Hotelbeds hotels.
        
        Args:
            raw_hotels: List of raw hotel dicts from loader (enriched with codeDescription)
            
        Returns:
            Tuple of (normalized_hotels, error_messages)
        """
        normalized_hotels = []
        errors = []
        total_rooms = 0
        total_rooms_with_occupancy = 0
        
        for idx, raw_hotel in enumerate(raw_hotels):
            try:
                normalized_hotel = self.hotel_normalizer.normalize_hotelbeds_hotel(raw_hotel)
                
                if not normalized_hotel.rooms:
                    errors.append(
                        f"Hotelbeds Hotel #{idx} ({raw_hotel.get('hotelCode')}): No rooms after normalization"
                    )
                
                normalized_hotels.append(normalized_hotel)
                total_rooms += len(normalized_hotel.rooms)
                total_rooms_with_occupancy += sum(
                    1 for room in normalized_hotel.rooms if room.occupancy
                )
            
            except Exception as e:
                error_msg = f"Failed to normalize Hotelbeds hotel #{idx}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(
            f"Normalized {len(normalized_hotels)}/{len(raw_hotels)} Hotelbeds hotels. "
            f"Total rooms: {total_rooms}. Rooms with occupancy: {total_rooms_with_occupancy}. "
            f"Errors: {len(errors)}"
        )
        
        return normalized_hotels, errors

    def normalize_axisdata_hotels(
        self,
        raw_hotels: List[Dict[str, Any]],
    ) -> Tuple[List[NormalizedHotel], List[str]]:
        """
        Normalize a list of AxisData hotels.
        
        Args:
            raw_hotels: List of raw hotel dicts from loader
            
        Returns:
            Tuple of (normalized_hotels, error_messages)
        """
        normalized_hotels = []
        errors = []
        total_rooms = 0
        total_rooms_with_occupancy = 0
        
        for idx, raw_hotel in enumerate(raw_hotels):
            try:
                normalized_hotel = self.hotel_normalizer.normalize_axisdata_hotel(raw_hotel)
                
                if not normalized_hotel.rooms:
                    errors.append(
                        f"AxisData Hotel #{idx} ({raw_hotel.get('hotelCode')}): No rooms after normalization"
                    )
                
                normalized_hotels.append(normalized_hotel)
                total_rooms += len(normalized_hotel.rooms)
                total_rooms_with_occupancy += sum(
                    1 for room in normalized_hotel.rooms if room.occupancy
                )
            
            except Exception as e:
                error_msg = f"Failed to normalize AxisData hotel #{idx}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        logger.info(
            f"Normalized {len(normalized_hotels)}/{len(raw_hotels)} AxisData hotels. "
            f"Total rooms: {total_rooms}. Rooms with occupancy: {total_rooms_with_occupancy}. "
            f"Errors: {len(errors)}"
        )
        
        return normalized_hotels, errors

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def save_hotelbeds_normalized_data(
    hotelbeds: List[NormalizedHotel],
    output_path: str = "hotelbed_norm_data.json"
) -> None:
    """
    Save normalized Hotelbeds data to JSON file.
    
    Args:
        hotelbeds: List of normalized Hotelbeds hotels
        output_path: Path where JSON file will be saved
    """
    hotelbeds_data = [asdict(hotel) for hotel in hotelbeds]
    
    # Calculate statistics
    total_rooms = sum(len(hotel['rooms']) for hotel in hotelbeds_data)
    total_rooms_with_occupancy = sum(
        sum(1 for room in hotel['rooms'] if room['occupancy'])
        for hotel in hotelbeds_data
    )
    
    data = {
        "hotels": hotelbeds_data,
        "metadata": {
            "total_count": len(hotelbeds_data),
            "total_rooms": total_rooms,
            "total_rooms_with_occupancy": total_rooms_with_occupancy,
            "source": "hotelbeds"
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Hotelbeds normalized data saved to {output_path}")
    print(f"✓ Hotelbeds normalized data saved to {output_path}")
    print(f"  - {len(hotelbeds_data)} hotels")
    print(f"  - {total_rooms} rooms")
    print(f"  - {total_rooms_with_occupancy} rooms with occupancy data")

def save_axisdata_normalized_data(
    axisdata: List[NormalizedHotel],
    output_path: str = "axisdata_norm_data.json"
) -> None:
    """
    Save normalized AxisData data to JSON file.
    
    Args:
        axisdata: List of normalized AxisData hotels
        output_path: Path where JSON file will be saved
    """
    axisdata_data = [asdict(hotel) for hotel in axisdata]
    
    # Calculate statistics
    total_rooms = sum(len(hotel['rooms']) for hotel in axisdata_data)
    total_rooms_with_occupancy = sum(
        sum(1 for room in hotel['rooms'] if room['occupancy'])
        for hotel in axisdata_data
    )
    
    data = {
        "hotels": axisdata_data,
        "metadata": {
            "total_count": len(axisdata_data),
            "total_rooms": total_rooms,
            "total_rooms_with_occupancy": total_rooms_with_occupancy,
            "source": "axisdata"
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"AxisData normalized data saved to {output_path}")
    print(f"✓ AxisData normalized data saved to {output_path}")
    print(f"  - {len(axisdata_data)} hotels")
    print(f"  - {total_rooms} rooms")
    print(f"  - {total_rooms_with_occupancy} rooms with occupancy data")

def print_normalization_sample(
    hotelbeds: List[NormalizedHotel],
    axisdata: List[NormalizedHotel],
):
    """Print sample normalized data for verification."""
    print("\n" + "=" * 80)
    print("NORMALIZATION SAMPLE")
    print("=" * 80)
    
    if hotelbeds:
        hotel = hotelbeds[0]
        print(f"\nHOTELBEDS SAMPLE:")
        print(f"  Hotel Code: {hotel.hotel_code}")
        print(f"  Hotel Name: {hotel.hotel_name}")
        print(f"  Normalized Name: {hotel.normalized_hotel_name}")
        print(f"  Place ID: {hotel.place_id}")
        print(f"  Rooms: {len(hotel.rooms)}")
        
        if hotel.rooms:
            room = hotel.rooms[0]
            print(f"\n  Sample Room:")
            print(f"    Room Code: {room.room_code}")
            print(f"    Room Name: {room.room_name}")
            print(f"    Description: {room.room_description}")
            print(f"    Normalized: {room.normalized_description}")
            if room.code_description:
                print(f"    Code Description: {room.code_description}")
                print(f"    Normalized Code Description: {room.normalized_code_description}")
            
            # ✓ NEW: Print occupancy data
            if room.occupancy:
                print(f"    Occupancy:")
                print(f"      Min/Max Occupancy: {room.occupancy.min_occupancy}/{room.occupancy.max_occupancy}")
                print(f"      Min/Max Adults: {room.occupancy.min_adults}/{room.occupancy.max_adults}")
                print(f"      Max Children: {room.occupancy.max_children}")
    
    if axisdata:
        hotel = axisdata[0]
        print(f"\n\nAXISDATA SAMPLE:")
        print(f"  Hotel Code: {hotel.hotel_code}")
        print(f"  Hotel Name: {hotel.hotel_name}")
        print(f"  Normalized Name: {hotel.normalized_hotel_name}")
        print(f"  Place ID: {hotel.place_id}")
        print(f"  Rooms: {len(hotel.rooms)}")
        
        if hotel.rooms:
            room = hotel.rooms[0]
            print(f"\n  Sample Room:")
            print(f"    Room Code: {room.room_code}")
            print(f"    Provider Code: {room.provider_code}")
            print(f"    Room Name: {room.room_name}")
            print(f"    Description: {room.room_description}")
            print(f"    Normalized: {room.normalized_description}")
            if room.code_description:
                print(f"    Code Description: {room.code_description}")
                print(f"    Normalized Code Description: {room.normalized_code_description}")
            
            # ✓ NEW: Print occupancy data
            if room.occupancy:
                print(f"    Occupancy:")
                print(f"      Min/Max Occupancy: {room.occupancy.min_occupancy}/{room.occupancy.max_occupancy}")
                print(f"      Min/Max Adults: {room.occupancy.min_adults}/{room.occupancy.max_adults}")
                print(f"      Min/Max Children: {room.occupancy.min_children}/{room.occupancy.max_children}")
                if room.occupancy.max_child_age:
                    print(f"      Max Child Age: {room.occupancy.max_child_age}")
                if room.occupancy.standard_occupancy:
                    print(f"      Standard Occupancy: {room.occupancy.standard_occupancy}")
    
    print("\n" + "=" * 80)

# ========================================================================
# USAGE EXAMPLES
# ========================================================================

if __name__ == "__main__":
    # Load raw data
    hotelbeds_raw, axisdata_raw, validation = load_all_data(
        hotelbeds_path=Path("data/outputs/enriched_hotelbeds_hotels.json"),
        axisdata_path=Path("data/inputs/axisdata_hotels.json"),
        strict_mode=False,
    )
    
    # Normalize data
    normalizer = DataNormalizer()
    hotelbeds_normalized, hb_errors = normalizer.normalize_hotelbeds_hotels(hotelbeds_raw)
    axisdata_normalized, ad_errors = normalizer.normalize_axisdata_hotels(axisdata_raw)
    
    # Save normalized data to separate JSON files
    save_hotelbeds_normalized_data(hotelbeds_normalized, output_path="data/outputs/hotelbed_norm_data.json")
    save_axisdata_normalized_data(axisdata_normalized, output_path="data/outputs/axisdata_norm_data.json")
    
    # Print samples
    print_normalization_sample(hotelbeds_normalized, axisdata_normalized)
    
    if hb_errors:
        print(f"\nHotelbeds Normalization Errors:")
        for error in hb_errors:
            print(f"  - {error}")
    
    if ad_errors:
        print(f"\nAxisData Normalization Errors:")
        for error in ad_errors:
            print(f"  - {error}")
