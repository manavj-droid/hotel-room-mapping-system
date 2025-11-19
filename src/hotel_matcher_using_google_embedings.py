import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import re

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ========================================================================
# CONFIGURATION
# ========================================================================

# Input embedding files
HOTELBEDS_HOTEL_EMBEDDINGS = Path("data/google_embedings/hotelbeds_hotel_embeddings_google.json")
HOTELBEDS_ROOM_EMBEDDINGS = Path("data/google_embedings/hotelbeds_room_embeddings_google.json")
AXISDATA_HOTEL_EMBEDDINGS = Path("data/google_embedings/axisdata_hotel_embeddings_google.json")
AXISDATA_ROOM_EMBEDDINGS = Path("data/google_embedings/axisdata_room_embeddings_google.json")

# Output file
OUTPUT_FILE = Path("data/mapped/mapped_data_google.json")

# Matching parameters
ROOM_SIMILARITY_THRESHOLD = 0.90
TOP_K_MATCHES = 3
INCLUDE_EMBEDDINGS = False

# ✓ UPDATED: Enhanced validation with occupancy
ENABLE_STRICT_VALIDATION = True

# Penalty configuration
BEDROOM_MISMATCH_PENALTY = 0.30          # -30% for bedroom count mismatch
MAX_OCCUPANCY_PENALTY_HIGH = 0.25        # -25% for ≥2 guests difference
MAX_OCCUPANCY_PENALTY_LOW = 0.10         # -10% for 1 guest difference
MAX_ADULTS_PENALTY_HIGH = 0.25           # -25% for ≥2 adults difference
MAX_ADULTS_PENALTY_LOW = 0.10            # -10% for 1 adult difference
MAX_CHILDREN_PENALTY = 0.15              # -15% for ≥2 children difference
ROOM_TYPE_INCOMPATIBLE_PENALTY = 0.50    # -50% for incompatible room types

# ========================================================================
# DATA MODELS
# ========================================================================

@dataclass
class HotelEmbedding:
    """Hotel with embedding."""
    hotel_code: str
    place_id: str
    hotel_name: str
    provider: str
    embedding: np.ndarray

@dataclass
class RoomEmbedding:
    """Room with embedding and occupancy data."""
    room_id: str
    hotel_code: str
    room_code: str
    provider: str
    text: str
    base_description: str  # ✓ NEW
    occupancy_text: str  # ✓ NEW
    occupancy: Dict[str, Any]  # ✓ NEW: Full occupancy object
    embedding: np.ndarray

@dataclass
class RoomMatch:
    """A matched room pair with validation details."""
    similarity_score: float
    adjusted_score: float
    hotelbeds_room_code: str
    hotelbeds_room_id: str
    hotelbeds_text: str
    hotelbeds_base_description: str  # ✓ NEW
    hotelbeds_occupancy_text: str  # ✓ NEW
    hotelbeds_occupancy: Dict[str, Any]  # ✓ NEW
    axisdata_room_code: str
    axisdata_room_id: str
    axisdata_text: str
    axisdata_base_description: str  # ✓ NEW
    axisdata_occupancy_text: str  # ✓ NEW
    axisdata_occupancy: Dict[str, Any]  # ✓ NEW
    validation_flags: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnmatchedRoom:
    """An unmatched room with full details."""
    room_code: str
    room_id: str
    text: str
    base_description: str  # ✓ NEW
    occupancy_text: str  # ✓ NEW
    occupancy: Dict[str, Any]  # ✓ NEW
    reason: str

@dataclass
class HotelPairMapping:
    """Complete mapping for a matched hotel pair."""
    place_id: str
    confidence: float
    hotelbeds_hotel_code: str
    hotelbeds_hotel_name: str
    hotelbeds_total_rooms: int
    axisdata_hotel_code: str
    axisdata_hotel_name: str
    axisdata_total_rooms: int
    room_matches: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_hotelbeds_rooms: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_axisdata_rooms: List[Dict[str, Any]] = field(default_factory=list)

# ========================================================================
# ROOM TYPE & OCCUPANCY VALIDATOR
# ========================================================================

class RoomTypeValidator:
    """Validates room matches for semantic and occupancy consistency."""
    
    def __init__(self):
        """Initialize validator with pattern rules."""
        self.bedroom_patterns = [
            (r'\b(\d+)\s*bedroom', 'explicit_number'),
            (r'\bone\s*bedroom', 'one'),
            (r'\btwo\s*bedroom', 'two'),
            (r'\bthree\s*bedroom', 'three'),
            (r'\bfour\s*bedroom', 'four'),
            (r'\bstudio\b', 'studio'),
        ]
        
        self.room_type_patterns = {
            'apartment': r'\b(apartment|apt)\b',
            'suite': r'\bsuite\b',
            'studio': r'\bstudio\b',
            'room': r'\broom\b',
            'double': r'\bdouble\b',
            'single': r'\bsingle\b',
            'twin': r'\btwin\b',
        }
    
    def extract_bedroom_count(self, text: str) -> Optional[int]:
        """Extract bedroom count from room text."""
        text_lower = text.lower()
        
        for pattern, match_type in self.bedroom_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if match_type == 'explicit_number':
                    return int(match.group(1))
                elif match_type == 'one':
                    return 1
                elif match_type == 'two':
                    return 2
                elif match_type == 'three':
                    return 3
                elif match_type == 'four':
                    return 4
                elif match_type == 'studio':
                    return 0
        
        return None
    
    def extract_room_type(self, text: str) -> Optional[str]:
        """Extract primary room type from text."""
        text_lower = text.lower()
        
        for room_type, pattern in self.room_type_patterns.items():
            if re.search(pattern, text_lower):
                return room_type
        
        return None
    
    def validate_match(
        self,
        hb_text: str,
        ad_text: str,
        hb_occupancy: Dict[str, Any],
        ad_occupancy: Dict[str, Any],
        similarity_score: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Validate a room match and apply penalties if needed.
        
        ✓ UPDATED: Now includes occupancy validation
        
        Args:
            hb_text: Hotelbeds room text
            ad_text: AxisData room text
            hb_occupancy: Hotelbeds occupancy dict
            ad_occupancy: AxisData occupancy dict
            similarity_score: Original cosine similarity
        
        Returns:
            Tuple of (adjusted_score, validation_flags)
        """
        validation_flags = {
            'bedroom_match': True,
            'room_type_match': True,
            'occupancy_match': True,  # ✓ NEW
            'penalties_applied': []
        }
        
        adjusted_score = similarity_score
        
        # ========================================
        # 1. BEDROOM COUNT VALIDATION
        # ========================================
        hb_bedrooms = self.extract_bedroom_count(hb_text)
        ad_bedrooms = self.extract_bedroom_count(ad_text)
        
        if hb_bedrooms is not None and ad_bedrooms is not None:
            if hb_bedrooms != ad_bedrooms:
                validation_flags['bedroom_match'] = False
                validation_flags['hb_bedrooms'] = hb_bedrooms
                validation_flags['ad_bedrooms'] = ad_bedrooms
                
                adjusted_score *= (1 - BEDROOM_MISMATCH_PENALTY)
                validation_flags['penalties_applied'].append(
                    f'bedroom_mismatch: HB={hb_bedrooms} AD={ad_bedrooms} (-{BEDROOM_MISMATCH_PENALTY*100:.0f}%)'
                )
        
        # ========================================
        # 2. OCCUPANCY VALIDATION (NEW!)
        # ========================================
        occupancy_penalties = self._validate_occupancy(
            hb_occupancy,
            ad_occupancy,
            validation_flags
        )
        
        for penalty_name, penalty_value in occupancy_penalties:
            adjusted_score *= (1 - penalty_value)
            validation_flags['penalties_applied'].append(penalty_name)
        
        if occupancy_penalties:
            validation_flags['occupancy_match'] = False
        
        # ========================================
        # 3. ROOM TYPE VALIDATION
        # ========================================
        hb_type = self.extract_room_type(hb_text)
        ad_type = self.extract_room_type(ad_text)
        
        if hb_type and ad_type:
            # Studio should not match with multi-bedroom
            if (hb_type == 'studio' and ad_bedrooms and ad_bedrooms > 0) or \
               (ad_type == 'studio' and hb_bedrooms and hb_bedrooms > 0):
                validation_flags['room_type_match'] = False
                adjusted_score *= (1 - ROOM_TYPE_INCOMPATIBLE_PENALTY)
                validation_flags['penalties_applied'].append(
                    f'studio_bedroom_mismatch (-{ROOM_TYPE_INCOMPATIBLE_PENALTY*100:.0f}%)'
                )
        
        return adjusted_score, validation_flags
    
    def _validate_occupancy(
        self,
        hb_occupancy: Dict[str, Any],
        ad_occupancy: Dict[str, Any],
        validation_flags: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Validate occupancy fields and return penalties.
        
        ✓ NEW: Comprehensive occupancy validation
        
        Returns:
            List of (penalty_description, penalty_value) tuples
        """
        penalties = []
        
        if not hb_occupancy or not ad_occupancy:
            return penalties
        
        # Store occupancy details for debugging
        validation_flags['hb_occupancy'] = hb_occupancy
        validation_flags['ad_occupancy'] = ad_occupancy
        
        # ========================================
        # A. MAX OCCUPANCY (Total Guests) - HIGHEST PRIORITY
        # ========================================
        hb_max_occ = hb_occupancy.get('max_occupancy')
        ad_max_occ = ad_occupancy.get('max_occupancy')
        
        if hb_max_occ is not None and ad_max_occ is not None:
            occ_diff = abs(hb_max_occ - ad_max_occ)
            
            if occ_diff >= 2:
                penalties.append((
                    f'max_occupancy_high_diff: HB={hb_max_occ} AD={ad_max_occ} (-{MAX_OCCUPANCY_PENALTY_HIGH*100:.0f}%)',
                    MAX_OCCUPANCY_PENALTY_HIGH
                ))
            elif occ_diff == 1:
                penalties.append((
                    f'max_occupancy_low_diff: HB={hb_max_occ} AD={ad_max_occ} (-{MAX_OCCUPANCY_PENALTY_LOW*100:.0f}%)',
                    MAX_OCCUPANCY_PENALTY_LOW
                ))
        
        # ========================================
        # B. MAX ADULTS - HIGH PRIORITY
        # ========================================
        hb_max_adults = hb_occupancy.get('max_adults')
        ad_max_adults = ad_occupancy.get('max_adults')
        
        if hb_max_adults is not None and ad_max_adults is not None:
            adults_diff = abs(hb_max_adults - ad_max_adults)
            
            if adults_diff >= 2:
                penalties.append((
                    f'max_adults_high_diff: HB={hb_max_adults} AD={ad_max_adults} (-{MAX_ADULTS_PENALTY_HIGH*100:.0f}%)',
                    MAX_ADULTS_PENALTY_HIGH
                ))
            elif adults_diff == 1:
                penalties.append((
                    f'max_adults_low_diff: HB={hb_max_adults} AD={ad_max_adults} (-{MAX_ADULTS_PENALTY_LOW*100:.0f}%)',
                    MAX_ADULTS_PENALTY_LOW
                ))
        
        # ========================================
        # C. MAX CHILDREN - MEDIUM PRIORITY
        # ========================================
        hb_max_children = hb_occupancy.get('max_children')
        ad_max_children = ad_occupancy.get('max_children')
        
        if hb_max_children is not None and ad_max_children is not None:
            children_diff = abs(hb_max_children - ad_max_children)
            
            if children_diff >= 2:
                penalties.append((
                    f'max_children_diff: HB={hb_max_children} AD={ad_max_children} (-{MAX_CHILDREN_PENALTY*100:.0f}%)',
                    MAX_CHILDREN_PENALTY
                ))
        
        return penalties

# ========================================================================
# CLASS 1: EMBEDDINGS LOADER
# ========================================================================

class EmbeddingsLoader:
    """Loads and manages embeddings with occupancy data from JSON files."""

    def __init__(self):
        """Initialize loader."""
        self.hotelbeds_hotels: Dict[str, HotelEmbedding] = {}
        self.axisdata_hotels: Dict[str, HotelEmbedding] = {}
        self.hotelbeds_rooms: Dict[str, List[RoomEmbedding]] = defaultdict(list)
        self.axisdata_rooms: Dict[str, List[RoomEmbedding]] = defaultdict(list)

    def load_all_embeddings(self):
        """Load all embeddings from JSON files."""
        logger.info("=" * 80)
        logger.info("LOADING GOOGLE EMBEDDINGS WITH OCCUPANCY DATA")
        logger.info("=" * 80)

        # Load hotel embeddings
        logger.info("\nLoading Hotelbeds hotel embeddings...")
        self._load_hotel_embeddings(
            HOTELBEDS_HOTEL_EMBEDDINGS,
            "hotelbeds",
            self.hotelbeds_hotels
        )

        logger.info("Loading AxisData hotel embeddings...")
        self._load_hotel_embeddings(
            AXISDATA_HOTEL_EMBEDDINGS,
            "axisdata",
            self.axisdata_hotels
        )

        # Load room embeddings with occupancy
        logger.info("\nLoading Hotelbeds room embeddings...")
        self._load_room_embeddings(
            HOTELBEDS_ROOM_EMBEDDINGS,
            "hotelbeds",
            self.hotelbeds_rooms
        )

        logger.info("Loading AxisData room embeddings...")
        self._load_room_embeddings(
            AXISDATA_ROOM_EMBEDDINGS,
            "axisdata",
            self.axisdata_rooms
        )

        logger.info("\n✓ All embeddings loaded successfully")
        logger.info(f"  Hotelbeds: {len(self.hotelbeds_hotels)} hotels, "
                   f"{sum(len(rooms) for rooms in self.hotelbeds_rooms.values())} rooms")
        logger.info(f"  AxisData: {len(self.axisdata_hotels)} hotels, "
                   f"{sum(len(rooms) for rooms in self.axisdata_rooms.values())} rooms")

    def _load_hotel_embeddings(
        self,
        file_path: Path,
        provider: str,
        storage: Dict[str, HotelEmbedding]
    ):
        """Load hotel embeddings from JSON."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        for hotel_data in data['embeddings']:
            hotel = HotelEmbedding(
                hotel_code=hotel_data['hotel_code'],
                place_id=hotel_data['place_id'],
                hotel_name=hotel_data['hotel_name'],
                provider=provider,
                embedding=np.array(hotel_data['embedding'])
            )
            storage[hotel.hotel_code] = hotel

        logger.info(f"  ✓ Loaded {len(storage)} {provider} hotels")

    def _load_room_embeddings(
        self,
        file_path: Path,
        provider: str,
        storage: Dict[str, List[RoomEmbedding]]
    ):
        """
        Load room embeddings with occupancy data from JSON.
        
        ✓ UPDATED: Now extracts occupancy data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        rooms_with_occupancy = 0

        for room_data in data['embeddings']:
            # Extract occupancy data
            occupancy = room_data.get('occupancy', {})
            if occupancy:
                rooms_with_occupancy += 1
            
            room = RoomEmbedding(
                room_id=room_data['room_id'],
                hotel_code=room_data['hotel_code'],
                room_code=room_data['room_code'],
                provider=provider,
                text=room_data['text_for_embedding'],
                base_description=room_data.get('base_description', ''),  # ✓ NEW
                occupancy_text=room_data.get('occupancy_text', ''),  # ✓ NEW
                occupancy=occupancy,  # ✓ NEW
                embedding=np.array(room_data['embedding'])
            )
            storage[room.hotel_code].append(room)

        total_rooms = sum(len(rooms) for rooms in storage.values())
        logger.info(f"  ✓ Loaded {total_rooms} {provider} rooms")
        logger.info(f"  ✓ Rooms with occupancy data: {rooms_with_occupancy}")

# ========================================================================
# CLASS 2: HOTEL PAIR MATCHER
# ========================================================================

class HotelPairMatcher:
    """Matches hotels by place_id."""

    def __init__(
        self,
        hotelbeds_hotels: Dict[str, HotelEmbedding],
        axisdata_hotels: Dict[str, HotelEmbedding]
    ):
        """Initialize matcher."""
        self.hotelbeds_hotels = hotelbeds_hotels
        self.axisdata_hotels = axisdata_hotels
        self.matched_pairs: List[Tuple[HotelEmbedding, HotelEmbedding]] = []

    def match_by_place_id(self) -> List[Tuple[HotelEmbedding, HotelEmbedding]]:
        """Match hotels by place_id (deterministic)."""
        logger.info("\n" + "=" * 80)
        logger.info("MATCHING HOTELS BY PLACE_ID")
        logger.info("=" * 80)

        # Group by place_id
        hb_by_place_id = defaultdict(list)
        for hotel in self.hotelbeds_hotels.values():
            hb_by_place_id[hotel.place_id].append(hotel)

        ad_by_place_id = defaultdict(list)
        for hotel in self.axisdata_hotels.values():
            ad_by_place_id[hotel.place_id].append(hotel)

        # Find matches
        common_place_ids = set(hb_by_place_id.keys()) & set(ad_by_place_id.keys())

        for place_id in sorted(common_place_ids):
            hb_hotels = hb_by_place_id[place_id]
            ad_hotels = ad_by_place_id[place_id]

            for hb_hotel in hb_hotels:
                for ad_hotel in ad_hotels:
                    self.matched_pairs.append((hb_hotel, ad_hotel))

        logger.info(f"✓ Found {len(common_place_ids)} common place_ids")
        logger.info(f"✓ Created {len(self.matched_pairs)} hotel pairs")

        # Log unmatched
        unmatched_hb = [h for h in self.hotelbeds_hotels.values()
                       if h.place_id not in common_place_ids]
        unmatched_ad = [h for h in self.axisdata_hotels.values()
                       if h.place_id not in common_place_ids]

        if unmatched_hb:
            logger.info(f"  Unmatched Hotelbeds: {len(unmatched_hb)}")
        if unmatched_ad:
            logger.info(f"  Unmatched AxisData: {len(unmatched_ad)}")

        return self.matched_pairs

# ========================================================================
# CLASS 3: ROOM PAIR MATCHER (WITH OCCUPANCY VALIDATION)
# ========================================================================

class RoomPairMatcher:
    """Matches rooms within hotel pairs using embeddings with occupancy validation."""

    def __init__(
        self,
        hotelbeds_rooms: Dict[str, List[RoomEmbedding]],
        axisdata_rooms: Dict[str, List[RoomEmbedding]],
        similarity_threshold: float = ROOM_SIMILARITY_THRESHOLD,
        top_k: int = TOP_K_MATCHES,
        enable_validation: bool = ENABLE_STRICT_VALIDATION
    ):
        """Initialize matcher."""
        self.hotelbeds_rooms = hotelbeds_rooms
        self.axisdata_rooms = axisdata_rooms
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.enable_validation = enable_validation
        self.validator = RoomTypeValidator() if enable_validation else None

    def match_rooms_for_hotel_pair(
        self,
        hb_hotel: HotelEmbedding,
        ad_hotel: HotelEmbedding
    ) -> Tuple[List[RoomMatch], List[UnmatchedRoom], List[UnmatchedRoom]]:
        """
        Match rooms for a specific hotel pair with occupancy validation.
        
        ✓ UPDATED: Now uses occupancy data in validation
        """
        hb_rooms = self.hotelbeds_rooms.get(hb_hotel.hotel_code, [])
        ad_rooms = self.axisdata_rooms.get(ad_hotel.hotel_code, [])

        if not hb_rooms or not ad_rooms:
            unmatched_hb = [
                UnmatchedRoom(
                    room_code=r.room_code,
                    room_id=r.room_id,
                    text=r.text,
                    base_description=r.base_description,
                    occupancy_text=r.occupancy_text,
                    occupancy=r.occupancy,
                    reason="no_axisdata_rooms_available"
                )
                for r in hb_rooms
            ]
            unmatched_ad = [
                UnmatchedRoom(
                    room_code=r.room_code,
                    room_id=r.room_id,
                    text=r.text,
                    base_description=r.base_description,
                    occupancy_text=r.occupancy_text,
                    occupancy=r.occupancy,
                    reason="no_hotelbeds_rooms_available"
                )
                for r in ad_rooms
            ]
            return [], unmatched_hb, unmatched_ad

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(hb_rooms, ad_rooms)

        # Find best matches with occupancy validation
        matches, unmatched_hb, unmatched_ad = self._match_with_validation(
            hb_rooms,
            ad_rooms,
            similarity_matrix
        )

        return matches, unmatched_hb, unmatched_ad

    def _calculate_similarity_matrix(
        self,
        hb_rooms: List[RoomEmbedding],
        ad_rooms: List[RoomEmbedding]
    ) -> np.ndarray:
        """Calculate cosine similarity matrix."""
        hb_embeddings = np.array([room.embedding for room in hb_rooms])
        ad_embeddings = np.array([room.embedding for room in ad_rooms])

        # Cosine similarity
        hb_normalized = hb_embeddings / np.linalg.norm(hb_embeddings, axis=1, keepdims=True)
        ad_normalized = ad_embeddings / np.linalg.norm(ad_embeddings, axis=1, keepdims=True)

        similarity_matrix = np.dot(hb_normalized, ad_normalized.T)

        return similarity_matrix

    def _match_with_validation(
        self,
        hb_rooms: List[RoomEmbedding],
        ad_rooms: List[RoomEmbedding],
        similarity_matrix: np.ndarray
    ) -> Tuple[List[RoomMatch], List[UnmatchedRoom], List[UnmatchedRoom]]:
        """
        Match rooms with semantic and occupancy validation.
        
        ✓ UPDATED: Now passes occupancy data to validator
        """
        matches = []
        matched_hb_indices = set()
        matched_ad_indices = set()
        
        unmatched_hb_rooms = []
        unmatched_ad_rooms = []

        # For each Hotelbeds room, find best valid match
        for hb_idx, hb_room in enumerate(hb_rooms):
            similarities = similarity_matrix[hb_idx]

            # Get all candidates above threshold
            candidate_indices = [
                (ad_idx, sim)
                for ad_idx, sim in enumerate(similarities)
                if sim >= self.similarity_threshold and ad_idx not in matched_ad_indices
            ]

            # Sort by similarity descending
            candidate_indices.sort(key=lambda x: x[1], reverse=True)

            if not candidate_indices:
                max_sim = float(np.max(similarities))
                reason = "below_threshold" if max_sim < self.similarity_threshold else "no_match_found"
                
                unmatched_hb_rooms.append(UnmatchedRoom(
                    room_code=hb_room.room_code,
                    room_id=hb_room.room_id,
                    text=hb_room.text,
                    base_description=hb_room.base_description,
                    occupancy_text=hb_room.occupancy_text,
                    occupancy=hb_room.occupancy,
                    reason=reason
                ))
                continue

            # Find best valid match with occupancy validation
            best_match = None
            best_adjusted_score = 0
            best_validation_flags = None

            for ad_idx, similarity in candidate_indices:
                ad_room = ad_rooms[ad_idx]

                # Apply validation with occupancy
                if self.enable_validation and self.validator:
                    adjusted_score, validation_flags = self.validator.validate_match(
                        hb_room.text,
                        ad_room.text,
                        hb_room.occupancy,  # ✓ NEW: Pass occupancy
                        ad_room.occupancy,  # ✓ NEW: Pass occupancy
                        similarity
                    )
                else:
                    adjusted_score = similarity
                    validation_flags = {
                        'bedroom_match': True,
                        'room_type_match': True,
                        'occupancy_match': True,
                        'penalties_applied': []
                    }

                # Check if adjusted score still meets threshold
                if adjusted_score >= self.similarity_threshold:
                    if adjusted_score > best_adjusted_score:
                        best_adjusted_score = adjusted_score
                        best_match = (ad_idx, similarity, adjusted_score, validation_flags)

            if best_match:
                ad_idx, original_score, adjusted_score, validation_flags = best_match
                ad_room = ad_rooms[ad_idx]

                match = RoomMatch(
                    similarity_score=float(original_score),
                    adjusted_score=float(adjusted_score),
                    hotelbeds_room_code=hb_room.room_code,
                    hotelbeds_room_id=hb_room.room_id,
                    hotelbeds_text=hb_room.text,
                    hotelbeds_base_description=hb_room.base_description,  # ✓ NEW
                    hotelbeds_occupancy_text=hb_room.occupancy_text,  # ✓ NEW
                    hotelbeds_occupancy=hb_room.occupancy,  # ✓ NEW
                    axisdata_room_code=ad_room.room_code,
                    axisdata_room_id=ad_room.room_id,
                    axisdata_text=ad_room.text,
                    axisdata_base_description=ad_room.base_description,  # ✓ NEW
                    axisdata_occupancy_text=ad_room.occupancy_text,  # ✓ NEW
                    axisdata_occupancy=ad_room.occupancy,  # ✓ NEW
                    validation_flags=validation_flags
                )
                matches.append(match)

                matched_hb_indices.add(hb_idx)
                matched_ad_indices.add(ad_idx)
            else:
                # No valid match found after validation
                unmatched_hb_rooms.append(UnmatchedRoom(
                    room_code=hb_room.room_code,
                    room_id=hb_room.room_id,
                    text=hb_room.text,
                    base_description=hb_room.base_description,
                    occupancy_text=hb_room.occupancy_text,
                    occupancy=hb_room.occupancy,
                    reason="failed_validation_checks"
                ))

        # All unmatched AxisData rooms
        for ad_idx, ad_room in enumerate(ad_rooms):
            if ad_idx not in matched_ad_indices:
                unmatched_ad_rooms.append(UnmatchedRoom(
                    room_code=ad_room.room_code,
                    room_id=ad_room.room_id,
                    text=ad_room.text,
                    base_description=ad_room.base_description,
                    occupancy_text=ad_room.occupancy_text,
                    occupancy=ad_room.occupancy,
                    reason="no_hotelbeds_match_found"
                ))

        return matches, unmatched_hb_rooms, unmatched_ad_rooms

# ========================================================================
# CLASS 4: UNIFIED MATCHER (ORCHESTRATOR)
# ========================================================================

class UnifiedMatcher:
    """Orchestrates the complete matching pipeline with occupancy validation."""

    def __init__(self):
        """Initialize matcher."""
        self.loader = EmbeddingsLoader()
        self.hotel_matcher = None
        self.room_matcher = None
        self.results: List[HotelPairMapping] = []

    def run_complete_matching(self) -> List[HotelPairMapping]:
        """Run the complete matching pipeline with occupancy validation."""
        # Step 1: Load embeddings with occupancy
        self.loader.load_all_embeddings()

        # Step 2: Match hotels by place_id
        self.hotel_matcher = HotelPairMatcher(
            self.loader.hotelbeds_hotels,
            self.loader.axisdata_hotels
        )
        hotel_pairs = self.hotel_matcher.match_by_place_id()

        # Step 3: Match rooms with occupancy validation
        logger.info("\n" + "=" * 80)
        logger.info("MATCHING ROOMS WITH OCCUPANCY VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Validation enabled: {ENABLE_STRICT_VALIDATION}")
        logger.info(f"Similarity threshold: {ROOM_SIMILARITY_THRESHOLD}")
        logger.info(f"Bedroom mismatch penalty: {BEDROOM_MISMATCH_PENALTY * 100:.0f}%")
        logger.info(f"Max occupancy penalty (high): {MAX_OCCUPANCY_PENALTY_HIGH * 100:.0f}%")
        logger.info(f"Max adults penalty (high): {MAX_ADULTS_PENALTY_HIGH * 100:.0f}%")

        self.room_matcher = RoomPairMatcher(
            self.loader.hotelbeds_rooms,
            self.loader.axisdata_rooms
        )

        total_room_matches = 0
        total_unmatched_hb = 0
        total_unmatched_ad = 0
        total_validation_flags = 0
        total_occupancy_warnings = 0

        for idx, (hb_hotel, ad_hotel) in enumerate(hotel_pairs, 1):
            logger.info(f"\nPair {idx}/{len(hotel_pairs)}: {hb_hotel.hotel_code} ↔ {ad_hotel.hotel_code}")

            room_matches, unmatched_hb, unmatched_ad = self.room_matcher.match_rooms_for_hotel_pair(
                hb_hotel,
                ad_hotel
            )

            # Count validation flags
            flagged_matches = sum(
                1 for m in room_matches
                if not m.validation_flags.get('bedroom_match', True) or
                   not m.validation_flags.get('room_type_match', True) or
                   not m.validation_flags.get('occupancy_match', True)
            )
            
            occupancy_warnings = sum(
                1 for m in room_matches
                if not m.validation_flags.get('occupancy_match', True)
            )

            total_room_matches += len(room_matches)
            total_unmatched_hb += len(unmatched_hb)
            total_unmatched_ad += len(unmatched_ad)
            total_validation_flags += flagged_matches
            total_occupancy_warnings += occupancy_warnings

            logger.info(f"  ✓ Matched: {len(room_matches)} rooms")
            if flagged_matches > 0:
                logger.info(f"  ⚠ Validation warnings: {flagged_matches} (occupancy: {occupancy_warnings})")
            logger.info(f"  ✗ Unmatched HB: {len(unmatched_hb)}, AD: {len(unmatched_ad)}")

            # Create mapping
            mapping = HotelPairMapping(
                place_id=hb_hotel.place_id,
                confidence=1.0,
                hotelbeds_hotel_code=hb_hotel.hotel_code,
                hotelbeds_hotel_name=hb_hotel.hotel_name,
                hotelbeds_total_rooms=len(self.loader.hotelbeds_rooms.get(hb_hotel.hotel_code, [])),
                axisdata_hotel_code=ad_hotel.hotel_code,
                axisdata_hotel_name=ad_hotel.hotel_name,
                axisdata_total_rooms=len(self.loader.axisdata_rooms.get(ad_hotel.hotel_code, [])),
                room_matches=[asdict(m) for m in room_matches],
                unmatched_hotelbeds_rooms=[asdict(r) for r in unmatched_hb],
                unmatched_axisdata_rooms=[asdict(r) for r in unmatched_ad]
            )

            self.results.append(mapping)

        logger.info("\n" + "=" * 80)
        logger.info("MATCHING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Total hotel pairs: {len(hotel_pairs)}")
        logger.info(f"✓ Total room matches: {total_room_matches}")
        logger.info(f"⚠ Matches with validation warnings: {total_validation_flags}")
        logger.info(f"⚠ Matches with occupancy warnings: {total_occupancy_warnings}")
        logger.info(f"✗ Total unmatched HB rooms: {total_unmatched_hb}")
        logger.info(f"✗ Total unmatched AD rooms: {total_unmatched_ad}")

        return self.results

# ========================================================================
# OUTPUT FUNCTIONS
# ========================================================================

def save_mapped_data(
    results: List[HotelPairMapping],
    output_path: Path = OUTPUT_FILE
):
    """Save mapping results to JSON with occupancy data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate summary statistics
    total_room_matches = sum(len(mapping.room_matches) for mapping in results)

    similarities = [
        match['similarity_score']
        for mapping in results
        for match in mapping.room_matches
    ]
    
    adjusted_similarities = [
        match['adjusted_score']
        for mapping in results
        for match in mapping.room_matches
    ]
    
    # ✓ NEW: Count occupancy-related warnings
    occupancy_warnings = sum(
        1 for mapping in results
        for match in mapping.room_matches
        if not match['validation_flags'].get('occupancy_match', True)
    )

    summary = {
        "total_hotel_pairs": len(results),
        "total_room_matches": total_room_matches,
        "occupancy_warnings": occupancy_warnings,  # ✓ NEW
        "total_unmatched_rooms": {
            "hotelbeds": sum(len(m.unmatched_hotelbeds_rooms) for m in results),
            "axisdata": sum(len(m.unmatched_axisdata_rooms) for m in results)
        },
        "room_similarity_stats": {
            "original": {
                "mean": float(np.mean(similarities)) if similarities else 0.0,
                "median": float(np.median(similarities)) if similarities else 0.0,
                "min": float(np.min(similarities)) if similarities else 0.0,
                "max": float(np.max(similarities)) if similarities else 0.0,
            },
            "adjusted": {
                "mean": float(np.mean(adjusted_similarities)) if adjusted_similarities else 0.0,
                "median": float(np.median(adjusted_similarities)) if adjusted_similarities else 0.0,
                "min": float(np.min(adjusted_similarities)) if adjusted_similarities else 0.0,
                "max": float(np.max(adjusted_similarities)) if adjusted_similarities else 0.0,
            }
        }
    }

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "similarity_threshold": ROOM_SIMILARITY_THRESHOLD,
            "matching_strategy": "rank_1_with_occupancy_validation",
            "model": "google/text-embedding-004",
            "embedding_dim": 768,
            "validation_enabled": ENABLE_STRICT_VALIDATION,
            "penalties": {  # ✓ NEW: Document all penalties
                "bedroom_mismatch": BEDROOM_MISMATCH_PENALTY,
                "max_occupancy_high": MAX_OCCUPANCY_PENALTY_HIGH,
                "max_occupancy_low": MAX_OCCUPANCY_PENALTY_LOW,
                "max_adults_high": MAX_ADULTS_PENALTY_HIGH,
                "max_adults_low": MAX_ADULTS_PENALTY_LOW,
                "max_children": MAX_CHILDREN_PENALTY,
                "room_type_incompatible": ROOM_TYPE_INCOMPATIBLE_PENALTY
            }
        },
        "summary": summary,
        "hotel_pairs": [asdict(mapping) for mapping in results]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✓ Saved mapping results to {output_path}")

def print_sample_mappings(results: List[HotelPairMapping], sample_size: int = 2):
    """Print sample mappings for verification with occupancy details."""
    print("\n" + "=" * 80)
    print("SAMPLE HOTEL & ROOM MAPPINGS (WITH OCCUPANCY VALIDATION)")
    print("=" * 80)

    for idx, mapping in enumerate(results[:sample_size], 1):
        print(f"\nHotel Pair #{idx}:")
        print(f"  Place ID: {mapping.place_id}")
        print(f"  Hotelbeds: {mapping.hotelbeds_hotel_code} ({mapping.hotelbeds_hotel_name})")
        print(f"    Total rooms: {mapping.hotelbeds_total_rooms}")
        print(f"  AxisData: {mapping.axisdata_hotel_code} ({mapping.axisdata_hotel_name})")
        print(f"    Total rooms: {mapping.axisdata_total_rooms}")

        print(f"\n  Room Matches: {len(mapping.room_matches)}")
        for match in mapping.room_matches[:3]:
            print(f"\n    Similarity {match['similarity_score']:.3f} → Adjusted {match['adjusted_score']:.3f}")
            print(f"      HB: {match['hotelbeds_room_code']}")
            print(f"          Text: {match['hotelbeds_text']}")
            print(f"          Occupancy: {match['hotelbeds_occupancy_text']}")
            print(f"      AD: {match['axisdata_room_code']}")
            print(f"          Text: {match['axisdata_text']}")
            print(f"          Occupancy: {match['axisdata_occupancy_text']}")
            
            # Show validation flags
            flags = match.get('validation_flags', {})
            if not flags.get('bedroom_match', True):
                print(f"      ⚠ Bedroom mismatch: HB={flags.get('hb_bedrooms')} AD={flags.get('ad_bedrooms')}")
            if not flags.get('occupancy_match', True):
                hb_occ = flags.get('hb_occupancy', {})
                ad_occ = flags.get('ad_occupancy', {})
                print(f"      ⚠ Occupancy mismatch:")
                print(f"         HB: max_occ={hb_occ.get('max_occupancy')} max_adults={hb_occ.get('max_adults')} max_children={hb_occ.get('max_children')}")
                print(f"         AD: max_occ={ad_occ.get('max_occupancy')} max_adults={ad_occ.get('max_adults')} max_children={ad_occ.get('max_children')}")
            if flags.get('penalties_applied'):
                print(f"      ⚠ Penalties: {', '.join(flags['penalties_applied'])}")

        print(f"\n  Unmatched Hotelbeds rooms: {len(mapping.unmatched_hotelbeds_rooms)}")
        for room in mapping.unmatched_hotelbeds_rooms[:2]:
            print(f"    {room['room_code']}")
            print(f"      Text: {room['text']}")
            print(f"      Occupancy: {room['occupancy_text']}")
            print(f"      Reason: {room['reason']}")

        print(f"\n  Unmatched AxisData rooms: {len(mapping.unmatched_axisdata_rooms)}")
        for room in mapping.unmatched_axisdata_rooms[:2]:
            print(f"    {room['room_code']}")
            print(f"      Text: {room['text']}")
            print(f"      Occupancy: {room['occupancy_text']}")
            print(f"      Reason: {room['reason']}")

    print("\n" + "=" * 80)

# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("HOTEL & ROOM MATCHER - GOOGLE EMBEDDINGS + OCCUPANCY VALIDATION")
    logger.info("=" * 80)

    try:
        # Run matching
        matcher = UnifiedMatcher()
        results = matcher.run_complete_matching()

        # Save results
        save_mapped_data(results)

        # Print sample
        print_sample_mappings(results, sample_size=2)

        logger.info("\n✓ MATCHING COMPLETE!")
        logger.info(f"✓ Output saved to: {OUTPUT_FILE}")

        return 0

    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
