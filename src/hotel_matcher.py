import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ========================================================================
# CONFIGURATION
# ========================================================================

# Input embedding files
HOTELBEDS_HOTEL_EMBEDDINGS = Path("data/embeddings/hotelbeds_hotel_embeddings.json")
HOTELBEDS_ROOM_EMBEDDINGS = Path("data/embeddings/hotelbeds_room_embeddings.json")
AXISDATA_HOTEL_EMBEDDINGS = Path("data/embeddings/axisdata_hotel_embeddings.json")
AXISDATA_ROOM_EMBEDDINGS = Path("data/embeddings/axisdata_room_embeddings.json")

# Output file
OUTPUT_FILE = Path("data/mapped/mapped_data.json")

# Matching parameters
ROOM_SIMILARITY_THRESHOLD = 0.75  # Minimum similarity to consider a match
TOP_K_MATCHES = 3  # Return top-3 room matches per Hotelbeds room
INCLUDE_EMBEDDINGS = False  # Include embeddings in output (bloats file)


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
    embedding: np.ndarray  # Numpy array for computation


@dataclass
class RoomEmbedding:
    """Room with embedding."""
    room_id: str
    hotel_code: str
    room_code: str
    provider: str
    text: str
    embedding: np.ndarray  # Numpy array for computation


@dataclass
class RoomMatch:
    """A matched room pair (rank 1 only)."""
    similarity_score: float
    hotelbeds_room_code: str
    hotelbeds_room_id: str
    hotelbeds_text: str
    axisdata_room_code: str
    axisdata_room_id: str
    axisdata_text: str


@dataclass
class UnmatchedRoom:
    """An unmatched room with full details."""
    room_code: str
    room_id: str
    text: str
    reason: str  # Why it's unmatched: "no_match_found" or "below_threshold" or "rank_2_or_higher"


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
# CLASS 1: EMBEDDINGS LOADER
# ========================================================================


class EmbeddingsLoader:
    """Loads and manages embeddings from JSON files."""

    def __init__(self):
        """Initialize loader."""
        self.hotelbeds_hotels: Dict[str, HotelEmbedding] = {}
        self.axisdata_hotels: Dict[str, HotelEmbedding] = {}
        self.hotelbeds_rooms: Dict[str, List[RoomEmbedding]] = defaultdict(list)
        self.axisdata_rooms: Dict[str, List[RoomEmbedding]] = defaultdict(list)

    def load_all_embeddings(self):
        """Load all embeddings from JSON files."""
        logger.info("=" * 80)
        logger.info("LOADING EMBEDDINGS")
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

        # Load room embeddings
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
        """Load room embeddings from JSON."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        for room_data in data['embeddings']:
            room = RoomEmbedding(
                room_id=room_data['room_id'],
                hotel_code=room_data['hotel_code'],
                room_code=room_data['room_code'],
                provider=provider,
                text=room_data['text_for_embedding'],
                embedding=np.array(room_data['embedding'])
            )
            storage[room.hotel_code].append(room)

        total_rooms = sum(len(rooms) for rooms in storage.values())
        logger.info(f"  ✓ Loaded {total_rooms} {provider} rooms")


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
        """
        Match hotels by place_id (deterministic).

        Returns:
            List of (hotelbeds_hotel, axisdata_hotel) tuples
        """
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

            # Match all combinations (usually 1:1)
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
# CLASS 3: ROOM PAIR MATCHER
# ========================================================================


class RoomPairMatcher:
    """Matches rooms within hotel pairs using embeddings."""

    def __init__(
        self,
        hotelbeds_rooms: Dict[str, List[RoomEmbedding]],
        axisdata_rooms: Dict[str, List[RoomEmbedding]],
        similarity_threshold: float = ROOM_SIMILARITY_THRESHOLD,
        top_k: int = TOP_K_MATCHES
    ):
        """Initialize matcher."""
        self.hotelbeds_rooms = hotelbeds_rooms
        self.axisdata_rooms = axisdata_rooms
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def match_rooms_for_hotel_pair(
        self,
        hb_hotel: HotelEmbedding,
        ad_hotel: HotelEmbedding
    ) -> Tuple[List[RoomMatch], List[UnmatchedRoom], List[UnmatchedRoom]]:
        """
        Match rooms for a specific hotel pair.
        
        Only rank 1 matches are returned in matches.
        All other ranks and unmatched rooms go to unmatched lists with full details.

        Args:
            hb_hotel: Hotelbeds hotel
            ad_hotel: AxisData hotel

        Returns:
            Tuple of (matches, unmatched_hb_rooms, unmatched_ad_rooms)
        """
        hb_rooms = self.hotelbeds_rooms.get(hb_hotel.hotel_code, [])
        ad_rooms = self.axisdata_rooms.get(ad_hotel.hotel_code, [])

        if not hb_rooms or not ad_rooms:
            # No rooms to match - all are unmatched
            unmatched_hb = [
                UnmatchedRoom(
                    room_code=r.room_code,
                    room_id=r.room_id,
                    text=r.text,
                    reason="no_axisdata_rooms_available"
                )
                for r in hb_rooms
            ]
            unmatched_ad = [
                UnmatchedRoom(
                    room_code=r.room_code,
                    room_id=r.room_id,
                    text=r.text,
                    reason="no_hotelbeds_rooms_available"
                )
                for r in ad_rooms
            ]
            return [], unmatched_hb, unmatched_ad

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(hb_rooms, ad_rooms)

        # Find best matches - only rank 1
        matches, unmatched_hb, unmatched_ad = self._match_rank_1_only(
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
        """
        Calculate cosine similarity matrix between room embeddings.

        Returns:
            2D array of shape (len(hb_rooms), len(ad_rooms))
        """
        hb_embeddings = np.array([room.embedding for room in hb_rooms])
        ad_embeddings = np.array([room.embedding for room in ad_rooms])

        # Cosine similarity = dot product of normalized vectors
        hb_normalized = hb_embeddings / np.linalg.norm(hb_embeddings, axis=1, keepdims=True)
        ad_normalized = ad_embeddings / np.linalg.norm(ad_embeddings, axis=1, keepdims=True)

        similarity_matrix = np.dot(hb_normalized, ad_normalized.T)

        return similarity_matrix

    def _match_rank_1_only(
        self,
        hb_rooms: List[RoomEmbedding],
        ad_rooms: List[RoomEmbedding],
        similarity_matrix: np.ndarray
    ) -> Tuple[List[RoomMatch], List[UnmatchedRoom], List[UnmatchedRoom]]:
        """
        Match rooms - only rank 1 matches are returned.
        All rank 2+ and unmatched rooms go to unmatched lists.

        Returns:
            Tuple of (rank_1_matches, unmatched_hb_rooms, unmatched_ad_rooms)
        """
        rank_1_matches = []
        matched_hb_indices = set()
        matched_ad_indices = set()
        
        unmatched_hb_rooms = []
        unmatched_ad_rooms = []

        # For each Hotelbeds room, find best match
        for hb_idx, hb_room in enumerate(hb_rooms):
            similarities = similarity_matrix[hb_idx]

            # Get all candidates above threshold (not yet matched)
            candidate_indices = [
                (ad_idx, sim)
                for ad_idx, sim in enumerate(similarities)
                if sim >= self.similarity_threshold and ad_idx not in matched_ad_indices
            ]

            # Sort by similarity descending
            candidate_indices.sort(key=lambda x: x[1], reverse=True)

            if not candidate_indices:
                # No match found (below threshold or all already matched)
                max_sim = float(np.max(similarities))
                reason = "below_threshold" if max_sim < self.similarity_threshold else "no_match_found"
                
                unmatched_hb_rooms.append(UnmatchedRoom(
                    room_code=hb_room.room_code,
                    room_id=hb_room.room_id,
                    text=hb_room.text,
                    reason=reason
                ))
            else:
                # Take only rank 1 (best match)
                ad_idx, similarity = candidate_indices[0]
                ad_room = ad_rooms[ad_idx]

                match = RoomMatch(
                    similarity_score=float(similarity),
                    hotelbeds_room_code=hb_room.room_code,
                    hotelbeds_room_id=hb_room.room_id,
                    hotelbeds_text=hb_room.text,
                    axisdata_room_code=ad_room.room_code,
                    axisdata_room_id=ad_room.room_id,
                    axisdata_text=ad_room.text
                )
                rank_1_matches.append(match)

                # Mark as matched
                matched_hb_indices.add(hb_idx)
                matched_ad_indices.add(ad_idx)

        # All unmatched AxisData rooms
        for ad_idx, ad_room in enumerate(ad_rooms):
            if ad_idx not in matched_ad_indices:
                unmatched_ad_rooms.append(UnmatchedRoom(
                    room_code=ad_room.room_code,
                    room_id=ad_room.room_id,
                    text=ad_room.text,
                    reason="no_hotelbeds_match_found"
                ))

        return rank_1_matches, unmatched_hb_rooms, unmatched_ad_rooms


# ========================================================================
# CLASS 4: UNIFIED MATCHER (ORCHESTRATOR)
# ========================================================================


class UnifiedMatcher:
    """Orchestrates the complete matching pipeline."""

    def __init__(self):
        """Initialize matcher."""
        self.loader = EmbeddingsLoader()
        self.hotel_matcher = None
        self.room_matcher = None
        self.results: List[HotelPairMapping] = []

    def run_complete_matching(self) -> List[HotelPairMapping]:
        """
        Run the complete matching pipeline.

        Returns:
            List of HotelPairMapping objects
        """
        # Step 1: Load embeddings
        self.loader.load_all_embeddings()

        # Step 2: Match hotels by place_id
        self.hotel_matcher = HotelPairMatcher(
            self.loader.hotelbeds_hotels,
            self.loader.axisdata_hotels
        )
        hotel_pairs = self.hotel_matcher.match_by_place_id()

        # Step 3: Match rooms for each hotel pair
        logger.info("\n" + "=" * 80)
        logger.info("MATCHING ROOMS FOR EACH HOTEL PAIR (RANK 1 ONLY)")
        logger.info("=" * 80)

        self.room_matcher = RoomPairMatcher(
            self.loader.hotelbeds_rooms,
            self.loader.axisdata_rooms
        )

        total_room_matches = 0
        total_unmatched_hb = 0
        total_unmatched_ad = 0

        for idx, (hb_hotel, ad_hotel) in enumerate(hotel_pairs, 1):
            logger.info(f"\nPair {idx}/{len(hotel_pairs)}: {hb_hotel.hotel_code} ↔ {ad_hotel.hotel_code}")

            # Match rooms (only rank 1)
            room_matches, unmatched_hb, unmatched_ad = self.room_matcher.match_rooms_for_hotel_pair(
                hb_hotel,
                ad_hotel
            )

            total_room_matches += len(room_matches)
            total_unmatched_hb += len(unmatched_hb)
            total_unmatched_ad += len(unmatched_ad)

            logger.info(f"  ✓ Matched: {len(room_matches)} rooms (rank 1 only)")
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
        logger.info(f"✓ Total room matches (rank 1 only): {total_room_matches}")
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
    """
    Save mapping results to JSON.

    Args:
        results: List of HotelPairMapping objects
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate summary statistics
    total_room_matches = sum(len(mapping.room_matches) for mapping in results)

    similarities = [
        match['similarity_score']
        for mapping in results
        for match in mapping.room_matches
    ]

    summary = {
        "total_hotel_pairs": len(results),
        "total_room_matches": total_room_matches,
        "total_unmatched_rooms": {
            "hotelbeds": sum(len(m.unmatched_hotelbeds_rooms) for m in results),
            "axisdata": sum(len(m.unmatched_axisdata_rooms) for m in results)
        },
        "room_similarity_stats": {
            "mean": float(np.mean(similarities)) if similarities else 0.0,
            "median": float(np.median(similarities)) if similarities else 0.0,
            "min": float(np.min(similarities)) if similarities else 0.0,
            "max": float(np.max(similarities)) if similarities else 0.0,
        }
    }

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "similarity_threshold": ROOM_SIMILARITY_THRESHOLD,
            "matching_strategy": "rank_1_only",
            "model": "all-MiniLM-L6-v2",
            "embedding_dim": 384
        },
        "summary": summary,
        "hotel_pairs": [asdict(mapping) for mapping in results]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✓ Saved mapping results to {output_path}")


def print_sample_mappings(results: List[HotelPairMapping], sample_size: int = 2):
    """Print sample mappings for verification."""
    print("\n" + "=" * 80)
    print("SAMPLE HOTEL & ROOM MAPPINGS (RANK 1 ONLY)")
    print("=" * 80)

    for idx, mapping in enumerate(results[:sample_size], 1):
        print(f"\nHotel Pair #{idx}:")
        print(f"  Place ID: {mapping.place_id}")
        print(f"  Hotelbeds: {mapping.hotelbeds_hotel_code} ({mapping.hotelbeds_hotel_name})")
        print(f"    Total rooms: {mapping.hotelbeds_total_rooms}")
        print(f"  AxisData: {mapping.axisdata_hotel_code} ({mapping.axisdata_hotel_name})")
        print(f"    Total rooms: {mapping.axisdata_total_rooms}")

        print(f"\n  Room Matches (rank 1 only): {len(mapping.room_matches)}")
        for match in mapping.room_matches[:3]:
            print(f"    Similarity {match['similarity_score']:.3f}")
            print(f"      HB: {match['hotelbeds_room_code']} - {match['hotelbeds_text'][:50]}...")
            print(f"      AD: {match['axisdata_room_code']} - {match['axisdata_text'][:50]}...")

        print(f"\n  Unmatched Hotelbeds rooms: {len(mapping.unmatched_hotelbeds_rooms)}")
        for room in mapping.unmatched_hotelbeds_rooms[:3]:
            print(f"    {room['room_code']} - {room['text'][:50]}... (Reason: {room['reason']})")

        print(f"\n  Unmatched AxisData rooms: {len(mapping.unmatched_axisdata_rooms)}")
        for room in mapping.unmatched_axisdata_rooms[:3]:
            print(f"    {room['room_code']} - {room['text'][:50]}... (Reason: {room['reason']})")

    print("\n" + "=" * 80)


# ========================================================================
# MAIN EXECUTION
# ========================================================================


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("HOTEL & ROOM MATCHER (RANK 1 ONLY)")
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