# import logging
# import json
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import google.generativeai as genai
# from tqdm import tqdm
# import time

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

# # ========================================================================
# # CONFIGURATION
# # ========================================================================

# GOOGLE_API_KEY = ""

# # Input files
# HOTELBEDS_HOTELS_JSON = Path("data/outputs/hotelbed_norm_data.json")
# AXISDATA_HOTELS_JSON = Path("data/outputs/axisdata_norm_data.json")

# # Output files (Google embeddings)
# HOTELBEDS_HOTEL_EMBEDDINGS_OUT = Path("data/google_embedings/hotelbeds_hotel_embeddings_google.json")
# HOTELBEDS_ROOM_EMBEDDINGS_OUT = Path("data/google_embedings/hotelbeds_room_embeddings_google.json")
# AXISDATA_HOTEL_EMBEDDINGS_OUT = Path("data/google_embedings/axisdata_hotel_embeddings_google.json")
# AXISDATA_ROOM_EMBEDDINGS_OUT = Path("data/google_embedings/axisdata_room_embeddings_google.json")

# EMBEDDING_MODEL = "models/text-embedding-004"
# BATCH_SIZE = 30
# RATE_LIMIT_DELAY = 1.0

# # ✓ NEW: Flag to control whether to include occupancy in embeddings
# INCLUDE_OCCUPANCY_IN_EMBEDDING = True

# # ========================================================================
# # OCCUPANCY TEXT FORMATTER
# # ========================================================================

# class OccupancyFormatter:
#     """
#     Format occupancy data into human-readable text for embeddings.
#     Only includes common attributes that have valid values (not null).
#     """
    
#     # Common occupancy fields that both providers share
#     COMMON_FIELDS = [
#         'min_occupancy',
#         'max_occupancy',
#         'min_adults',
#         'max_adults',
#         'min_children',
#         'max_children',
#     ]
    
#     # Optional fields (only include if not null)
#     OPTIONAL_FIELDS = [
#         'max_child_age',
#         'min_infants',
#         'max_infants',
#         'max_infant_age',
#         'standard_occupancy',
#     ]
    
#     @staticmethod
#     def has_valid_value(value: Any) -> bool:
#         """Check if a value is valid (not None, not 0 for some fields)."""
#         return value is not None and value != ""
    
#     @staticmethod
#     def format_occupancy(occupancy: Optional[Dict[str, Any]]) -> str:
#         """
#         Convert occupancy dict to natural language text.
#         Only includes common fields with valid values.
        
#         Example output:
#         "accommodates 1 to 4 guests, 1 to 4 adults, up to 2 children"
#         "accommodates 2 to 6 guests, 2 to 6 adults"
        
#         Args:
#             occupancy: Occupancy dict from normalized data
            
#         Returns:
#             Formatted occupancy string
#         """
#         if not occupancy or not isinstance(occupancy, dict):
#             return ""
        
#         parts = []
        
#         # ✓ COMMON FIELD 1: Total occupancy (min/max guests)
#         min_occ = occupancy.get('min_occupancy')
#         max_occ = occupancy.get('max_occupancy')
        
#         if OccupancyFormatter.has_valid_value(min_occ) and OccupancyFormatter.has_valid_value(max_occ):
#             if min_occ == max_occ:
#                 parts.append(f"accommodates {max_occ} guests")
#             else:
#                 parts.append(f"accommodates {min_occ} to {max_occ} guests")
#         elif OccupancyFormatter.has_valid_value(max_occ):
#             parts.append(f"accommodates up to {max_occ} guests")
        
#         # ✓ COMMON FIELD 2: Adults
#         min_adults = occupancy.get('min_adults')
#         max_adults = occupancy.get('max_adults')
        
#         if OccupancyFormatter.has_valid_value(min_adults) and OccupancyFormatter.has_valid_value(max_adults):
#             if min_adults == max_adults:
#                 parts.append(f"{max_adults} adults")
#             else:
#                 parts.append(f"{min_adults} to {max_adults} adults")
#         elif OccupancyFormatter.has_valid_value(max_adults):
#             parts.append(f"up to {max_adults} adults")
        
#         # ✓ COMMON FIELD 3: Children (only if max_children > 0)
#         max_children = occupancy.get('max_children')
        
#         if OccupancyFormatter.has_valid_value(max_children) and max_children > 0:
#             parts.append(f"up to {max_children} children")
            
#             # ✓ OPTIONAL: Add child age only if available
#             max_child_age = occupancy.get('max_child_age')
#             if OccupancyFormatter.has_valid_value(max_child_age):
#                 # Modify last part to include age
#                 parts[-1] = f"up to {max_children} children (max age {max_child_age})"
        
#         # ✓ OPTIONAL: Infants (only include if both providers have it)
#         # Since Hotelbeds has null for infant fields, we skip this
#         max_infants = occupancy.get('max_infants')
#         if OccupancyFormatter.has_valid_value(max_infants) and max_infants > 0:
#             # Only include if not null
#             infant_text = f"up to {max_infants} infants"
#             max_infant_age = occupancy.get('max_infant_age')
#             if OccupancyFormatter.has_valid_value(max_infant_age):
#                 infant_text += f" (max age {max_infant_age})"
#             parts.append(infant_text)
        
#         # ✓ OPTIONAL: Standard occupancy (only if available)
#         standard_occ = occupancy.get('standard_occupancy')
#         if OccupancyFormatter.has_valid_value(standard_occ) and standard_occ > 0:
#             parts.append(f"standard occupancy {standard_occ}")
        
#         return ", ".join(parts) if parts else ""
    
#     @staticmethod
#     def get_common_occupancy_summary(occupancy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Extract only the common occupancy fields with valid values.
#         Used for storing clean occupancy data in output.
        
#         Returns:
#             Dict with only common fields that have valid values
#         """
#         if not occupancy or not isinstance(occupancy, dict):
#             return {}
        
#         summary = {}
        
#         # Add common fields
#         for field in OccupancyFormatter.COMMON_FIELDS:
#             value = occupancy.get(field)
#             if OccupancyFormatter.has_valid_value(value):
#                 summary[field] = value
        
#         # Add optional fields only if they have valid values
#         for field in OccupancyFormatter.OPTIONAL_FIELDS:
#             value = occupancy.get(field)
#             if OccupancyFormatter.has_valid_value(value):
#                 summary[field] = value
        
#         return summary

# # ========================================================================
# # GOOGLE EMBEDDINGS GENERATOR
# # ========================================================================

# class GoogleEmbeddingsGenerator:
#     """Generate embeddings using Google's text-embedding-004 model."""

#     def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
#         genai.configure(api_key=api_key)
#         self.model = model
#         logger.info(f"Initialized Google Embeddings Generator with model: {model}")

#     def embed_text(self, text: str) -> np.ndarray:
#         result = genai.embed_content(
#             model=self.model,
#             content=text,
#             task_type="semantic_similarity"
#         )
#         return np.array(result['embedding'])

#     def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
#         embeddings = []
#         for text in texts:
#             try:
#                 embedding = self.embed_text(text)
#                 embeddings.append(embedding)
#             except Exception as e:
#                 logger.error(f"Error embedding text: {text[:50]}... - {e}")
#                 embeddings.append(np.zeros(768))
#         return embeddings

# # ========================================================================
# # HELPER: LOAD HOTELS FROM NORMALIZED JSON
# # ========================================================================

# def load_hotels(hotels_json: Path) -> List[Dict]:
#     """Load hotels from normalized JSON file."""
#     with open(hotels_json, 'r') as f:
#         data = json.load(f)
    
#     # Extract hotels from "hotels" key
#     if isinstance(data, dict) and 'hotels' in data:
#         return data['hotels']
#     elif isinstance(data, list):
#         return data
#     else:
#         raise ValueError(f"Unexpected JSON structure. Expected 'hotels' key, got: {list(data.keys())}")

# # ========================================================================
# # HELPER: EXTRACT ROOMS FROM HOTEL FILE
# # ========================================================================

# def extract_rooms_from_hotels(hotels_json: Path, provider: str) -> List[Dict]:
#     """
#     Extract all rooms from hotels file with occupancy-enhanced text for embeddings.
    
#     ✓ UPDATED: Now includes only common occupancy fields with valid values.
#     """
#     hotels = load_hotels(hotels_json)
#     occupancy_formatter = OccupancyFormatter()
    
#     rooms = []
#     rooms_with_occupancy = 0
#     rooms_with_valid_occupancy = 0
    
#     for hotel in hotels:
#         hotel_code = hotel.get("hotel_code")
#         if not hotel_code:
#             logger.warning(f"Hotel missing hotel_code: {hotel.get('hotel_name', 'Unknown')}")
#             continue
            
#         for room in hotel.get("rooms", []):
#             room_entry = dict(room)
#             room_entry["hotel_code"] = hotel_code
            
#             # Extract base description text
#             if provider == "hotelbeds":
#                 # Hotelbeds: use normalized_code_description
#                 base_text = room.get("normalized_code_description") or room.get("code_description") or ""
#                 room_entry["room_id"] = room.get("room_id", "")
#                 room_entry["room_code"] = room.get("room_code", "")
#             else:  # axisdata
#                 # AxisData: use normalized_description
#                 base_text = room.get("normalized_description") or room.get("room_description") or room.get("room_name") or ""
#                 room_entry["room_id"] = room.get("room_id", "")
#                 room_entry["room_code"] = room.get("room_code", "")
            
#             base_text = base_text.lower() if base_text else ""
            
#             # ✓ NEW: Add occupancy information to embedding text (only common fields)
#             occupancy = room.get("occupancy")
#             occupancy_text = ""
#             common_occupancy = {}
            
#             if occupancy:
#                 rooms_with_occupancy += 1
            
#             if INCLUDE_OCCUPANCY_IN_EMBEDDING and occupancy:
#                 occupancy_text = occupancy_formatter.format_occupancy(occupancy)
#                 common_occupancy = occupancy_formatter.get_common_occupancy_summary(occupancy)
                
#                 if occupancy_text:
#                     rooms_with_valid_occupancy += 1
            
#             # Combine base text with occupancy text
#             if occupancy_text:
#                 text_for_embedding = f"{base_text}. {occupancy_text}"
#             else:
#                 text_for_embedding = base_text
            
#             room_entry["text_for_embedding"] = text_for_embedding
#             room_entry["base_description"] = base_text
#             room_entry["occupancy_text"] = occupancy_text
#             room_entry["occupancy_common"] = common_occupancy  # ✓ NEW: Only common fields
            
#             # Skip rooms without any text
#             if not text_for_embedding:
#                 logger.warning(f"Room missing text in {hotel_code}: {room.get('room_id', 'Unknown')}")
#                 continue
                
#             rooms.append(room_entry)
    
#     logger.info(
#         f"  Extracted {len(rooms)} rooms from {provider}. "
#         f"Rooms with occupancy: {rooms_with_occupancy}, "
#         f"with valid common occupancy: {rooms_with_valid_occupancy}"
#     )
    
#     return rooms

# # ========================================================================
# # HOTEL EMBEDDINGS
# # ========================================================================

# def generate_hotel_embeddings(
#     input_path: Path, 
#     output_path: Path, 
#     provider: str, 
#     generator: GoogleEmbeddingsGenerator
# ):
#     """Generate hotel embeddings."""
#     logger.info(f"\nGenerating {provider} hotel embeddings...")
#     logger.info(f"  Input: {input_path}")
#     logger.info(f"  Output: {output_path}")

#     hotels = load_hotels(input_path)
#     logger.info(f"  Loaded {len(hotels)} hotels")

#     embeddings_data = []

#     for i in tqdm(range(0, len(hotels), BATCH_SIZE), desc=f"{provider} hotels"):
#         batch = hotels[i:i+BATCH_SIZE]
        
#         # Use "hotel_name" field from normalized data
#         texts = [hotel['hotel_name'].lower() for hotel in batch]
        
#         batch_embeddings = generator.embed_batch(texts)
        
#         for hotel, embedding in zip(batch, batch_embeddings):
#             embeddings_data.append({
#                 "hotel_code": hotel['hotel_code'],
#                 "place_id": hotel['place_id'],
#                 "hotel_name": hotel['hotel_name'],
#                 "embedding": embedding.tolist()
#             })
        
#         if i + BATCH_SIZE < len(hotels):
#             time.sleep(RATE_LIMIT_DELAY)

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_data = {
#         "metadata": {
#             "model": EMBEDDING_MODEL,
#             "provider": provider,
#             "total_count": len(embeddings_data),
#             "embedding_dim": 768
#         },
#         "embeddings": embeddings_data
#     }
    
#     with open(output_path, 'w') as f:
#         json.dump(output_data, f, indent=2)
    
#     logger.info(f"  ✓ Saved {len(embeddings_data)} hotel embeddings")

# # ========================================================================
# # ROOM EMBEDDINGS
# # ========================================================================

# def generate_room_embeddings(
#     input_hotels_path: Path, 
#     output_path: Path, 
#     provider: str, 
#     generator: GoogleEmbeddingsGenerator
# ):
#     """
#     Generate room embeddings from hotels file.
    
#     ✓ UPDATED: Now includes only common occupancy fields in embeddings and metadata.
#     """
#     logger.info(f"\nGenerating {provider} room embeddings...")
#     logger.info(f"  Input: {input_hotels_path}")
#     logger.info(f"  Output: {output_path}")
#     logger.info(f"  Include occupancy: {INCLUDE_OCCUPANCY_IN_EMBEDDING}")

#     rooms = extract_rooms_from_hotels(input_hotels_path, provider)
#     logger.info(f"  Extracted {len(rooms)} rooms")

#     embeddings_data = []

#     for i in tqdm(range(0, len(rooms), BATCH_SIZE), desc=f"{provider} rooms"):
#         batch = rooms[i:i+BATCH_SIZE]
        
#         texts = [room['text_for_embedding'] for room in batch]
#         batch_embeddings = generator.embed_batch(texts)
        
#         for room, embedding in zip(batch, batch_embeddings):
#             # ✓ UPDATED: Include only common occupancy fields
#             embeddings_data.append({
#                 "room_id": room['room_id'],
#                 "hotel_code": room['hotel_code'],
#                 "room_code": room['room_code'],
#                 "text_for_embedding": room['text_for_embedding'],
#                 "base_description": room['base_description'],
#                 "occupancy_text": room['occupancy_text'],
#                 "occupancy": room['occupancy_common'],  # ✓ UPDATED: Only common fields
#                 "embedding": embedding.tolist()
#             })
        
#         if i + BATCH_SIZE < len(rooms):
#             time.sleep(RATE_LIMIT_DELAY)

#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     # Calculate statistics
#     rooms_with_occupancy = sum(1 for room in embeddings_data if room['occupancy'])
    
#     output_data = {
#         "metadata": {
#             "model": EMBEDDING_MODEL,
#             "provider": provider,
#             "total_count": len(embeddings_data),
#             "rooms_with_occupancy": rooms_with_occupancy,
#             "embedding_dim": 768,
#             "includes_occupancy": INCLUDE_OCCUPANCY_IN_EMBEDDING,
#             "occupancy_fields": "common_only",  # ✓ NEW: Indicator
#             "common_occupancy_fields": OccupancyFormatter.COMMON_FIELDS,  # ✓ NEW
#             "optional_occupancy_fields": OccupancyFormatter.OPTIONAL_FIELDS  # ✓ NEW
#         },
#         "embeddings": embeddings_data
#     }
    
#     with open(output_path, 'w') as f:
#         json.dump(output_data, f, indent=2)
    
#     logger.info(f"  ✓ Saved {len(embeddings_data)} room embeddings")
#     logger.info(f"  ✓ {rooms_with_occupancy} rooms have common occupancy data")

# # ========================================================================
# # MAIN EXECUTION
# # ========================================================================

# def main():
#     """Main execution."""
#     logger.info("=" * 80)
#     logger.info("GOOGLE EMBEDDINGS GENERATOR (COMMON OCCUPANCY FIELDS ONLY)")
#     logger.info("=" * 80)
#     logger.info(f"Occupancy inclusion: {INCLUDE_OCCUPANCY_IN_EMBEDDING}")
#     logger.info(f"Common fields: {OccupancyFormatter.COMMON_FIELDS}")
#     logger.info(f"Optional fields: {OccupancyFormatter.OPTIONAL_FIELDS}")

#     try:
#         generator = GoogleEmbeddingsGenerator(api_key=GOOGLE_API_KEY)
        
#         # Generate hotel embeddings
#         generate_hotel_embeddings(
#             HOTELBEDS_HOTELS_JSON,
#             HOTELBEDS_HOTEL_EMBEDDINGS_OUT,
#             "hotelbeds",
#             generator
#         )
        
#         generate_hotel_embeddings(
#             AXISDATA_HOTELS_JSON,
#             AXISDATA_HOTEL_EMBEDDINGS_OUT,
#             "axisdata",
#             generator
#         )
        
#         # Generate room embeddings (with common occupancy fields only)
#         generate_room_embeddings(
#             HOTELBEDS_HOTELS_JSON,
#             HOTELBEDS_ROOM_EMBEDDINGS_OUT,
#             "hotelbeds",
#             generator
#         )
        
#         generate_room_embeddings(
#             AXISDATA_HOTELS_JSON,
#             AXISDATA_ROOM_EMBEDDINGS_OUT,
#             "axisdata",
#             generator
#         )
        
#         logger.info("\n" + "=" * 80)
#         logger.info("✓ ALL EMBEDDINGS GENERATED SUCCESSFULLY!")
#         logger.info("=" * 80)
#         logger.info("\nGenerated embeddings include:")
#         logger.info("  - Room descriptions (normalized)")
#         logger.info("  - Common occupancy constraints:")
#         logger.info("    * min/max occupancy (guests)")
#         logger.info("    * min/max adults")
#         logger.info("    * min/max children")
#         logger.info("  - Optional fields (only if not null):")
#         logger.info("    * max_child_age")
#         logger.info("    * min/max_infants (skipped if null)")
#         logger.info("    * standard_occupancy (skipped if null)")
#         logger.info("\nNext steps:")
#         logger.info("1. Verify the generated embedding files")
#         logger.info("2. Check that occupancy_text only includes common fields")
#         logger.info("3. Run room_matcher_google.py to perform matching")
        
#         return 0

#     except Exception as e:
#         logger.error(f"FATAL ERROR: {e}", exc_info=True)
#         return 1

# if __name__ == "__main__":
#     import sys
#     sys.exit(main())
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ========================================================================
# CONFIGURATION
# ========================================================================

GOOGLE_API_KEY = ""

HOTELBEDS_HOTELS_JSON = Path("data/outputs/hotelbed_norm_data.json")
AXISDATA_HOTELS_JSON = Path("data/outputs/axisdata_norm_data.json")

HOTELBEDS_HOTEL_EMBEDDINGS_OUT = Path("data/google_embedings/hotelbeds_hotel_embeddings_google.json")
HOTELBEDS_ROOM_EMBEDDINGS_OUT = Path("data/google_embedings/hotelbeds_room_embeddings_google.json")
AXISDATA_HOTEL_EMBEDDINGS_OUT = Path("data/google_embedings/axisdata_hotel_embeddings_google.json")
AXISDATA_ROOM_EMBEDDINGS_OUT = Path("data/google_embedings/axisdata_room_embeddings_google.json")

EMBEDDING_MODEL = "models/text-embedding-004"
BATCH_SIZE = 30
RATE_LIMIT_DELAY = 1.0

INCLUDE_OCCUPANCY_IN_EMBEDDING = True

class OccupancyFormatter:
    COMMON_FIELDS = [
        'min_occupancy',
        'max_occupancy',
        'min_adults',
        'max_adults',
        'min_children',
        'max_children',
    ]
    OPTIONAL_FIELDS = [
        'max_child_age',
        'min_infants',
        'max_infants',
        'max_infant_age',
        'standard_occupancy',
    ]
    @staticmethod
    def has_valid_value(value: Any) -> bool:
        return value is not None and value != ""
    @staticmethod
    def format_occupancy(occupancy: Optional[Dict[str, Any]]) -> str:
        if not occupancy or not isinstance(occupancy, dict):
            return ""
        parts = []
        min_occ = occupancy.get('min_occupancy')
        max_occ = occupancy.get('max_occupancy')
        if OccupancyFormatter.has_valid_value(min_occ) and OccupancyFormatter.has_valid_value(max_occ):
            if min_occ == max_occ:
                parts.append(f"accommodates {max_occ} guests")
            else:
                parts.append(f"accommodates {min_occ} to {max_occ} guests")
        elif OccupancyFormatter.has_valid_value(max_occ):
            parts.append(f"accommodates up to {max_occ} guests")
        min_adults = occupancy.get('min_adults')
        max_adults = occupancy.get('max_adults')
        if OccupancyFormatter.has_valid_value(min_adults) and OccupancyFormatter.has_valid_value(max_adults):
            if min_adults == max_adults:
                parts.append(f"{max_adults} adults")
            else:
                parts.append(f"{min_adults} to {max_adults} adults")
        elif OccupancyFormatter.has_valid_value(max_adults):
            parts.append(f"up to {max_adults} adults")
        max_children = occupancy.get('max_children')
        if OccupancyFormatter.has_valid_value(max_children) and max_children > 0:
            parts.append(f"up to {max_children} children")
            max_child_age = occupancy.get('max_child_age')
            if OccupancyFormatter.has_valid_value(max_child_age):
                parts[-1] = f"up to {max_children} children (max age {max_child_age})"
        max_infants = occupancy.get('max_infants')
        if OccupancyFormatter.has_valid_value(max_infants) and max_infants > 0:
            infant_text = f"up to {max_infants} infants"
            max_infant_age = occupancy.get('max_infant_age')
            if OccupancyFormatter.has_valid_value(max_infant_age):
                infant_text += f" (max age {max_infant_age})"
            parts.append(infant_text)
        standard_occ = occupancy.get('standard_occupancy')
        if OccupancyFormatter.has_valid_value(standard_occ) and standard_occ > 0:
            parts.append(f"standard occupancy {standard_occ}")
        return ", ".join(parts) if parts else ""
    @staticmethod
    def get_common_occupancy_summary(occupancy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not occupancy or not isinstance(occupancy, dict):
            return {}
        summary = {}
        for field in OccupancyFormatter.COMMON_FIELDS:
            value = occupancy.get(field)
            if OccupancyFormatter.has_valid_value(value):
                summary[field] = value
        for field in OccupancyFormatter.OPTIONAL_FIELDS:
            value = occupancy.get(field)
            if OccupancyFormatter.has_valid_value(value):
                summary[field] = value
        return summary

# ========================================================================
# GOOGLE EMBEDDINGS GENERATOR
# ========================================================================

class GoogleEmbeddingsGenerator:
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        genai.configure(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Google Embeddings Generator with model: {model}")

    def embed_text(self, text: str) -> np.ndarray:
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="semantic_similarity"
        )
        return np.array(result['embedding'])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error embedding text: {text[:50]}... - {e}")
                embeddings.append(np.zeros(768))
        return embeddings

# ========================================================================
# HELPER: LOAD HOTELS FROM NORMALIZED JSON
# ========================================================================

def load_hotels(hotels_json: Path) -> List[Dict]:
    with open(hotels_json, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'hotels' in data:
        return data['hotels']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON structure. Expected 'hotels' key, got: {list(data.keys())}")

# ========================================================================
# HELPER: EXTRACT ROOMS FROM HOTEL FILE
# ========================================================================

def extract_rooms_from_hotels(hotels_json: Path, provider: str) -> List[Dict]:
    hotels = load_hotels(hotels_json)
    occupancy_formatter = OccupancyFormatter()
    rooms = []
    rooms_with_occupancy = 0
    rooms_with_valid_occupancy = 0
    for hotel in hotels:
        hotel_code = hotel.get("hotel_code")
        if not hotel_code:
            logger.warning(f"Hotel missing hotel_code: {hotel.get('hotel_name', 'Unknown')}")
            continue
        for room in hotel.get("rooms", []):
            room_entry = dict(room)
            room_entry["hotel_code"] = hotel_code
            if provider == "hotelbeds":
                base_text = room.get("normalized_code_description") or room.get("code_description") or ""
                room_entry["room_id"] = room.get("room_id", "")
                room_entry["room_code"] = room.get("room_code", "")
            else:
                base_text = room.get("normalized_description") or room.get("room_description") or room.get("room_name") or ""
                room_entry["room_id"] = room.get("room_id", "")
                room_entry["room_code"] = room.get("room_code", "")
            base_text = base_text.lower() if base_text else ""
            occupancy = room.get("occupancy")
            occupancy_text = ""
            common_occupancy = {}
            if occupancy:
                rooms_with_occupancy += 1
            if INCLUDE_OCCUPANCY_IN_EMBEDDING and occupancy:
                occupancy_text = occupancy_formatter.format_occupancy(occupancy)
                common_occupancy = occupancy_formatter.get_common_occupancy_summary(occupancy)
                if occupancy_text:
                    rooms_with_valid_occupancy += 1
            if occupancy_text:
                text_for_embedding = f"{base_text}. {occupancy_text}"
            else:
                text_for_embedding = base_text
            room_entry["text_for_embedding"] = text_for_embedding
            room_entry["base_description"] = base_text
            room_entry["occupancy_text"] = occupancy_text
            room_entry["occupancy_common"] = common_occupancy
            if not text_for_embedding:
                logger.warning(f"Room missing text in {hotel_code}: {room.get('room_id', 'Unknown')}")
                continue
            rooms.append(room_entry)
    logger.info(
        f"  Extracted {len(rooms)} rooms from {provider}. "
        f"Rooms with occupancy: {rooms_with_occupancy}, "
        f"with valid common occupancy: {rooms_with_valid_occupancy}"
    )
    return rooms

# ========================================================================
# HOTEL EMBEDDINGS
# ========================================================================

def generate_hotel_embeddings(
    input_path: Path, 
    output_path: Path, 
    provider: str, 
    generator: GoogleEmbeddingsGenerator
):
    logger.info(f"\nGenerating {provider} hotel embeddings...")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    hotels = load_hotels(input_path)
    logger.info(f"  Loaded {len(hotels)} hotels")
    embeddings_data = []
    for i in tqdm(range(0, len(hotels), BATCH_SIZE), desc=f"{provider} hotels"):
        batch = hotels[i:i+BATCH_SIZE]
        texts = [hotel['hotel_name'].lower() for hotel in batch]
        # --- Count characters for batch ---
        char_counts = [len(t) for t in texts]
        total_chars = sum(char_counts)
        logger.info(f"Batch {i//BATCH_SIZE+1}: {len(batch)} hotels, Total chars: {total_chars}, Avg chars/hotel: {total_chars / len(batch):.2f}")
        batch_embeddings = generator.embed_batch(texts)
        for hotel, embedding in zip(batch, batch_embeddings):
            embeddings_data.append({
                "hotel_code": hotel['hotel_code'],
                "place_id": hotel['place_id'],
                "hotel_name": hotel['hotel_name'],
                "embedding": embedding.tolist()
            })
        if i + BATCH_SIZE < len(hotels):
            time.sleep(RATE_LIMIT_DELAY)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "model": EMBEDDING_MODEL,
            "provider": provider,
            "total_count": len(embeddings_data),
            "embedding_dim": 768
        },
        "embeddings": embeddings_data
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"  ✓ Saved {len(embeddings_data)} hotel embeddings")

# ========================================================================
# ROOM EMBEDDINGS
# ========================================================================

def generate_room_embeddings(
    input_hotels_path: Path, 
    output_path: Path, 
    provider: str, 
    generator: GoogleEmbeddingsGenerator
):
    logger.info(f"\nGenerating {provider} room embeddings...")
    logger.info(f"  Input: {input_hotels_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Include occupancy: {INCLUDE_OCCUPANCY_IN_EMBEDDING}")
    rooms = extract_rooms_from_hotels(input_hotels_path, provider)
    logger.info(f"  Extracted {len(rooms)} rooms")
    embeddings_data = []
    for i in tqdm(range(0, len(rooms), BATCH_SIZE), desc=f"{provider} rooms"):
        batch = rooms[i:i+BATCH_SIZE]
        texts = [room['text_for_embedding'] for room in batch]
        # --- Count characters for batch ---
        char_counts = [len(t) for t in texts]
        total_chars = sum(char_counts)
        logger.info(f"Batch {i//BATCH_SIZE+1}: {len(batch)} rooms, Total chars: {total_chars}, Avg chars/room: {total_chars / len(batch):.2f}")
        batch_embeddings = generator.embed_batch(texts)
        for room, embedding in zip(batch, batch_embeddings):
            embeddings_data.append({
                "room_id": room['room_id'],
                "hotel_code": room['hotel_code'],
                "room_code": room['room_code'],
                "text_for_embedding": room['text_for_embedding'],
                "base_description": room['base_description'],
                "occupancy_text": room['occupancy_text'],
                "occupancy": room['occupancy_common'],
                "embedding": embedding.tolist()
            })
        if i + BATCH_SIZE < len(rooms):
            time.sleep(RATE_LIMIT_DELAY)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rooms_with_occupancy = sum(1 for room in embeddings_data if room['occupancy'])
    output_data = {
        "metadata": {
            "model": EMBEDDING_MODEL,
            "provider": provider,
            "total_count": len(embeddings_data),
            "rooms_with_occupancy": rooms_with_occupancy,
            "embedding_dim": 768,
            "includes_occupancy": INCLUDE_OCCUPANCY_IN_EMBEDDING,
            "occupancy_fields": "common_only",
            "common_occupancy_fields": OccupancyFormatter.COMMON_FIELDS,
            "optional_occupancy_fields": OccupancyFormatter.OPTIONAL_FIELDS
        },
        "embeddings": embeddings_data
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"  ✓ Saved {len(embeddings_data)} room embeddings")
    logger.info(f"  ✓ {rooms_with_occupancy} rooms have common occupancy data")

# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    logger.info("=" * 80)
    logger.info("GOOGLE EMBEDDINGS GENERATOR (COMMON OCCUPANCY FIELDS ONLY)")
    logger.info("=" * 80)
    logger.info(f"Occupancy inclusion: {INCLUDE_OCCUPANCY_IN_EMBEDDING}")
    logger.info(f"Common fields: {OccupancyFormatter.COMMON_FIELDS}")
    logger.info(f"Optional fields: {OccupancyFormatter.OPTIONAL_FIELDS}")
    try:
        generator = GoogleEmbeddingsGenerator(api_key=GOOGLE_API_KEY)
        # Generate hotel embeddings
        generate_hotel_embeddings(
            HOTELBEDS_HOTELS_JSON,
            HOTELBEDS_HOTEL_EMBEDDINGS_OUT,
            "hotelbeds",
            generator
        )
        generate_hotel_embeddings(
            AXISDATA_HOTELS_JSON,
            AXISDATA_HOTEL_EMBEDDINGS_OUT,
            "axisdata",
            generator
        )
        # Generate room embeddings (with common occupancy fields only)
        generate_room_embeddings(
            HOTELBEDS_HOTELS_JSON,
            HOTELBEDS_ROOM_EMBEDDINGS_OUT,
            "hotelbeds",
            generator
        )
        generate_room_embeddings(
            AXISDATA_HOTELS_JSON,
            AXISDATA_ROOM_EMBEDDINGS_OUT,
            "axisdata",
            generator
        )
        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL EMBEDDINGS GENERATED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nGenerated embeddings include:")
        logger.info("  - Room descriptions (normalized)")
        logger.info("  - Common occupancy constraints:")
        logger.info("    * min/max occupancy (guests)")
        logger.info("    * min/max adults")
        logger.info("    * min/max children")
        logger.info("  - Optional fields (only if not null):")
        logger.info("    * max_child_age")
        logger.info("    * min/max_infants (skipped if null)")
        logger.info("    * standard_occupancy (skipped if null)")
        logger.info("\nNext steps:")
        logger.info("1. Verify the generated embedding files")
        logger.info("2. Check that occupancy_text only includes common fields")
        logger.info("3. Run room_matcher_google.py to perform matching")
        return 0
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
