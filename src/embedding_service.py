
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import warnings
import torch

warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ========================================================================
# CONFIGURATION
# ========================================================================

# Input normalized data files
NORMALIZED_HOTELBEDS_PATH = Path("data/outputs/hotelbed_norm_data.json")
NORMALIZED_AXISDATA_PATH = Path("data/outputs/axisdata_norm_data.json")

# Output embeddings directory
EMBEDDINGS_OUTPUT_DIR = Path("data/embeddings")

# Model name
MODEL_NAME = "all-MiniLM-L6-v2"

# Batch size for processing
BATCH_SIZE = 32

# ========================================================================
# AUTO DEVICE DETECTION
# ========================================================================


def get_device() -> str:
    """
    Auto-detect available device (CUDA or CPU).
    
    Returns:
        Device string: "cuda" if GPU available, "cpu" otherwise
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("⚠ No CUDA GPU detected, using CPU")
    
    return device


# ========================================================================
# DATA MODELS
# ========================================================================


@dataclass
class HotelEmbedding:
    """Hotel with its embedding."""
    hotel_code: str
    place_id: str
    hotel_name: str
    provider: str  # 'hotelbeds' or 'axisdata'
    text_for_embedding: str  # The normalized text used for embedding
    embedding: List[float]  # Embedding vector (stored as list for JSON serialization)


@dataclass
class RoomEmbedding:
    """Room with its embedding."""
    room_id: str
    hotel_code: str
    room_code: str
    provider: str  # 'hotelbeds' or 'axisdata'
    text_for_embedding: str  # The normalized text used for embedding
    embedding: List[float]  # Embedding vector (stored as list for JSON serialization)


@dataclass
class EmbeddingsOutput:
    """Complete embeddings output structure."""
    provider: str
    hotel_count: int
    total_room_count: int
    hotel_embeddings: List[Dict[str, Any]]
    room_embeddings: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ========================================================================
# EMBEDDING SERVICE
# ========================================================================


class EmbeddingService:
    """Service for generating semantic embeddings."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        device: Optional[str] = None,
    ):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the SentenceTransformer model
            batch_size: Batch size for processing
            device: Device to use ("cuda" or "cpu"), auto-detect if None
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts using batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            2D numpy array of embeddings (n_samples, embedding_dim)
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return np.array([])
        
        logger.info(f"Embedding {len(texts)} texts (batch size: {self.batch_size})")
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_size_actual = len(batch)
            
            try:
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=batch_size_actual,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
                embeddings.append(batch_embeddings)
                logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            except Exception as e:
                logger.error(f"Error processing batch at index {i}: {e}")
                embeddings.append(np.zeros((batch_size_actual, self.embedding_dim)))
        
        # Concatenate all batch embeddings
        all_embeddings = np.vstack(embeddings) if embeddings else np.array([])
        logger.info(f"✓ Generated {len(all_embeddings)} embeddings")
        
        return all_embeddings


# ========================================================================
# NORMALIZED DATA LOADER
# ========================================================================


class NormalizedDataLoader:
    """Loads normalized hotel and room data from JSON."""

    @staticmethod
    def load_normalized_data(file_path: Path) -> List[Dict[str, Any]]:
        """
        Load normalized data from JSON file.
        
        Expected format:
        - Top-level: list of hotels OR dict with 'hotels' key
        - Each hotel: dict with hotel_code, place_id, hotel_name, rooms, etc.
        
        Args:
            file_path: Path to normalized JSON file
            
        Returns:
            List of hotel dictionaries
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Normalized data file not found: {file_path}")
        
        logger.info(f"Loading normalized data from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different data structures
            if isinstance(data, list):
                hotels = data
            elif isinstance(data, dict):
                # Try common keys
                if 'hotels' in data:
                    hotels = data['hotels']
                elif 'data' in data:
                    hotels = data['data']
                else:
                    # If dict has hotel_code at top level, wrap it in a list
                    hotels = [data]
            else:
                raise ValueError(f"Unexpected data type: {type(data)}")
            
            logger.info(f"✓ Loaded {len(hotels)} hotels from {file_path}")
            return hotels
            
        except Exception as e:
            logger.error(f"Error loading normalized data: {e}")
            raise


# ========================================================================
# EMBEDDINGS GENERATOR
# ========================================================================


class HotelRoomEmbeddingsGenerator:
    """Generates embeddings for normalized hotel and room data."""

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize embeddings generator.
        
        Args:
            embedding_service: EmbeddingService instance
        """
        self.embedding_service = embedding_service

    def generate_embeddings(
        self,
        normalized_hotels: List[Dict[str, Any]],
        provider: str,
    ) -> EmbeddingsOutput:
        """
        Generate embeddings for all hotels and rooms in normalized data.
        
        Args:
            normalized_hotels: List of normalized hotel dicts
            provider: Provider name ('hotelbeds' or 'axisdata')
            
        Returns:
            EmbeddingsOutput object with hotel and room embeddings
        """
        logger.info(f"Generating embeddings for {provider}...")
        
        hotel_embeddings_list = []
        room_embeddings_list = []
        
        # Extract hotel names and room texts
        hotel_texts = []
        hotel_metadata = []
        
        room_texts = []
        room_metadata = []
        
        # Process each hotel
        total_rooms = 0
        for hotel_data in normalized_hotels:
            # Handle both object and dict formats
            if hasattr(hotel_data, '__dict__'):
                # It's an object (NormalizedHotel)
                hotel_code = hotel_data.hotel_code
                place_id = hotel_data.place_id
                hotel_name = hotel_data.hotel_name
                normalized_hotel_name = hotel_data.normalized_hotel_name
                rooms = hotel_data.rooms
            else:
                # It's a dict
                hotel_code = hotel_data.get('hotel_code', '')
                place_id = hotel_data.get('place_id', '')
                hotel_name = hotel_data.get('hotel_name', '')
                normalized_hotel_name = hotel_data.get('normalized_hotel_name', hotel_name)
                rooms = hotel_data.get('rooms', [])
            
            # Collect hotel text for embedding
            hotel_texts.append(normalized_hotel_name)
            hotel_metadata.append({
                'hotel_code': hotel_code,
                'place_id': place_id,
                'hotel_name': hotel_name,
                'provider': provider,
            })
            
            # Process rooms for this hotel
            total_rooms += len(rooms)
            
            for room_data in rooms:
                # Handle both object and dict formats
                if hasattr(room_data, '__dict__'):
                    # It's an object (NormalizedRoom)
                    room_id = room_data.room_id
                    room_code = room_data.room_code
                    text = (
                        room_data.normalized_code_description or
                        room_data.normalized_description or
                        ''
                    )
                else:
                    # It's a dict
                    room_id = room_data.get('room_id', '')
                    room_code = room_data.get('room_code', '')
                    text = (
                        room_data.get('normalized_code_description') or
                        room_data.get('normalized_description') or
                        ''
                    )
                
                if not text:
                    logger.warning(
                        f"No text for room {room_code} in hotel {hotel_code}"
                    )
                    continue
                
                room_texts.append(text)
                room_metadata.append({
                    'room_id': room_id,
                    'hotel_code': hotel_code,
                    'room_code': room_code,
                    'provider': provider,
                })
        
        logger.info(f"Total hotels: {len(hotel_texts)}, Total rooms: {total_rooms}")
        
        # Generate embeddings for hotels
        if hotel_texts:
            logger.info(f"Embedding {len(hotel_texts)} hotels...")
            hotel_embeddings_np = self.embedding_service.embed_texts(hotel_texts)
            
            for idx, (embedding, metadata) in enumerate(zip(hotel_embeddings_np, hotel_metadata)):
                hotel_emb = HotelEmbedding(
                    hotel_code=metadata['hotel_code'],
                    place_id=metadata['place_id'],
                    hotel_name=metadata['hotel_name'],
                    provider=provider,
                    text_for_embedding=hotel_texts[idx],
                    embedding=embedding.tolist(),  # Convert to list for JSON
                )
                hotel_embeddings_list.append(asdict(hotel_emb))
        
        # Generate embeddings for rooms
        if room_texts:
            logger.info(f"Embedding {len(room_texts)} rooms...")
            room_embeddings_np = self.embedding_service.embed_texts(room_texts)
            
            for idx, (embedding, metadata) in enumerate(zip(room_embeddings_np, room_metadata)):
                room_emb = RoomEmbedding(
                    room_id=metadata['room_id'],
                    hotel_code=metadata['hotel_code'],
                    room_code=metadata['room_code'],
                    provider=metadata['provider'],
                    text_for_embedding=room_texts[idx],
                    embedding=embedding.tolist(),  # Convert to list for JSON
                )
                room_embeddings_list.append(asdict(room_emb))
        
        logger.info(f"✓ Generated embeddings for {len(hotel_embeddings_list)} hotels and {len(room_embeddings_list)} rooms")
        
        # Create output structure
        output = EmbeddingsOutput(
            provider=provider,
            hotel_count=len(hotel_embeddings_list),
            total_room_count=len(room_embeddings_list),
            hotel_embeddings=hotel_embeddings_list,
            room_embeddings=room_embeddings_list,
            metadata={
                'model_name': self.embedding_service.model_name,
                'embedding_dim': self.embedding_service.embedding_dim,
                'device': self.embedding_service.device,
            }
        )
        
        return output


# ========================================================================
# EMBEDDINGS PERSISTENCE
# ========================================================================


class EmbeddingsPersistence:
    """Handles saving and loading embeddings."""

    @staticmethod
    def save_embeddings(
        embeddings_output: EmbeddingsOutput,
        output_dir: Path,
    ) -> Tuple[Path, Path]:
        """
        Save embeddings to JSON files.
        
        Args:
            embeddings_output: EmbeddingsOutput object
            output_dir: Output directory
            
        Returns:
            Tuple of (hotel_embeddings_path, room_embeddings_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        provider = embeddings_output.provider
        
        # Save hotel embeddings
        hotel_embeddings_path = output_dir / f"{provider}_hotel_embeddings.json"
        hotel_data = {
            'provider': embeddings_output.provider,
            'hotel_count': embeddings_output.hotel_count,
            'embeddings': embeddings_output.hotel_embeddings,
            'metadata': embeddings_output.metadata,
        }
        with open(hotel_embeddings_path, 'w') as f:
            json.dump(hotel_data, f, indent=2)
        logger.info(f"✓ Saved hotel embeddings to {hotel_embeddings_path}")
        
        # Save room embeddings
        room_embeddings_path = output_dir / f"{provider}_room_embeddings.json"
        room_data = {
            'provider': embeddings_output.provider,
            'total_room_count': embeddings_output.total_room_count,
            'embeddings': embeddings_output.room_embeddings,
            'metadata': embeddings_output.metadata,
        }
        with open(room_embeddings_path, 'w') as f:
            json.dump(room_data, f, indent=2)
        logger.info(f"✓ Saved room embeddings to {room_embeddings_path}")
        
        return hotel_embeddings_path, room_embeddings_path

    @staticmethod
    def load_hotel_embeddings(file_path: Path) -> Dict[str, Any]:
        """
        Load hotel embeddings from JSON file.
        
        Args:
            file_path: Path to hotel embeddings JSON file
            
        Returns:
            Loaded embeddings data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Hotel embeddings file not found: {file_path}")
        
        logger.info(f"Loading hotel embeddings from: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"✓ Loaded {len(data['embeddings'])} hotel embeddings")
        return data

    @staticmethod
    def load_room_embeddings(file_path: Path) -> Dict[str, Any]:
        """
        Load room embeddings from JSON file.
        
        Args:
            file_path: Path to room embeddings JSON file
            
        Returns:
            Loaded embeddings data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Room embeddings file not found: {file_path}")
        
        logger.info(f"Loading room embeddings from: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"✓ Loaded {len(data['embeddings'])} room embeddings")
        return data


# ========================================================================
# MAIN EXECUTION
# ========================================================================


def main():
    """
    Main execution: Generate embeddings for both providers.
    
    Input files:
    - data/outputs/hotelbed_norm_data.json
    - data/outputs/axisdata_norm_data.json
    
    Output files:
    - data/embeddings/hotelbeds_hotel_embeddings.json
    - data/embeddings/hotelbeds_room_embeddings.json
    - data/embeddings/axisdata_hotel_embeddings.json
    - data/embeddings/axisdata_room_embeddings.json
    """
    
    logger.info("=" * 80)
    logger.info("EMBEDDING SERVICE - STARTING")
    logger.info("=" * 80)
    
    try:
        # Initialize embedding service (auto device detection)
        embedding_service = EmbeddingService(
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            device=None,  # Auto-detect
        )
        
        # Initialize generators and persistence
        embeddings_generator = HotelRoomEmbeddingsGenerator(embedding_service)
        persistence = EmbeddingsPersistence()
        data_loader = NormalizedDataLoader()
        
        # Process Hotelbeds
        logger.info("\n" + "-" * 80)
        logger.info("PROCESSING HOTELBEDS")
        logger.info("-" * 80)
        
        if NORMALIZED_HOTELBEDS_PATH.exists():
            hotelbeds_data = data_loader.load_normalized_data(NORMALIZED_HOTELBEDS_PATH)
            hotelbeds_embeddings = embeddings_generator.generate_embeddings(
                hotelbeds_data,
                provider="hotelbeds"
            )
            hb_hotel_path, hb_room_path = persistence.save_embeddings(
                hotelbeds_embeddings,
                EMBEDDINGS_OUTPUT_DIR,
            )
            logger.info(f"✓ Hotelbeds embeddings saved")
        else:
            logger.warning(f"Hotelbeds normalized data not found: {NORMALIZED_HOTELBEDS_PATH}")
        
        # Process AxisData
        logger.info("\n" + "-" * 80)
        logger.info("PROCESSING AXISDATA")
        logger.info("-" * 80)
        
        if NORMALIZED_AXISDATA_PATH.exists():
            axisdata_data = data_loader.load_normalized_data(NORMALIZED_AXISDATA_PATH)
            axisdata_embeddings = embeddings_generator.generate_embeddings(
                axisdata_data,
                provider="axisdata"
            )
            ad_hotel_path, ad_room_path = persistence.save_embeddings(
                axisdata_embeddings,
                EMBEDDINGS_OUTPUT_DIR,
            )
            logger.info(f"✓ AxisData embeddings saved")
        else:
            logger.warning(f"AxisData normalized data not found: {NORMALIZED_AXISDATA_PATH}")
        
        logger.info("\n" + "=" * 80)
        logger.info("EMBEDDING SERVICE - COMPLETE")
        logger.info("=" * 80)
        logger.info("\nOutput embeddings files (ready for hotel_matcher.py and room_matcher.py):")
        logger.info(f"  - {EMBEDDINGS_OUTPUT_DIR}/hotelbeds_hotel_embeddings.json")
        logger.info(f"  - {EMBEDDINGS_OUTPUT_DIR}/hotelbeds_room_embeddings.json")
        logger.info(f"  - {EMBEDDINGS_OUTPUT_DIR}/axisdata_hotel_embeddings.json")
        logger.info(f"  - {EMBEDDINGS_OUTPUT_DIR}/axisdata_room_embeddings.json")
        logger.info("\nNext steps:")
        logger.info("  1. Use hotel_matcher.py to match hotels by place_id with embeddings")
        logger.info("  2. Use room_matcher.py to match rooms for each matched hotel")
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())