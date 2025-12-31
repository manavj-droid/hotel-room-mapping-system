"""Configuration settings for the hotel room mapping system."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs"
LOGS_DIR = DATA_DIR / "logs"

# Input files
HOTELBEDS_HOTELS_FILE = INPUT_DIR / "hotelbeds_hotels.json"
AXISDATA_HOTELS_FILE = INPUT_DIR / "axisdata_hotels.json"
HOTELBEDS_ROOM_TYPES_FILE = INPUT_DIR / "hotelbeds_room_types.json"

# Output files
HOTEL_ROOM_MAPPING_FILE = OUTPUT_DIR / "hotel_room_mapping.json"

# Log files
MAPPING_AUDIT_FILE = LOGS_DIR / "mapping_audit.jsonl"
MAPPING_ERRORS_FILE = LOGS_DIR / "mapping_errors.log"

# Matching thresholds
HOTEL_NAME_SIMILARITY_THRESHOLD = 0.85
ROOM_SEMANTIC_SIMILARITY_THRESHOLD = 0.80
EMBEDDING_CACHE_DIR = PROJECT_ROOT / ".embeddings_cache"

# Embedding service configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create necessary directories if they don't exist
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR, EMBEDDING_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
